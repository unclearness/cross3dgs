package org.unclearness.cross3dgs.capture

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import java.io.File

/**
 * Extracts still frames from a video for SfM input.
 *
 * Sampling strategy (initial implementation): the video is divided into
 * [targetFrames] equal time windows; within each window several candidate
 * frames are decoded and the sharpest one (largest luma-gradient variance) is
 * kept. Uniform sampling keeps view overlap predictable for sequential
 * matching, and per-window sharpness selection discards motion-blurred frames,
 * which are the main failure source with handheld video.
 *
 * Alternatives worth evaluating later (documented in README): fixed-fps
 * sampling, keyframe(sync-sample)-only sampling for speed, and feature-based
 * selection (e.g. skip frames with too little/too much parallax against the
 * previously kept frame).
 */
object VideoFrameExtractor {

    private const val CANDIDATES_PER_FRAME = 2
    private const val MAX_DECODE_DIM = 1920

    /**
     * Returns the number of frames written to [outDir] as frame_%04d.jpg.
     * [onProgress] is called with (done, total) from the calling thread.
     *
     * When [poses] is given (ARCore capture), per-frame pose priors are
     * interpolated at each kept frame's timestamp and written to [priorsOut]
     * in the format consumed by the native pipeline (see
     * libs/cross3dgs_sfm/src/cross3dgs_sfm.cc).
     */
    fun extract(
        context: Context,
        source: Uri,
        outDir: File,
        targetFrames: Int,
        poses: PosesFile? = null,
        priorsOut: File? = null,
        onProgress: (Int, Int) -> Unit,
    ): Int {
        val retriever = MediaMetadataRetriever()
        try {
            if (source.scheme == null || source.scheme == "file") {
                retriever.setDataSource(source.path)
            } else {
                retriever.setDataSource(context, source)
            }
            val durationMs = retriever
                .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: return 0
            if (durationMs <= 0) return 0

            check(!outDir.exists() || outDir.listFiles().orEmpty().isEmpty()) {
                "frame output already exists; create a new run instead of overwriting"
            }
            check(outDir.mkdirs() || outDir.isDirectory) {
                "failed to create frame output directory"
            }
            check(priorsOut == null || !priorsOut.exists()) {
                "frame-prior output already exists; refusing to overwrite"
            }

            val priorLines = StringBuilder()
            var priorCount = 0
            var decodedSize: IntArray? = null

            var written = 0
            val slots = targetFrames * CANDIDATES_PER_FRAME
            for (i in 0 until targetFrames) {
                var best: Bitmap? = null
                var bestScore = -1.0
                var bestTimeUs = 0L
                for (j in 0 until CANDIDATES_PER_FRAME) {
                    val slot = i * CANDIDATES_PER_FRAME + j
                    val timeUs = ((slot + 0.5) / slots * durationMs * 1000).toLong()
                    val bmp = decodeScaled(retriever, timeUs) ?: continue
                    val score = sharpness(bmp)
                    if (score > bestScore) {
                        best?.recycle()
                        best = bmp
                        bestScore = score
                        bestTimeUs = timeUs
                    } else {
                        bmp.recycle()
                    }
                }
                best?.let { bmp ->
                    val name = "frame_%04d.jpg".format(i)
                    File(outDir, name).outputStream().use {
                        bmp.compress(Bitmap.CompressFormat.JPEG, 92, it)
                    }
                    if (decodedSize == null) decodedSize = intArrayOf(bmp.width, bmp.height)
                    bmp.recycle()
                    written++
                    if (poses != null) {
                        val tAbs = poses.videoStartNs + bestTimeUs * 1000
                        poses.priorAt(tAbs)?.let { p ->
                            priorLines.append(name)
                                .append(' ').append(p.position.joinToString(" "))
                                .append(' ').append(p.gravityCam.joinToString(" "))
                                .append(' ').append(p.sigma).append('\n')
                            priorCount++
                        }
                    }
                }
                onProgress(i + 1, targetFrames)
            }

            if (poses != null && priorsOut != null && priorCount > 0) {
                val header = StringBuilder()
                val intr = poses.intrinsics
                val isz = poses.intrinsicsSize
                val dsz = decodedSize
                if (intr != null && isz != null && dsz != null) {
                    // scale capture-resolution intrinsics to the decoded size
                    val sx = dsz[0].toDouble() / isz[0]
                    val sy = dsz[1].toDouble() / isz[1]
                    header.append("# intrinsics ")
                        .append(intr[0] * sx).append(' ').append(intr[1] * sy)
                        .append(' ').append(intr[2] * sx).append(' ').append(intr[3] * sy)
                        .append(' ').append(dsz[0]).append(' ').append(dsz[1]).append('\n')
                }
                priorsOut.writeText(header.toString() + priorLines.toString())
            } else {
                check(priorsOut == null || !priorsOut.exists()) {
                    "unexpected pre-existing frame-prior output"
                }
            }
            return written
        } finally {
            retriever.release()
        }
    }

    private fun decodeScaled(retriever: MediaMetadataRetriever, timeUs: Long): Bitmap? {
        val w = retriever.extractMetadata(
            MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
        val h = retriever.extractMetadata(
            MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
        return if (w > MAX_DECODE_DIM || h > MAX_DECODE_DIM) {
            val scale = MAX_DECODE_DIM.toFloat() / maxOf(w, h)
            retriever.getScaledFrameAtTime(
                timeUs,
                MediaMetadataRetriever.OPTION_CLOSEST,
                (w * scale).toInt(),
                (h * scale).toInt())
        } else {
            retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST)
        }
    }

    /** Variance of the horizontal/vertical luma gradient on a downscaled copy. */
    private fun sharpness(bmp: Bitmap): Double {
        val w = 160
        val h = maxOf(1, bmp.height * w / maxOf(1, bmp.width))
        val small = Bitmap.createScaledBitmap(bmp, w, h, true)
        val pixels = IntArray(w * h)
        small.getPixels(pixels, 0, w, 0, 0, w, h)
        small.recycle()
        val luma = DoubleArray(w * h)
        for (idx in pixels.indices) {
            val p = pixels[idx]
            luma[idx] = 0.299 * ((p shr 16) and 0xFF) +
                0.587 * ((p shr 8) and 0xFF) + 0.114 * (p and 0xFF)
        }
        var sum = 0.0
        var sumSq = 0.0
        var n = 0
        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                val g = Math.abs(luma[y * w + x + 1] - luma[y * w + x - 1]) +
                    Math.abs(luma[(y + 1) * w + x] - luma[(y - 1) * w + x])
                sum += g
                sumSq += g * g
                n++
            }
        }
        if (n == 0) return 0.0
        val mean = sum / n
        return sumSq / n - mean * mean
    }
}
