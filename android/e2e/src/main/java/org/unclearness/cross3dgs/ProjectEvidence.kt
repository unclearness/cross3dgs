package org.unclearness.cross3dgs

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.view.PixelCopy
import android.view.SurfaceView
import android.view.View
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

/** Project-local provenance and automatic per-stage screenshots. */
object ProjectEvidence {
    private val inventoryRoots = listOf(
        "input/capture", "extract", "sfm", "training", "output",
        "screenshots", "pipeline.log", "telemetry.csv",
    )

    fun environment(activity: Activity): Map<String, String> {
        val packageInfo = activity.packageManager.getPackageInfo(activity.packageName, 0)
        return linkedMapOf(
            "manufacturer" to Build.MANUFACTURER,
            "model" to Build.MODEL,
            "device" to Build.DEVICE,
            "android_sdk" to Build.VERSION.SDK_INT.toString(),
            "android_release" to Build.VERSION.RELEASE,
            "build_fingerprint" to Build.FINGERPRINT,
            "application_id" to activity.packageName,
            "application_version" to (packageInfo.versionName ?: "unknown"),
        )
    }

    fun pipelineConfig(
        frameTargetCount: Int = 60,
        optimizationConfig: OptimizationConfig = OptimizationConfig(),
        preset: OptimizationPreset,
        tileBinCapacity: Int,
        sfmLivePreview: Boolean = true,
        sfmThreads: String = "default",
        sfmCpus: String = "all",
    ): Map<String, String> = linkedMapOf(
        "frame_extractor" to "uniform_nx2_candidates_sharpest_luma_v2",
        "frame_target_count" to frameTargetCount.toString(),
        "frame_candidates_per_target" to "2",
        "frame_max_decode_dimension" to "1920",
        "frame_jpeg_quality" to "92",
        "sfm" to "incremental_colmap_cpu_sift8192_single_camera_v1",
        "sfm_feature_max_image_size" to "2400",
        "sfm_max_sift_features" to "8192",
        "sfm_pairing" to "exhaustive_le_100_else_sequential",
        "sfm_exhaustive_block_size" to "20",
        "sfm_sequential_overlap" to "15",
        "sfm_init_min_num_inliers" to "50",
        "sfm_position_priors" to "enabled_when_present",
        "sfm_gravity_priors" to "recorded_not_full_6dof_initialization",
        // Display-only observer (read-only DB queries, reconstruction copies);
        // it shares the CPU with SfM, so it is recorded for timing.
        "sfm_live_preview" to
            if (sfmLivePreview) "library_rendered_max_4fps" else "disabled",
        // Worker threads "extraction,matching,mapping" (0 = one per core);
        // timing only, features do not depend on it.
        "sfm_threads" to sfmThreads,
        // CPUs the SfM workers may run on ("all" or e.g. "4-7"); timing only.
        "sfm_cpus" to sfmCpus,
        "optimizer_initialization" to "colmap_sparse_points",
        "optimizer_training_view_seed" to "42",
        // LoadColmapScene's default split, shared with the desktop/benchmark
        // path: images.bin sorted by name, zero-based index % 8 == 0 is held
        // out of training and used for the held-out evaluation in bench.json.
        "optimizer_eval_interval" to "8",
        "optimizer_holdout_rule" to
            "sorted_image_index_modulo_eval_interval_zero_based",
        "optimizer_holdout_evaluator" to
            "native_psnr_plus_render_target_png_for_host_rescoring",
        "optimizer_iterations" to optimizationConfig.iterations.toString(),
        "optimizer_gaussian_capacity" to
            optimizationConfig.gaussianCapacity.toString(),
        "optimizer_sh_degree" to optimizationConfig.shDegree.toString(),
        "optimizer_density_strategy" to optimizationConfig.densityStrategy,
        "optimizer_precision" to "fp32",
        // The quality recipe and the pair budget are what separate this path
        // from the benchmark presets, so a run records both rather than
        // leaving them to be inferred from the preset name.
        "optimizer_preset" to preset.name.lowercase(),
        "optimizer_full_image_ssim" to preset.fullImageSsim.toString(),
        "optimizer_mcmc_clone_offset" to preset.mcmcCloneOffset.toString(),
        "optimizer_mcmc_gradient_sampling" to
            preset.mcmcGradientSampling.toString(),
        "optimizer_tile_bin_capacity" to tileBinCapacity.toString(),
    )

    fun inventory(workspace: ProjectWorkspace): Map<String, String> {
        val files = inventoryRoots.flatMap { relative ->
            val item = File(workspace.root, relative)
            when {
                item.isFile -> listOf(item)
                item.isDirectory -> item.walkTopDown().filter(File::isFile).toList()
                else -> emptyList()
            }
        }.distinct().filterNot {
            it == workspace.manifestFile || it.name.endsWith(".pending")
        }.sortedBy { it.relativeTo(workspace.root).invariantSeparatorsPath }
        return files.associate { file ->
            file.relativeTo(workspace.root).invariantSeparatorsPath to sha256(file)
        }
    }

    fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    /**
     * Captures the window UI and, when supplied, the SurfaceView's independently
     * queued buffer. A Window PixelCopy alone contains a black hole where a
     * SurfaceView is composed on affected Android devices.
     */
    fun captureWindow(
        activity: Activity,
        output: File,
        timeoutMs: Long = DEFAULT_SCREENSHOT_TIMEOUT_MS,
        surfaceView: SurfaceView? = null,
        fallbackSurfaceImage: File? = null,
        requireSurfaceContent: Boolean = false,
        preferFallbackSurfaceImage: Boolean = false,
        overlayViews: List<View> = emptyList(),
        onComplete: (Boolean) -> Unit,
    ) {
        require(timeoutMs > 0L) { "screenshot timeout must be positive" }
        val handler = Handler(Looper.getMainLooper())
        val completed = AtomicBoolean(false)
        val pending = File(output.absolutePath + ".pending")
        var timeout: Runnable? = null
        fun finish(ok: Boolean): Boolean {
            if (!completed.compareAndSet(false, true)) return false
            timeout?.let { handler.removeCallbacks(it) }
            if (!ok) {
                runCatching { output.delete() }
                runCatching { pending.delete() }
            }
            if (Looper.myLooper() == Looper.getMainLooper()) {
                onComplete(ok)
            } else {
                handler.post { onComplete(ok) }
            }
            return true
        }
        val view = activity.window.decorView
        if (view.width <= 0 || view.height <= 0) {
            finish(false)
            return
        }
        output.parentFile?.mkdirs()
        if (output.exists() || pending.exists()) {
            onComplete(false)
            return
        }
        val bitmap = try {
            Bitmap.createBitmap(view.width, view.height, Bitmap.Config.ARGB_8888)
        } catch (_: Throwable) {
            finish(false)
            return
        }
        timeout = Runnable { finish(false) }.also {
            handler.postDelayed(it, timeoutMs)
        }
        fun encodeAndFinish() {
            thread(name = "capture-window-png", isDaemon = true) {
                var moved = false
                try {
                    if (!completed.get()) {
                        val compressed = pending.outputStream().use {
                            bitmap.compress(Bitmap.CompressFormat.PNG, 100, it)
                        }
                        if (compressed && pending.length() > 0L &&
                            !completed.get() && !output.exists()) {
                            moved = pending.renameTo(output)
                        }
                    }
                } catch (_: Throwable) {
                    moved = false
                } finally {
                    bitmap.recycle()
                }
                if (moved) {
                    if (!finish(true)) runCatching { output.delete() }
                } else {
                    finish(false)
                }
                runCatching { pending.delete() }
            }
        }

        fun composeSurface(surfaceBitmap: Bitmap?) {
            if (completed.get()) {
                surfaceBitmap?.recycle()
                bitmap.recycle()
                return
            }
            var source = surfaceBitmap
            val directHasContent = source?.let(::hasImageContent) == true
            if ((preferFallbackSurfaceImage && fallbackSurfaceImage?.isFile == true) ||
                source == null ||
                (!directHasContent && fallbackSurfaceImage?.isFile == true)) {
                source?.recycle()
                source = fallbackSurfaceImage?.takeIf(File::isFile)?.let { file ->
                    runCatching { BitmapFactory.decodeFile(file.absolutePath) }.getOrNull()
                }
            }
            if (surfaceView != null) {
                if (source == null || (requireSurfaceContent && !hasImageContent(source))) {
                    source?.recycle()
                    bitmap.recycle()
                    finish(false)
                    return
                }
                val decorLocation = IntArray(2)
                val surfaceLocation = IntArray(2)
                view.getLocationInWindow(decorLocation)
                surfaceView.getLocationInWindow(surfaceLocation)
                val left = (surfaceLocation[0] - decorLocation[0]).toFloat()
                val top = (surfaceLocation[1] - decorLocation[1]).toFloat()
                val canvas = Canvas(bitmap)
                canvas.save()
                canvas.clipRect(
                    left, top,
                    left + surfaceView.width, top + surfaceView.height,
                )
                canvas.drawColor(Color.BLACK)
                val scale = minOf(
                    surfaceView.width.toFloat() / source.width,
                    surfaceView.height.toFloat() / source.height,
                )
                val drawWidth = source.width * scale
                val drawHeight = source.height * scale
                val drawLeft = left + (surfaceView.width - drawWidth) * 0.5f
                val drawTop = top + (surfaceView.height - drawHeight) * 0.5f
                canvas.drawBitmap(
                    source, null,
                    RectF(
                        drawLeft, drawTop,
                        drawLeft + drawWidth, drawTop + drawHeight,
                    ),
                    Paint(Paint.FILTER_BITMAP_FLAG),
                )
                canvas.restore()
                for (overlay in overlayViews) {
                    if (!overlay.isShown || overlay.width <= 0 || overlay.height <= 0) {
                        continue
                    }
                    val overlayLocation = IntArray(2)
                    overlay.getLocationInWindow(overlayLocation)
                    canvas.save()
                    canvas.translate(
                        (overlayLocation[0] - decorLocation[0]).toFloat(),
                        (overlayLocation[1] - decorLocation[1]).toFloat(),
                    )
                    overlay.draw(canvas)
                    canvas.restore()
                }
            }
            source?.recycle()
            encodeAndFinish()
        }

        fun captureSurfaceOrFallback() {
            val sourceView = surfaceView
            if (sourceView == null) {
                composeSurface(null)
                return
            }
            if (preferFallbackSurfaceImage && fallbackSurfaceImage?.isFile == true) {
                composeSurface(null)
                return
            }
            if (sourceView.width <= 0 || sourceView.height <= 0 ||
                !sourceView.holder.surface.isValid) {
                composeSurface(null)
                return
            }
            val surfaceBitmap = try {
                Bitmap.createBitmap(
                    sourceView.width, sourceView.height, Bitmap.Config.ARGB_8888)
            } catch (_: Throwable) {
                composeSurface(null)
                return
            }
            try {
                PixelCopy.request(sourceView, surfaceBitmap, { result ->
                    if (result == PixelCopy.SUCCESS) {
                        composeSurface(surfaceBitmap)
                    } else {
                        surfaceBitmap.recycle()
                        composeSurface(null)
                    }
                }, handler)
            } catch (_: Throwable) {
                surfaceBitmap.recycle()
                composeSurface(null)
            }
        }

        fun captureViewHierarchyFallback() {
            runCatching {
                val canvas = Canvas(bitmap)
                view.draw(canvas)
            }.onSuccess {
                captureSurfaceOrFallback()
            }.onFailure {
                bitmap.recycle()
                finish(false)
            }
        }

        try {
            PixelCopy.request(activity.window, bitmap, { result ->
                if (completed.get()) {
                    bitmap.recycle()
                    return@request
                }
                if (result == PixelCopy.SUCCESS) {
                    captureSurfaceOrFallback()
                } else {
                    captureViewHierarchyFallback()
                }
            }, handler)
        } catch (_: Throwable) {
            captureViewHierarchyFallback()
        }
    }

    private fun hasImageContent(bitmap: Bitmap): Boolean {
        if (bitmap.width <= 0 || bitmap.height <= 0) return false
        val xStep = maxOf(1, bitmap.width / 64)
        val yStep = maxOf(1, bitmap.height / 64)
        var sampled = 0
        var nonBlack = 0
        var lumaMin = 255
        var lumaMax = 0
        var lumaSum = 0.0
        var lumaSquaredSum = 0.0
        var y = 0
        while (y < bitmap.height) {
            var x = 0
            while (x < bitmap.width) {
                val pixel = bitmap.getPixel(x, y)
                val luminance = (
                    54 * Color.red(pixel) +
                        183 * Color.green(pixel) +
                        19 * Color.blue(pixel)
                    ) shr 8
                if (luminance > 8) nonBlack++
                lumaMin = minOf(lumaMin, luminance)
                lumaMax = maxOf(lumaMax, luminance)
                lumaSum += luminance
                lumaSquaredSum += luminance.toDouble() * luminance
                sampled++
                x += xStep
            }
            y += yStep
        }
        if (sampled == 0 || nonBlack.toDouble() / sampled < 0.01) return false
        val mean = lumaSum / sampled
        val variance = (lumaSquaredSum / sampled - mean * mean).coerceAtLeast(0.0)
        return lumaMax - lumaMin >= 16 && kotlin.math.sqrt(variance) >= 2.0
    }

    private const val DEFAULT_SCREENSHOT_TIMEOUT_MS = 12_000L
}
