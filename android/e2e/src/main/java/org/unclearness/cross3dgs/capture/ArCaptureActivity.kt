package org.unclearness.cross3dgs.capture

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.RectF
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.view.Gravity
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Coordinates2d
import com.google.ar.core.Frame
import com.google.ar.core.ImageMetadata
import com.google.ar.core.RecordingConfig
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import java.io.File
import java.io.BufferedWriter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.EnumSet
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.roundToInt
import kotlin.concurrent.thread

/**
 * ARCore capture: records the camera stream to a standard MP4 (Recording API)
 * while logging per-frame VIO poses + intrinsics to a *.arposes.txt file.
 * Landscape-locked so the recorded video
 * carries no rotation metadata and intrinsics match the decoded frames.
 *
 * The caller supplies a newly-created empty capture directory. This activity
 * writes one immutable video/pose/sensor set there and refuses any overwrite.
 */
class ArCaptureActivity : AppCompatActivity(), GLSurfaceView.Renderer,
    SensorEventListener {

    companion object {
        const val EXTRA_VIDEO_PATH = "video_path"
        const val EXTRA_POSES_PATH = "poses_path"
        const val EXTRA_CAPTURE_ROOT = "capture_root"
        const val EXTRA_SCREENSHOT_PATH = "screenshot_path"
        const val EXTRA_SENSOR_PATH = "sensor_path"
        const val EXTRA_CAPTURE_DURATION_MS = "capture_duration_ms"
        const val EXTRA_POSE_COUNT = "pose_count"
        const val EXTRA_SENSOR_COUNT = "sensor_count"
        private const val PERMISSION_REQUEST = 71
        private const val FOCUS_METADATA_SAMPLE_INTERVAL = 5
        private const val CAPTURE_EVIDENCE_TIMEOUT_MS = 30_000L
        /** Recorded frames before the live capture image (about half a second). */
        private const val LIVE_EVIDENCE_FRAME = 15
        // Camera frames to keep the preview covered once it runs.
        private const val COVER_FRAMES = 10
    }

    private var session: Session? = null
    private lateinit var surfaceView: GLSurfaceView
    private lateinit var recordButton: Button
    // Hides the preview while the camera opens: until its first frames arrive
    // the surface shows black and then unwritten (green) images.
    private lateinit var cameraCover: TextView
    private val coverFramesLeft = AtomicInteger(COVER_FRAMES)
    private lateinit var statusLabel: TextView

    private var cameraTexture = -1
    private var program = 0
    private var quadCoords: FloatBuffer = direct(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f))
    private var quadTexCoords: FloatBuffer = direct(FloatArray(8))
    private var viewportWidth = 1
    private var viewportHeight = 1
    private var displayGeometryDirty = true

    @Volatile private var recording = false
    private val poseLines = StringBuilder()
    private var numPoses = 0
    private var videoStartNs = -1L
    private var intrinsicsLine: String? = null
    private var cameraConfigLine: String? = null
    private var captureImageSize = "unknown"
    private var autoFocusSupported: Boolean? = null
    private var focusDistanceCalibration: Int? = null
    private var actualAfMode: Int? = null
    private var actualAfState: Int? = null
    private var lensFocusDistance: Float? = null
    private var focusMetadataFrameIndex = 0
    private var recordingStartedElapsedMs = 0L
    private lateinit var sensorManager: SensorManager
    private var sensorWriter: BufferedWriter? = null
    private var sensorSamples = 0

    // capture.png is the screen shortly after recording starts: the frame the
    // render thread just drew, read back with glReadPixels (a PixelCopy of an
    // ARCore GLSurfaceView is black on some devices), with the status and
    // record-button overlays drawn over it. UI thread, except where noted.
    private enum class LiveEvidence { NONE, PENDING, DONE, FAILED }
    private var liveEvidence = LiveEvidence.NONE
    private var liveEvidenceAbandoned = false
    private var onLiveEvidence: ((Boolean) -> Unit)? = null
    private val liveEvidenceRequested = AtomicBoolean(false)
    private var recordedFrames = 0  // render thread

    private lateinit var captureRoot: File
    private val videoFile: File get() = File(captureRoot, "video.mp4")
    private val posesFile: File get() = File(captureRoot, "arcore_poses.txt")
    private val sensorsFile: File get() = File(captureRoot, "sensors.csv")
    private val screenshotFile: File get() = File(captureRoot, "capture.png")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        captureRoot = intent.getStringExtra(EXTRA_CAPTURE_ROOT)?.let(::File)
            ?: error("immutable capture output directory is required")
        require(captureRoot.isDirectory &&
            captureRoot.listFiles().orEmpty().isEmpty()) {
            "capture output must be a new empty directory: $captureRoot"
        }
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager

        surfaceView = GLSurfaceView(this).apply {
            setEGLContextClientVersion(2)
            setRenderer(this@ArCaptureActivity)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        }
        recordButton = Button(this).apply {
            text = "Start recording"
            setOnClickListener { toggleRecording() }
        }
        statusLabel = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt())
            setBackgroundColor(0x80000000.toInt())
            setPadding(16, 8, 16, 8)
            text = "initializing"
        }
        cameraCover = TextView(this).apply {
            setTextColor(0xFFFFFFFF.toInt())
            setBackgroundColor(0xFF000000.toInt())
            gravity = Gravity.CENTER
            text = "Starting camera..."
        }
        val root = FrameLayout(this)
        root.addView(surfaceView)
        root.addView(cameraCover)
        root.addView(statusLabel, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.TOP or Gravity.START))
        root.addView(recordButton, FrameLayout.LayoutParams(
            FrameLayout.LayoutParams.WRAP_CONTENT, FrameLayout.LayoutParams.WRAP_CONTENT,
            Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL).apply { bottomMargin = 48 })
        // The status bar and the cutout sit over this screen, so the overlays
        // are inset out from under them. The preview itself stays full-bleed:
        // padding the root would letterbox the camera image.
        ViewCompat.setOnApplyWindowInsetsListener(root) { _, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or
                    WindowInsetsCompat.Type.displayCutout(),
            )
            (statusLabel.layoutParams as FrameLayout.LayoutParams).apply {
                leftMargin = bars.left + 16
                topMargin = bars.top + 16
                rightMargin = bars.right + 16
            }
            (recordButton.layoutParams as FrameLayout.LayoutParams).apply {
                bottomMargin = bars.bottom + 48
            }
            statusLabel.requestLayout()
            recordButton.requestLayout()
            insets
        }
        setContentView(root)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), PERMISSION_REQUEST)
        }
    }

    override fun onRequestPermissionsResult(
        code: Int, permissions: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(code, permissions, results)
        if (code == PERMISSION_REQUEST &&
            (results.isEmpty() || results[0] != PackageManager.PERMISSION_GRANTED)) {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) return
        if (session == null) {
            when (ArCoreApk.getInstance().requestInstall(this, true)) {
                ArCoreApk.InstallStatus.INSTALL_REQUESTED -> return
                ArCoreApk.InstallStatus.INSTALLED -> {}
            }
            try {
                session = Session(this)
                session?.let(::configureCaptureCamera)
            } catch (e: Exception) {
                Toast.makeText(this, "Cannot initialize ARCore: ${e.message}",
                    Toast.LENGTH_LONG).show()
                finish()
                return
            }
        }
        try {
            session?.resume()
        } catch (e: Exception) {
            Toast.makeText(this, "Cannot open the camera: ${e.message}", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        surfaceView.onResume()
        displayGeometryDirty = true
    }

    override fun onPause() {
        super.onPause()
        if (recording) stopRecordingAndFinish(abort = true)
        surfaceView.onPause()
        session?.pause()
    }

    override fun onDestroy() {
        session?.close()
        session = null
        super.onDestroy()
    }

    private fun toggleRecording() {
        val s = session ?: return
        if (recording) {
            stopRecordingAndFinish(abort = false)
            return
        }
        try {
            check(!videoFile.exists() && !posesFile.exists() &&
                !sensorsFile.exists() && !screenshotFile.exists()) {
                "capture output already exists; refusing to overwrite"
            }
            poseLines.clear()
            numPoses = 0
            sensorSamples = 0
            videoStartNs = -1
            intrinsicsLine = null
            recordedFrames = 0
            liveEvidence = LiveEvidence.NONE
            liveEvidenceAbandoned = false
            liveEvidenceRequested.set(false)
            // Started on the running session: pausing it would close and
            // reopen the camera, which stalls the preview.
            val config = RecordingConfig(s)
                .setMp4DatasetUri(Uri.fromFile(videoFile))
                .setAutoStopOnPause(false)
            s.startRecording(config)
            recording = true
            recordingStartedElapsedMs = SystemClock.elapsedRealtime()
            startSensorRecording()
            runOnUiThread { recordButton.text = "Stop recording" }
        } catch (e: Exception) {
            Toast.makeText(this, "Could not start recording: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun stopRecordingAndFinish(abort: Boolean) {
        val s = session ?: return
        recording = false
        stopSensorRecording()
        try {
            s.stopRecording()
        } catch (e: Exception) {
            // keep going; the dataset may still be usable
        }
        if (abort || numPoses < 10 || videoStartNs < 0) {
            // A live capture image of a discarded recording is not evidence.
            liveEvidenceAbandoned = true
            onLiveEvidence = null
            runCatching { screenshotFile.delete() }
            if (!abort) {
                Toast.makeText(this, "Too few tracked frames",
                    Toast.LENGTH_LONG).show()
            }
            setResult(Activity.RESULT_CANCELED)
            finish()
            return
        }
        val header = StringBuilder()
            .append("# cross3dgs ARCore capture v1\n")
            .append("# video_start_ns ").append(videoStartNs).append('\n')
        intrinsicsLine?.let { header.append(it).append('\n') }
        cameraConfigLine?.let { header.append(it).append('\n') }
        header.append("# gravity_world 0 -1 0\n")
            .append("# focus_format: timestamp_ns af_mode af_state lens_focus_distance\n")
            .append("# format: timestamp_ns tx ty tz qx qy qz qw (world_from_camera)\n")
        posesFile.writeText(header.toString() + poseLines.toString())
        val captureDurationMs =
            (SystemClock.elapsedRealtime() - recordingStartedElapsedMs).coerceAtLeast(0L)
        recordButton.isEnabled = false
        recordButton.text = "Saving..."
        fun publish() {
            setResult(Activity.RESULT_OK, intent
                .putExtra(EXTRA_CAPTURE_ROOT, captureRoot.absolutePath)
                .putExtra(EXTRA_VIDEO_PATH, videoFile.absolutePath)
                .putExtra(EXTRA_POSES_PATH, posesFile.absolutePath)
                .apply {
                    if (sensorsFile.isFile) {
                        putExtra(EXTRA_SENSOR_PATH, sensorsFile.absolutePath)
                    }
                }
                .putExtra(EXTRA_CAPTURE_DURATION_MS, captureDurationMs)
                .putExtra(EXTRA_POSE_COUNT, numPoses)
                .putExtra(EXTRA_SENSOR_COUNT, sensorSamples)
                .putExtra(EXTRA_SCREENSHOT_PATH, screenshotFile.absolutePath))
            finish()
        }
        when (liveEvidence) {
            LiveEvidence.DONE -> publish()
            LiveEvidence.PENDING -> onLiveEvidence = { captured ->
                if (captured) publish() else saveRecordedFrameEvidence(::publish)
            }
            else -> saveRecordedFrameEvidence(::publish)
        }
    }

    /**
     * Fallback when the live screen copy failed: capture.png from the MP4
     * itself, with the capture status drawn over it.
     */
    private fun saveRecordedFrameEvidence(publish: () -> Unit) {
        runCatching { screenshotFile.delete() }
        statusLabel.text = "Capture: Recorded video and ARCore poses · $numPoses poses\n" +
            "$captureImageSize  ${focusStatusText()}"
        val fallback = File(
            cacheDir,
            "capture_to_view_camera_frames/${captureRoot.name}/capture_source.png",
        )
        val evidenceCompleted = AtomicBoolean(false)
        val evidenceHandler = Handler(Looper.getMainLooper())
        lateinit var evidenceTimeout: Runnable
        fun completeEvidence(captured: Boolean) {
            if (!evidenceCompleted.compareAndSet(false, true)) return
            evidenceHandler.removeCallbacks(evidenceTimeout)
            runCatching { fallback.delete() }
            if (!captured) {
                runCatching { screenshotFile.delete() }
                Toast.makeText(
                    this, "Could not save the capture image", Toast.LENGTH_LONG).show()
                setResult(Activity.RESULT_CANCELED)
                finish()
                return
            }
            publish()
        }
        evidenceTimeout = Runnable { completeEvidence(false) }
        evidenceHandler.postDelayed(evidenceTimeout, CAPTURE_EVIDENCE_TIMEOUT_MS)
        thread(name = "capture-evidence") {
            val fallbackReady = runCatching {
                fallback.parentFile?.mkdirs()
                saveVideoFrameEvidence(fallback)
            }.getOrDefault(false)
            runOnUiThread {
                if (evidenceCompleted.get()) {
                    runCatching { fallback.delete() }
                    return@runOnUiThread
                }
                if (!fallbackReady || isFinishing || isDestroyed) {
                    completeEvidence(false)
                    return@runOnUiThread
                }
                org.unclearness.cross3dgs.ProjectEvidence.captureWindow(
                    this,
                    screenshotFile,
                    surfaceView = surfaceView,
                    fallbackSurfaceImage = fallback,
                    requireSurfaceContent = true,
                    preferFallbackSurfaceImage = true,
                    overlayViews = listOf(statusLabel, recordButton),
                ) { captured ->
                    completeEvidence(captured)
                }
            }
        }
    }

    /**
     * Produces capture evidence from the MP4 stream itself. PixelCopy of an
     * ARCore GLSurfaceView is black on some devices even while the preview is
     * visible, so a successful window copy is not proof that camera pixels
     * were captured. Five temporal candidates are decoded and the sharpest is
     * letterboxed without aspect distortion into the capture window extent.
     */
    private fun saveVideoFrameEvidence(outputFile: File): Boolean {
        check(videoFile.isFile && videoFile.length() > 0)
        check(!outputFile.exists()) { "refusing to overwrite camera evidence" }
        val retriever = MediaMetadataRetriever()
        var selected: Bitmap? = null
        try {
            retriever.setDataSource(videoFile.absolutePath)
            val durationMs = retriever.extractMetadata(
                MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull()
                ?: return false
            var selectedScore = Double.NEGATIVE_INFINITY
            for (fraction in doubleArrayOf(0.2, 0.35, 0.5, 0.65, 0.8)) {
                val timeUs = (durationMs * fraction * 1000.0).toLong()
                val candidate = retriever.getFrameAtTime(
                    timeUs, MediaMetadataRetriever.OPTION_CLOSEST) ?: continue
                val score = bitmapSharpness(candidate)
                if (score > selectedScore) {
                    selected?.recycle()
                    selected = candidate
                    selectedScore = score
                } else {
                    candidate.recycle()
                }
            }
            val frame = selected ?: return false
            val outputWidth = viewportWidth.coerceAtLeast(frame.width)
            val outputHeight = viewportHeight.coerceAtLeast(frame.height)
            val output = Bitmap.createBitmap(
                outputWidth, outputHeight, Bitmap.Config.ARGB_8888)
            try {
                val canvas = Canvas(output)
                canvas.drawColor(Color.BLACK)
                val scale = minOf(
                    outputWidth.toFloat() / frame.width,
                    outputHeight.toFloat() / frame.height,
                )
                val drawWidth = frame.width * scale
                val drawHeight = frame.height * scale
                val left = (outputWidth - drawWidth) * 0.5f
                val top = (outputHeight - drawHeight) * 0.5f
                canvas.drawBitmap(
                    frame, null,
                    RectF(left, top, left + drawWidth, top + drawHeight), null)

                outputFile.parentFile?.mkdirs()
                return outputFile.outputStream().use { stream ->
                    output.compress(Bitmap.CompressFormat.PNG, 100, stream)
                }
            } finally {
                output.recycle()
            }
        } finally {
            selected?.recycle()
            retriever.release()
        }
    }

    private fun bitmapSharpness(bitmap: Bitmap): Double {
        val width = 160
        val height = maxOf(1, bitmap.height * width / maxOf(1, bitmap.width))
        val small = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val pixels = IntArray(width * height)
        try {
            small.getPixels(pixels, 0, width, 0, 0, width, height)
        } finally {
            small.recycle()
        }
        val luma = DoubleArray(pixels.size)
        for (index in pixels.indices) {
            val pixel = pixels[index]
            luma[index] = 0.299 * ((pixel shr 16) and 0xFF) +
                0.587 * ((pixel shr 8) and 0xFF) + 0.114 * (pixel and 0xFF)
        }
        var sum = 0.0
        var sumSquared = 0.0
        var count = 0
        for (y in 1 until height - 1) {
            for (x in 1 until width - 1) {
                val gradient =
                    kotlin.math.abs(luma[y * width + x + 1] - luma[y * width + x - 1]) +
                    kotlin.math.abs(luma[(y + 1) * width + x] - luma[(y - 1) * width + x])
                sum += gradient
                sumSquared += gradient * gradient
                count++
            }
        }
        if (count == 0) return 0.0
        val mean = sum / count
        return sumSquared / count - mean * mean
    }

    private fun startSensorRecording() {
        sensorsFile.parentFile?.mkdirs()
        sensorWriter = sensorsFile.bufferedWriter().also {
            it.write("timestamp_ns,sensor_type,accuracy,v0,v1,v2,v3,v4\n")
        }
        listOf(
            Sensor.TYPE_ACCELEROMETER,
            Sensor.TYPE_GYROSCOPE,
            Sensor.TYPE_MAGNETIC_FIELD,
            Sensor.TYPE_GRAVITY,
            Sensor.TYPE_ROTATION_VECTOR,
        ).forEach { type ->
            sensorManager.getDefaultSensor(type)?.let { sensor ->
                sensorManager.registerListener(this, sensor, SensorManager.SENSOR_DELAY_GAME)
            }
        }
    }

    private fun stopSensorRecording() {
        sensorManager.unregisterListener(this)
        synchronized(this) {
            runCatching { sensorWriter?.flush() }
            runCatching { sensorWriter?.close() }
            sensorWriter = null
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!recording) return
        synchronized(this) {
            val writer = sensorWriter ?: return
            val values = (0 until 5).joinToString(",") { index ->
                event.values.getOrNull(index)?.toString().orEmpty()
            }
            writer.write(
                "${event.timestamp},${event.sensor.type},${event.accuracy},$values\n")
            sensorSamples++
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    // ---- GLSurfaceView.Renderer ----

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        cameraTexture = textures[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTexture)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
            GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
            GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        program = buildProgram()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewportWidth = width
        viewportHeight = height
        displayGeometryDirty = true
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        val s = session ?: return
        try {
            s.setCameraTextureName(cameraTexture)
            if (displayGeometryDirty) {
                @Suppress("DEPRECATION")
                val rotation = windowManager.defaultDisplay.rotation
                s.setDisplayGeometry(rotation, viewportWidth, viewportHeight)
                displayGeometryDirty = false
            }
            val frame = s.update()
            if (frame.hasDisplayGeometryChanged()) {
                val ndc = direct(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f))
                frame.transformCoordinates2d(
                    Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES, ndc,
                    Coordinates2d.TEXTURE_NORMALIZED, quadTexCoords)
            }
            if (frame.timestamp > 0) {
                drawBackground()
            }

            val camera = frame.camera
            val tracking = camera.trackingState == TrackingState.TRACKING
            val focusSampled = updateFocusMetadata(frame)
            if (recording && videoStartNs < 0 && frame.timestamp > 0) {
                // Anchor to the first frame in the MP4: its pts 0 corresponds
                // to this timestamp. Recording starts on the running session,
                // and the first frame drawn after that is one frame period
                // before the file's first frame. The next drawn frame is no
                // reference: drawing stalls after startRecording for a time
                // that varies by device. Anchoring at the first *tracked*
                // frame instead shifted the whole timeline by the tracking
                // warm-up and broke the prior alignment on one device.
                videoStartNs = frame.timestamp +
                    1_000_000_000L / s.cameraConfig.fpsRange.upper
            }
            if (recording && frame.timestamp > 0) recordedFrames++
            if (frame.timestamp > 0 && coverFramesLeft.get() > 0 &&
                coverFramesLeft.decrementAndGet() == 0) {
                runOnUiThread { cameraCover.visibility = View.GONE }
            }
            if (recording && recordedFrames >= LIVE_EVIDENCE_FRAME &&
                liveEvidenceRequested.compareAndSet(false, true)) {
                val shown = runCatching { readFramebuffer() }.getOrNull()
                runOnUiThread { saveLiveEvidence(shown) }
            }
            if (recording && focusSampled) {
                poseLines.append("# focus ").append(frame.timestamp)
                    .append(' ').append(actualAfMode ?: -1)
                    .append(' ').append(actualAfState ?: -1)
                    .append(' ').append(lensFocusDistance ?: Float.NaN)
                    .append('\n')
            }
            if (recording && tracking && videoStartNs >= 0) {
                if (intrinsicsLine == null) {
                    // The Recording API stores the CPU image stream, so log the
                    // CPU-image intrinsics (whose dimensions match the MP4),
                    // NOT textureIntrinsics (GPU stream, different aspect).
                    val intr = camera.imageIntrinsics
                    val f = intr.focalLength
                    val c = intr.principalPoint
                    val d = intr.imageDimensions
                    intrinsicsLine = "# intrinsics ${f[0]} ${f[1]} ${c[0]} ${c[1]} ${d[0]} ${d[1]}"
                }
                val p = camera.pose
                poseLines.append(frame.timestamp)
                    .append(' ').append(p.tx()).append(' ').append(p.ty())
                    .append(' ').append(p.tz())
                    .append(' ').append(p.qx()).append(' ').append(p.qy())
                    .append(' ').append(p.qz()).append(' ').append(p.qw())
                    .append('\n')
                numPoses++
            }
            runOnUiThread {
                statusLabel.text = buildString {
                    append(if (recording) {
                        "Capture: Recording video and ARCore poses · $numPoses poses"
                    } else {
                        "Capture: Ready to record"
                    })
                    append('\n').append(if (tracking) "TRACKING" else camera.trackingState.name)
                    append("  ").append(captureImageSize)
                    append("  ").append(focusStatusText())
                }
            }
        } catch (e: Exception) {
            // camera not available yet, etc.
        }
    }

    /** The frame just drawn, top row first. Render thread. */
    private fun readFramebuffer(): Bitmap {
        val width = viewportWidth
        val height = viewportHeight
        val pixels = ByteBuffer.allocateDirect(width * height * 4)
            .order(ByteOrder.nativeOrder())
        GLES20.glReadPixels(
            0, 0, width, height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, pixels)
        val bottomUp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        try {
            pixels.rewind()
            bottomUp.copyPixelsFromBuffer(pixels)
            return Bitmap.createBitmap(
                bottomUp, 0, 0, width, height,
                Matrix().apply { preScale(1f, -1f) }, false)
        } finally {
            bottomUp.recycle()
        }
    }

    private fun saveLiveEvidence(shown: Bitmap?) {
        if (shown == null || !recording || isFinishing || isDestroyed) {
            shown?.recycle()
            finishLiveEvidence(false)
            return
        }
        liveEvidence = LiveEvidence.PENDING
        val frameFile = File(
            cacheDir,
            "capture_to_view_camera_frames/${captureRoot.name}/capture_live.png",
        )
        thread(name = "capture-live-evidence") {
            val written = runCatching {
                frameFile.parentFile?.mkdirs()
                frameFile.delete()
                frameFile.outputStream().use {
                    shown.compress(Bitmap.CompressFormat.PNG, 100, it)
                }
            }.getOrDefault(false)
            shown.recycle()
            runOnUiThread {
                if (!written || isFinishing || isDestroyed) {
                    runCatching { frameFile.delete() }
                    finishLiveEvidence(false)
                    return@runOnUiThread
                }
                org.unclearness.cross3dgs.ProjectEvidence.captureWindow(
                    this,
                    screenshotFile,
                    surfaceView = surfaceView,
                    fallbackSurfaceImage = frameFile,
                    requireSurfaceContent = true,
                    preferFallbackSurfaceImage = true,
                    overlayViews = listOf(statusLabel, recordButton),
                ) { captured ->
                    runCatching { frameFile.delete() }
                    finishLiveEvidence(captured)
                }
            }
        }
    }

    private fun finishLiveEvidence(captured: Boolean) {
        if (liveEvidenceAbandoned) {
            runCatching { screenshotFile.delete() }
            liveEvidence = LiveEvidence.FAILED
            return
        }
        liveEvidence = if (captured) LiveEvidence.DONE else LiveEvidence.FAILED
        val continuation = onLiveEvidence
        onLiveEvidence = null
        continuation?.invoke(captured)
    }

    private fun drawBackground() {
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        GLES20.glDepthMask(false)
        GLES20.glUseProgram(program)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTexture)
        val posLoc = GLES20.glGetAttribLocation(program, "aPos")
        val texLoc = GLES20.glGetAttribLocation(program, "aTex")
        quadCoords.position(0)
        quadTexCoords.position(0)
        GLES20.glVertexAttribPointer(posLoc, 2, GLES20.GL_FLOAT, false, 0, quadCoords)
        GLES20.glVertexAttribPointer(texLoc, 2, GLES20.GL_FLOAT, false, 0, quadTexCoords)
        GLES20.glEnableVertexAttribArray(posLoc)
        GLES20.glEnableVertexAttribArray(texLoc)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(posLoc)
        GLES20.glDisableVertexAttribArray(texLoc)
        GLES20.glDepthMask(true)
    }

    /**
     * ARCore defaults to fixed focus on most devices because that is stable for
     * tracking.  The Recording API documentation recommends AUTO while taking
     * pictures or video, which is also essential for small-object SfM capture.
     */
    private fun configureCaptureCamera(s: Session) {
        // Prefer the camera config with the largest CPU image: the Recording
        // API records the CPU stream, and its default is only 640x480.
        val thirtyFps = s.getSupportedCameraConfigs(
            CameraConfigFilter(s).setTargetFps(
                EnumSet.of(CameraConfig.TargetFps.TARGET_FPS_30)))
        val fixedThirtyFps = thirtyFps.filter { it.fpsRange.upper <= 30 }
        val configs = fixedThirtyFps.ifEmpty { thirtyFps }.ifEmpty {
            s.getSupportedCameraConfigs(CameraConfigFilter(s))
        }
        val best = configs.maxByOrNull {
            it.imageSize.width.toLong() * it.imageSize.height
        }
        if (best != null) s.cameraConfig = best

        val config = s.config
        config.focusMode = Config.FocusMode.AUTO
        s.configure(config)

        val selected = s.cameraConfig
        val imageSize = selected.imageSize
        captureImageSize = "${imageSize.width}x${imageSize.height}"
        queryFocusCapabilities(selected.cameraId)?.let { capabilities ->
            autoFocusSupported = capabilities.autoFocusSupported
            focusDistanceCalibration = capabilities.distanceCalibration
        }
        cameraConfigLine = buildString {
            append("# camera_config cpu_image ").append(captureImageSize)
            append(" fps ").append(selected.fpsRange.lower)
                .append('-').append(selected.fpsRange.upper)
            append(" focus_mode_requested ").append(config.focusMode.name)
            append(" autofocus_supported ")
                .append(autoFocusSupported?.toString() ?: "unknown")
            append(" focus_distance_calibration ")
                .append(focusDistanceCalibrationLabel())
        }
    }

    private data class FocusCapabilities(
        val autoFocusSupported: Boolean,
        val distanceCalibration: Int?,
    )

    private fun queryFocusCapabilities(cameraId: String): FocusCapabilities? = runCatching {
        val manager = getSystemService(CameraManager::class.java)
        val characteristics = manager.getCameraCharacteristics(cameraId)
        val modes = characteristics
            .get(CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES) ?: intArrayOf()
        val supported = modes.any { mode ->
            mode == CaptureRequest.CONTROL_AF_MODE_AUTO ||
                mode == CaptureRequest.CONTROL_AF_MODE_MACRO ||
                mode == CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO ||
                mode == CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
        }
        FocusCapabilities(
            supported,
            characteristics.get(
                CameraCharacteristics.LENS_INFO_FOCUS_DISTANCE_CALIBRATION),
        )
    }.getOrNull()

    /** Returns true only on frames where a fresh metadata read was attempted. */
    private fun updateFocusMetadata(frame: Frame): Boolean {
        val shouldSample = focusMetadataFrameIndex % FOCUS_METADATA_SAMPLE_INTERVAL == 0
        focusMetadataFrameIndex++
        if (!shouldSample) return false
        actualAfMode = null
        actualAfState = null
        lensFocusDistance = null
        val metadata = runCatching { frame.imageMetadata }.getOrNull() ?: return true
        actualAfMode = runCatching {
            metadata.getByte(ImageMetadata.CONTROL_AF_MODE).toInt()
        }.getOrNull()
        actualAfState = runCatching {
            metadata.getByte(ImageMetadata.CONTROL_AF_STATE).toInt()
        }.getOrNull()
        lensFocusDistance = runCatching {
            metadata.getFloat(ImageMetadata.LENS_FOCUS_DISTANCE)
        }.getOrNull()
        return true
    }

    private fun focusStatusText(): String {
        if (autoFocusSupported == false) return "AF FIXED (unsupported)"
        val mode = when (actualAfMode) {
            CaptureRequest.CONTROL_AF_MODE_OFF -> "AF OFF"
            CaptureRequest.CONTROL_AF_MODE_MACRO -> "AF MACRO"
            CaptureRequest.CONTROL_AF_MODE_AUTO,
            CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO,
            CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE -> "AF AUTO"
            else -> "AF AUTO REQUESTED"
        }
        val state = when (actualAfState) {
            android.hardware.camera2.CaptureResult.CONTROL_AF_STATE_PASSIVE_SCAN,
            android.hardware.camera2.CaptureResult.CONTROL_AF_STATE_ACTIVE_SCAN -> "/searching"
            android.hardware.camera2.CaptureResult.CONTROL_AF_STATE_PASSIVE_FOCUSED,
            android.hardware.camera2.CaptureResult.CONTROL_AF_STATE_FOCUSED_LOCKED -> "/focused"
            android.hardware.camera2.CaptureResult.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED,
            android.hardware.camera2.CaptureResult.CONTROL_AF_STATE_PASSIVE_UNFOCUSED -> "/not focused"
            else -> ""
        }
        val hasMetricCalibration =
            focusDistanceCalibration ==
                CameraCharacteristics.LENS_INFO_FOCUS_DISTANCE_CALIBRATION_APPROXIMATE ||
                focusDistanceCalibration ==
                CameraCharacteristics.LENS_INFO_FOCUS_DISTANCE_CALIBRATION_CALIBRATED
        val distance = if (hasMetricCalibration) {
            lensFocusDistance?.takeIf { it > 0f }?.let {
                " ~${(100f / it).roundToInt()}cm"
            }.orEmpty()
        } else ""
        return mode + state + distance
    }

    private fun focusDistanceCalibrationLabel(): String =
        when (focusDistanceCalibration) {
            CameraCharacteristics.LENS_INFO_FOCUS_DISTANCE_CALIBRATION_UNCALIBRATED ->
                "UNCALIBRATED"
            CameraCharacteristics.LENS_INFO_FOCUS_DISTANCE_CALIBRATION_APPROXIMATE ->
                "APPROXIMATE"
            CameraCharacteristics.LENS_INFO_FOCUS_DISTANCE_CALIBRATION_CALIBRATED ->
                "CALIBRATED"
            else -> "unknown"
        }

    private fun buildProgram(): Int {
        val vs = """
            attribute vec2 aPos;
            attribute vec2 aTex;
            varying vec2 vTex;
            void main() { gl_Position = vec4(aPos, 0.0, 1.0); vTex = aTex; }
        """
        val fs = """
            #extension GL_OES_EGL_image_external : require
            precision mediump float;
            uniform samplerExternalOES uTex;
            varying vec2 vTex;
            void main() { gl_FragColor = texture2D(uTex, vTex); }
        """
        fun compile(type: Int, src: String): Int {
            val sh = GLES20.glCreateShader(type)
            GLES20.glShaderSource(sh, src)
            GLES20.glCompileShader(sh)
            return sh
        }
        val prog = GLES20.glCreateProgram()
        GLES20.glAttachShader(prog, compile(GLES20.GL_VERTEX_SHADER, vs))
        GLES20.glAttachShader(prog, compile(GLES20.GL_FRAGMENT_SHADER, fs))
        GLES20.glLinkProgram(prog)
        return prog
    }
}

private fun direct(data: FloatArray): FloatBuffer =
    ByteBuffer.allocateDirect(data.size * 4).order(ByteOrder.nativeOrder())
        .asFloatBuffer().put(data).apply { position(0) }
