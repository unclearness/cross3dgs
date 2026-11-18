package org.unclearness.cross3dgs

import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.text.InputType
import android.text.TextUtils
import android.view.MotionEvent
import android.view.View
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.WindowManager
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ArrayAdapter
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import org.unclearness.cross3dgs.capture.ArCaptureActivity
import org.unclearness.cross3dgs.capture.PosesFile
import org.unclearness.cross3dgs.capture.RunKeepAliveService
import org.unclearness.cross3dgs.capture.VideoFrameExtractor
import java.io.File
import java.nio.ByteBuffer
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong
import kotlin.concurrent.thread
import kotlin.math.roundToInt

class E2eActivity : Activity(), NativeBridge.Listener {
    private lateinit var workspace: ProjectWorkspace
    @Volatile private lateinit var manifest: ProjectManifest
    private lateinit var captureStore: CaptureStore
    private lateinit var runsRoot: File
    private var pendingCapture: CaptureRecord? = null
    private lateinit var status: TextView
    private lateinit var log: TextView
    private lateinit var logScroll: ScrollView
    private lateinit var progress: ProgressBar
    private lateinit var captureButton: Button
    private lateinit var savedCaptureButton: Button
    private lateinit var frozenReplayButton: Button
    private lateinit var openFinishedButton: Button
    private lateinit var extractButton: Button
    private lateinit var sfmButton: Button
    private lateinit var optimizeButton: Button
    private lateinit var presetButton: Button
    private lateinit var frameTargetButton: Button
    private lateinit var optimizationSettingsButton: Button
    private lateinit var viewButton: Button
    private lateinit var surface: SurfaceView
    // Panel captions over the surface while training, in the UI's own font.
    private lateinit var trainingLabels: FrameLayout
    private var preset = OptimizationPreset.DEMO
    private var frameTargetCount = DEFAULT_FRAME_TARGET_COUNT
    private var optimizationConfig = OptimizationConfig()
    private val viewerExecutor = Executors.newSingleThreadExecutor()
    // Decodes the frames shown while extraction runs, off the extraction
    // thread; a frame arriving while one is still decoding is skipped.
    private val framePreviewExecutor = Executors.newSingleThreadExecutor()
    private val framePreviewBusy = AtomicBoolean(false)
    private val viewerPending = AtomicBoolean(false)
    private val viewerDirty = AtomicBoolean(false)
    private val captureViewAfterNextRender = AtomicBoolean(false)
    private val firstViewEvidencePending = AtomicBoolean(false)
    private val evidenceInventoryFinalizing = AtomicBoolean(false)
    private val screenshotGeneration = AtomicLong(0L)
    private val mainHandler = Handler(Looper.getMainLooper())
    private val firstViewRenderTimeout = Runnable {
        failFirstViewEvidence(IllegalStateException(
            "first viewer render timed out after ${FIRST_VIEW_RENDER_TIMEOUT_MS}ms"))
    }
    private val viewerLock = Any()
    @Volatile private var viewerReady = false
    // Library-rendered SfM live preview; `--ez sfm_preview false` turns it off
    // (e.g. for timing A/Bs). Recorded in pipeline_config.sfm_live_preview.
    private val sfmPreviewEnabled by lazy { intent.getBooleanExtra("sfm_preview", true) }
    // `--es sfm_threads 8,8,4` (extraction,matching,mapping; 0 = one per core)
    // for per-device tuning. Recorded in pipeline_config.sfm_threads.
    private val sfmThreads by lazy {
        intent.getStringExtra("sfm_threads")
            ?.takeIf { it.matches(Regex("\\d+,\\d+,\\d+")) } ?: "default"
    }
    // `--es sfm_cpus 4-7` pins the SfM workers to those CPUs (e.g. the big
    // cores); recorded in pipeline_config.sfm_cpus.
    private val sfmCpus by lazy {
        intent.getStringExtra("sfm_cpus")
            ?.takeIf { SfmNativeBridge.parseCpuMask(it) != 0L } ?: "all"
    }
    private var orbitYaw = 0f
    private var orbitPitch = 0f
    // Pitch range of the prepared viewer (gravity-limited when known).
    private var orbitMinPitch = -1.35f
    private var orbitMaxPitch = 1.35f
    private var orbitZoom = 1f
    private var orbitPanX = 0f
    private var orbitPanY = 0f
    private var lastTouchX = 0f
    private var lastTouchY = 0f
    private var lastSpan = 0f
    private var optimizationStartedMs = 0L
    // The optimize screenshot taken while training runs (see onProgress), and
    // what onFinished continues with once it has settled. UI thread.
    private var trainingShot = TrainingShot.NONE
    private var afterTrainingShot: (() -> Unit)? = null
    // The optional screenshots taken during SfM (see PREVIEW_SCREENSHOTS),
    // recorded once SfM returns. UI thread.
    private val previewShotsRequested = mutableSetOf<String>()
    private val previewShots = linkedMapOf<String, File>()
    private var previewShotPending = false
    private var viewerStartedMs = 0L
    private val sfmTimingLock = Any()
    private var sfmSubstage: String? = null
    private var sfmSubstageStartedMs = 0L
    private val sfmStageTimes = linkedMapOf<String, Long>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val storageRoot = File(filesDir, "capture_to_view")
        captureStore = CaptureStore(File(storageRoot, "captures"))
        runsRoot = File(storageRoot, "runs").also(File::mkdirs)
        // Recreated or reopened from Recents: the re-delivered launch intent
        // must neither start automation again (configureAutomation) nor open
        // a fresh empty run; continue with the most recently updated run.
        val relaunched = savedInstanceState != null ||
            (intent.flags and Intent.FLAG_ACTIVITY_LAUNCHED_FROM_HISTORY) != 0
        val requestedId = intent.getStringExtra("run_id")
            ?: intent.getStringExtra("project_id")
            ?: runsRoot.takeIf { relaunched }
                ?.listFiles(File::isDirectory)?.maxByOrNull(File::lastModified)?.name
            ?: uniqueArtifactId("run")
        require(requestedId.matches(Regex("[A-Za-z0-9_.-]+"))) {
            "project_id contains unsupported characters"
        }
        workspace = ProjectWorkspace.open(runsRoot, requestedId)
        manifest = if (workspace.manifestFile.isFile) {
            workspace.load()
        } else {
            ProjectManifest.create(requestedId).also(workspace::save)
        }
        if (manifest.environment.isEmpty() || manifest.pipelineConfig.isEmpty()) {
            manifest = manifest.copy(
                environment = ProjectEvidence.environment(this),
                pipelineConfig = ProjectEvidence.pipelineConfig(
                    manifest.frameTargetCount, manifest.optimizationConfig,
                    manifest.preset, TILE_BIN_CAPACITY, sfmPreviewEnabled, sfmThreads, sfmCpus),
            ).also(workspace::save)
        }
        val resumeFinalInventory = manifest.stage == PipelineStage.VIEW &&
            manifest.screenshots.containsKey("view")
        evidenceInventoryFinalizing.set(resumeFinalInventory)
        if (manifest.stage == PipelineStage.COMPLETE || resumeFinalInventory) {
            ProjectTelemetry.stop()
        } else {
            ProjectTelemetry.start(
                this, workspace.telemetryFile, manifest.stage.name.lowercase())
        }
        val intentPreset = intent.getStringExtra("preset")?.uppercase()?.let {
            runCatching { OptimizationPreset.valueOf(it) }.getOrNull()
        }
        preset = intentPreset ?: manifest.preset
        // Settable per launch so automated runs no longer need to seed a
        // project manifest by hand (the seeding shortcut is what once shipped
        // a manifest without schema_version).
        frameTargetCount = intent.getIntExtra(
            "frame_target_count", manifest.frameTargetCount)
            .coerceIn(MIN_FRAME_TARGET_COUNT, MAX_FRAME_TARGET_COUNT)
        // A launch-supplied preset must carry its budget with it. Reusing a
        // capture bundle whose manifest was written at another preset would
        // otherwise relabel the run without changing iterations or capacity.
        optimizationConfig = if (intentPreset != null && intentPreset != manifest.preset) {
            OptimizationConfig.fromPreset(intentPreset)
        } else {
            manifest.optimizationConfig
        }
        // Shortens a test run without a settings dialog.
        intent.getIntExtra("iterations", 0).takeIf { it > 0 }?.let {
            optimizationConfig = optimizationConfig.copy(iterations = it)
        }
        buildUi()
        RunKeepAliveService.onCancelRequested = {
            NativeBridge.requestStop()
            SfmNativeBridge.cancel()
        }
        NativeBridge.setListener(this)
        if (SfmNativeBridge.isAvailable()) {
            SfmNativeBridge.logListener = ::onSfmLogLine
        }
        configureAutomation(relaunched)
        refresh()
        // A stage from an earlier activity can still be working; show that
        // rather than offering buttons that would refuse.
        if (activeStages.get() > 0) {
            setBusy(true, "A run started earlier is still working")
        }
    }

    private fun buildUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(16, 16, 16, 16)
        }
        // From Android 15 the window is edge to edge whether or not we ask, so
        // the status and navigation bars sit on top of the content unless the
        // root pads itself by their insets.
        ViewCompat.setOnApplyWindowInsetsListener(root) { view, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or
                    WindowInsetsCompat.Type.displayCutout(),
            )
            view.setPadding(
                bars.left + 16, bars.top + 16, bars.right + 16, bars.bottom + 16,
            )
            insets
        }
        // Two fixed lines (the SfM live preview captions use both) so the
        // surface below never moves between stages.
        status = TextView(this).apply {
            textSize = 12f
            // Three lines, so a long SfM caption (matching with thousands of
            // pairs) wraps without pushing its legend line out of view.
            setLines(3)
        }
        captureButton = compactButton("Capture") { startImmutableCapture() }
        savedCaptureButton = compactButton("Run saved capture") { chooseSavedCapture() }
        frozenReplayButton =
            compactButton("Replay frozen SfM input") { chooseFrozenReplay() }
        openFinishedButton = compactButton("Open finished run") { chooseFinishedRun() }
        extractButton = compactButton("Extract") { extractFrames() }
        sfmButton = compactButton("SfM") { runSfm() }
        optimizeButton = compactButton("Optimize") { runOptimization() }
        presetButton = compactButton {
            preset = if (preset == OptimizationPreset.DEMO) {
                OptimizationPreset.FULL
            } else {
                OptimizationPreset.DEMO
            }
            optimizationConfig = OptimizationConfig.fromPreset(preset)
            persistPendingConfiguration()
            refresh()
        }
        frameTargetButton = compactButton { showFrameTargetDialog() }
        optimizationSettingsButton = compactButton { showOptimizationSettingsDialog() }
        viewButton = compactButton("View result") { prepareViewerAndRender() }
        viewButton.isEnabled = false
        val resetViewButton = compactButton("Reset view") {
            synchronized(viewerLock) {
                orbitYaw = 0f
                orbitPitch = 0f
                orbitZoom = 1f
                orbitPanX = 0f
                orbitPanY = 0f
            }
            requestOrbitRender()
        }
        val cancelButton = compactButton("Cancel stage") {
            NativeBridge.requestStop()
            SfmNativeBridge.cancel()
            appendLog("Cancellation requested")
        }
        progress = ProgressBar(
            this, null, android.R.attr.progressBarStyleHorizontal)
        surface = SurfaceView(this)
        surface.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                NativeBridge.setSurface(holder.surface)
            }
            override fun surfaceChanged(
                holder: SurfaceHolder,
                format: Int,
                width: Int,
                height: Int,
            ) {
                NativeBridge.setSurface(holder.surface)
                if (firstViewEvidencePending.get()) requestOrbitRender()
            }
            override fun surfaceDestroyed(holder: SurfaceHolder) {
                NativeBridge.setSurface(null)
            }
        })
        surface.setOnTouchListener { _, event -> handleViewerTouch(event) }
        trainingLabels = FrameLayout(this).apply {
            visibility = View.GONE
            NativeBridge.trainingPreviewLabels().forEach {
                addView(trainingLabel(it), FrameLayout.LayoutParams(-2, -2))
            }
        }
        val surfaceHost = FrameLayout(this).apply {
            addView(surface, FrameLayout.LayoutParams(-1, -1))
            addView(trainingLabels, FrameLayout.LayoutParams(-1, -1))
        }
        log = TextView(this).apply {
            typeface = android.graphics.Typeface.MONOSPACE
            textSize = 10f
        }
        // Same arrangement as the iOS app: one stage row, the run and setting
        // controls two per row, then the surface takes most of the screen.
        root.addView(status)
        root.addView(buttonRow(captureButton, extractButton, sfmButton, optimizeButton))
        root.addView(buttonRow(openFinishedButton, savedCaptureButton))
        root.addView(buttonRow(frozenReplayButton, presetButton))
        root.addView(buttonRow(frameTargetButton, optimizationSettingsButton))
        root.addView(progress)
        root.addView(surfaceHost, LinearLayout.LayoutParams(-1, 0, 5f))
        root.addView(buttonRow(viewButton, resetViewButton, cancelButton))
        logScroll = ScrollView(this).apply { addView(log) }
        root.addView(logScroll, LinearLayout.LayoutParams(-1, 0, 2f))
        setContentView(root)
    }

    private fun trainingLabel(caption: String): TextView =
        TextView(this).apply {
            text = caption
            textSize = 12f
            setTextColor(Color.WHITE)
            setPadding(dp(6), dp(2), dp(6), dp(2))
            background = GradientDrawable().apply {
                setColor(0x99000000.toInt())
                cornerRadius = dp(4).toFloat()
            }
        }

    /** Puts each caption in the corner of its panel, as the surface fits it. */
    private fun placeTrainingLabels() {
        val layout = NativeBridge.trainingPreviewLayout()
        if (layout.isEmpty() || surface.width == 0 || surface.height == 0) return
        val scale = minOf(
            surface.width.toFloat() / layout[0],
            surface.height.toFloat() / layout[1],
        )
        val left = (surface.width - layout[0] * scale) / 2f
        val top = (surface.height - layout[1] * scale) / 2f
        val margin = dp(4)
        for (index in 0 until trainingLabels.childCount) {
            val panel = 2 + index * 4
            (trainingLabels.getChildAt(index) as TextView).apply {
                translationX = left + layout[panel] * scale + margin
                translationY = top + layout[panel + 1] * scale + margin
                maxWidth = (layout[panel + 2] * scale).toInt() - 2 * margin
            }
        }
        trainingLabels.visibility = View.VISIBLE
    }

    private fun dp(value: Int): Int =
        (value * resources.displayMetrics.density).roundToInt()

    // Small one-line button without the platform's minimum touch height, so
    // the controls take a fraction of the screen they took as full-width rows.
    private fun compactButton(label: String = "", onClick: () -> Unit): Button =
        Button(this).apply {
            text = label
            textSize = 12f
            isAllCaps = false
            minHeight = 0
            minimumHeight = 0
            minWidth = 0
            minimumWidth = 0
            setPadding(dp(4), dp(6), dp(4), dp(6))
            maxLines = 1
            ellipsize = TextUtils.TruncateAt.END
            setOnClickListener { onClick() }
        }

    private fun buttonRow(vararg buttons: Button): LinearLayout =
        LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            buttons.forEach { addView(it, LinearLayout.LayoutParams(0, -2, 1f)) }
        }

    private fun showFrameTargetDialog() {
        val input = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER
            setText(frameTargetCount.toString())
            selectAll()
        }
        val dialog = AlertDialog.Builder(this)
            .setTitle("Frames to extract")
            .setMessage("Enter $MIN_FRAME_TARGET_COUNT-$MAX_FRAME_TARGET_COUNT. " +
                "Each target decodes two candidates and keeps the sharper frame.")
            .setView(input)
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Apply", null)
            .create()
        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val requested = input.text.toString().toIntOrNull()
                if (requested == null || requested !in
                    MIN_FRAME_TARGET_COUNT..MAX_FRAME_TARGET_COUNT) {
                    input.error = "Enter $MIN_FRAME_TARGET_COUNT-$MAX_FRAME_TARGET_COUNT"
                    return@setOnClickListener
                }
                frameTargetCount = requested
                persistPendingConfiguration()
                refresh()
                dialog.dismiss()
            }
        }
        dialog.show()
    }

    private fun showOptimizationSettingsDialog() {
        fun numericField(value: Int): EditText = EditText(this).apply {
            inputType = InputType.TYPE_CLASS_NUMBER
            setText(value.toString())
            selectAll()
        }
        val iterationsInput = numericField(optimizationConfig.iterations)
        val capacityInput = numericField(optimizationConfig.gaussianCapacity)
        val shValues = listOf("0", "1", "2", "3")
        val densityValues = listOf("mcmc", "fixed")
        val shSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@E2eActivity,
                android.R.layout.simple_spinner_dropdown_item,
                shValues,
            )
            setSelection(shValues.indexOf(
                optimizationConfig.shDegree.toString()).coerceAtLeast(0))
        }
        val densitySpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@E2eActivity,
                android.R.layout.simple_spinner_dropdown_item,
                densityValues,
            )
            setSelection(densityValues.indexOf(
                optimizationConfig.densityStrategy).coerceAtLeast(0))
        }
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 0, 48, 0)
            addView(TextView(this@E2eActivity).apply { text = "Iterations" })
            addView(iterationsInput)
            addView(TextView(this@E2eActivity).apply { text = "Max Gaussians" })
            addView(capacityInput)
            addView(TextView(this@E2eActivity).apply { text = "SH degree" })
            addView(shSpinner)
            addView(TextView(this@E2eActivity).apply { text = "Density strategy" })
            addView(densitySpinner)
        }
        val dialog = AlertDialog.Builder(this)
            .setTitle("3DGS settings")
            .setMessage("Large capacities can exhaust device memory. " +
                "Demo/Full resets these fields to its defaults.")
            .setView(form)
            .setNegativeButton("Cancel", null)
            .setPositiveButton("Apply", null)
            .create()
        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val iterations = iterationsInput.text.toString().toIntOrNull()
                val capacity = capacityInput.text.toString().toIntOrNull()
                if (iterations == null || iterations !in
                    MIN_TRAINING_ITERATIONS..MAX_TRAINING_ITERATIONS) {
                    iterationsInput.error =
                        "Enter $MIN_TRAINING_ITERATIONS-$MAX_TRAINING_ITERATIONS"
                    return@setOnClickListener
                }
                if (capacity == null || capacity !in
                    MIN_GAUSSIAN_CAPACITY..MAX_GAUSSIAN_CAPACITY) {
                    capacityInput.error =
                        "Enter $MIN_GAUSSIAN_CAPACITY-$MAX_GAUSSIAN_CAPACITY"
                    return@setOnClickListener
                }
                optimizationConfig = OptimizationConfig(
                    iterations = iterations,
                    gaussianCapacity = capacity,
                    shDegree = shValues[shSpinner.selectedItemPosition].toInt(),
                    densityStrategy = densityValues[densitySpinner.selectedItemPosition],
                )
                persistPendingConfiguration()
                refresh()
                dialog.dismiss()
            }
        }
        dialog.show()
    }

    private fun persistPendingConfiguration() {
        if (manifest.stage != PipelineStage.CAPTURE ||
            manifest.captureId.isNotEmpty()) return
        manifest = manifest.copy(
            preset = preset,
            frameTargetCount = frameTargetCount,
            optimizationConfig = optimizationConfig,
            pipelineConfig = ProjectEvidence.pipelineConfig(
                frameTargetCount, optimizationConfig, preset,
                TILE_BIN_CAPACITY, sfmPreviewEnabled, sfmThreads, sfmCpus),
        ).also(workspace::save)
    }

    private fun prepareViewerAndRender() {
        trainingLabels.visibility = View.GONE
        check(manifest.stage == PipelineStage.VIEW ||
            manifest.stage == PipelineStage.COMPLETE) {
            "viewer requires stage=view, got ${manifest.stage}"
        }
        val requiresCompletionEvidence = manifest.stage == PipelineStage.VIEW
        if (requiresCompletionEvidence && !recordedScreenshotIsValid("optimize")) {
            evidenceInventoryFinalizing.set(false)
            recordFailure(IllegalStateException(
                "viewer requires a valid optimization screenshot"))
            setBusy(false)
            refresh()
            return
        }
        if (requiresCompletionEvidence && manifest.screenshots.containsKey("view")) {
            if (!recordedScreenshotIsValid("view")) {
                evidenceInventoryFinalizing.set(false)
                recordFailure(IllegalStateException(
                    "recorded viewer screenshot is missing or changed"))
                setBusy(false)
                refresh()
                return
            }
            firstViewEvidencePending.set(true)
            viewerStartedMs = SystemClock.elapsedRealtime()
            finalizeCompletedRun()
            return
        }
        setBusy(true, "Viewer: Preparing free-view renderer")
        if (requiresCompletionEvidence) ProjectTelemetry.setStage("view")
        viewerStartedMs = SystemClock.elapsedRealtime()
        thread {
            val checkpoint = File(workspace.output, "checkpoint.d3gs")
            viewerReady = NativeBridge.prepareViewer(
                workspace.root.absolutePath,
                manifest.imageDirectory,
                manifest.sparseDirectory,
                checkpoint.absolutePath,
                intent.getStringExtra("viewer_reference_image").orEmpty(),
                workspace.framePriorsFile.takeIf(File::isFile)?.absolutePath.orEmpty(),
                // Only the first view: a finished run is sealed.
                if (requiresCompletionEvidence) {
                    File(workspace.output, "splat_gravity_up.ply").absolutePath
                } else {
                    ""
                },
                manifest.optimizationConfig.shDegree,
            )
            if (viewerReady) {
                val range = NativeBridge.orbitPitchRange()
                synchronized(viewerLock) {
                    orbitMinPitch = range[0]
                    orbitMaxPitch = range[1]
                    orbitPitch = orbitPitch.coerceIn(orbitMinPitch, orbitMaxPitch)
                }
            }
            runOnUiThread {
                if (viewerReady) {
                    if (requiresCompletionEvidence) {
                        manifest = manifest.copy(
                            stageTimesMs = manifest.stageTimesMs + (
                                "view_prepare" to
                                    (SystemClock.elapsedRealtime() - viewerStartedMs)),
                        ).also(workspace::save)
                        appendLog(
                            "Viewer ready: drag=rotate, pinch=zoom, two fingers=pan")
                        // The view screenshot follows the first render; the
                        // status says what the screen now is, as on iOS.
                        status.text = "Viewer: Free-view rendering\n" +
                            "Drag = rotate · pinch = zoom · 2 fingers = pan"
                        firstViewEvidencePending.set(true)
                        captureViewAfterNextRender.set(true)
                        mainHandler.removeCallbacks(firstViewRenderTimeout)
                        mainHandler.postDelayed(
                            firstViewRenderTimeout, FIRST_VIEW_RENDER_TIMEOUT_MS)
                    } else {
                        setBusy(false)
                        // setBusy leaves whatever message it was given last, so
                        // a sealed run would keep reading "Preparing free-view
                        // renderer" for as long as it is on screen.
                        refresh()
                    }
                    requestOrbitRender()
                } else {
                    val error = IllegalStateException("viewer preparation failed")
                    if (requiresCompletionEvidence) {
                        recordFailure(error)
                    }
                    setBusy(false)
                    Toast.makeText(this, "Viewer preparation failed", Toast.LENGTH_LONG).show()
                    if (requiresCompletionEvidence) {
                        captureStageScreenshot("view_failed")
                    }
                }
            }
        }
    }

    /**
     * Centroid and spread of the touches, ignoring one pointer by index so a
     * finger that is lifting does not drag the centroid with it.
     */
    private fun touchFocus(event: MotionEvent, skip: Int): Triple<Float, Float, Float> {
        var sumX = 0f
        var sumY = 0f
        var count = 0
        for (i in 0 until event.pointerCount) {
            if (i == skip) continue
            sumX += event.getX(i)
            sumY += event.getY(i)
            count++
        }
        if (count == 0) return Triple(0f, 0f, 0f)
        val focusX = sumX / count
        val focusY = sumY / count
        var span = 0f
        if (count >= 2) {
            var spread = 0f
            for (i in 0 until event.pointerCount) {
                if (i == skip) continue
                spread += kotlin.math.hypot(
                    event.getX(i) - focusX, event.getY(i) - focusY,
                )
            }
            span = 2f * spread / count
        }
        return Triple(focusX, focusY, span)
    }

    private fun handleViewerTouch(event: MotionEvent): Boolean {
        if (!viewerReady) return false
        // On a lift, the remaining fingers define the new baseline; measuring
        // against the old centroid would jump the view by the distance the
        // centroid moved when the pointer disappeared.
        val skip = if (event.actionMasked == MotionEvent.ACTION_POINTER_UP) {
            event.actionIndex
        } else -1
        val (focusX, focusY, span) = touchFocus(event, skip)
        val active = event.pointerCount - if (skip >= 0) 1 else 0
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN,
            MotionEvent.ACTION_POINTER_DOWN,
            MotionEvent.ACTION_POINTER_UP -> {
                lastTouchX = focusX
                lastTouchY = focusY
                lastSpan = span
            }
            MotionEvent.ACTION_MOVE -> {
                val dx = focusX - lastTouchX
                val dy = focusY - lastTouchY
                val width = surface.width.coerceAtLeast(1)
                val height = surface.height.coerceAtLeast(1)
                synchronized(viewerLock) {
                    if (active <= 1) {
                        // Half a turn across the surface, so a comfortable
                        // swipe covers a face of the subject rather than
                        // several revolutions.
                        orbitYaw += dx / width * ORBIT_TURN_PER_SWEEP
                        // Pitch turns about the camera's right axis, whose
                        // sense is opposite the yaw axis, so a drag down has
                        // to subtract to tip the subject towards the viewer
                        // the way a drag right turns it to the right.
                        orbitPitch = (orbitPitch - dy / height * ORBIT_TURN_PER_SWEEP)
                            .coerceIn(orbitMinPitch, orbitMaxPitch)
                    } else {
                        // The camera pans opposite the drag so the subject
                        // travels with the fingers, matching the direction a
                        // one-finger drag already turns it. Scaling by the
                        // current distance keeps the subject under the
                        // fingers at every zoom level.
                        orbitPanX -= dx / width * orbitZoom
                        orbitPanY -= dy / height * orbitZoom
                        if (lastSpan > 1f && span > 1f) {
                            orbitZoom = (orbitZoom * lastSpan / span)
                                .coerceIn(0.2f, 5f)
                        }
                    }
                }
                lastTouchX = focusX
                lastTouchY = focusY
                lastSpan = span
                requestOrbitRender()
            }
        }
        return true
    }

    private fun requestOrbitRender() {
        if (!viewerReady || surface.width <= 0 || surface.height <= 0) return
        viewerDirty.set(true)
        if (!viewerPending.compareAndSet(false, true)) return
        viewerExecutor.execute {
            var rendered = false
            try {
                while (viewerDirty.getAndSet(false)) {
                    val values = synchronized(viewerLock) {
                        floatArrayOf(
                            orbitYaw, orbitPitch, orbitZoom,
                            orbitPanX, orbitPanY,
                        )
                    }
                    rendered = NativeBridge.renderOrbit(
                        values[0], values[1], values[2], values[3], values[4],
                        surface.width, surface.height,
                    ) || rendered
                }
            } finally {
                viewerPending.set(false)
                if (rendered && captureViewAfterNextRender.compareAndSet(true, false)) {
                    mainHandler.postDelayed(::captureFirstViewEvidence, 150L)
                } else if (!rendered && firstViewEvidencePending.get()) {
                    mainHandler.postDelayed({
                        if (firstViewEvidencePending.get()) requestOrbitRender()
                    }, VIEW_RENDER_RETRY_MS)
                }
                if (viewerDirty.get()) requestOrbitRender()
            }
        }
    }

    private fun captureFirstViewEvidence() {
        if (!firstViewEvidencePending.get() || manifest.stage != PipelineStage.VIEW) {
            return
        }
        mainHandler.removeCallbacks(firstViewRenderTimeout)
        captureStageScreenshot(
            "view",
            onSuccess = { measureViewerFrames(::finalizeCompletedRun) },
            onFailure = ::failFirstViewEvidence,
        )
    }

    /**
     * Time a bounded, deterministic orbit sweep so the sealed run carries a
     * viewer frame-time sample instead of only the view-stage wall time. It
     * runs after the "view" evidence screenshot, so the recorded figure keeps
     * the reference pose, restores that pose when it finishes, and always
     * continues to [onDone] so a measurement failure cannot strand the run in
     * stage=view.
     */
    private fun measureViewerFrames(onDone: () -> Unit) {
        // Close view_total here, before the sweep runs. finalizeCompletedRun
        // otherwise derives it from viewerStartedMs and would absorb this
        // instrumentation into the pipeline stage time, so end-to-end totals
        // would stop being comparable with runs recorded before the sweep.
        if ("view_total" !in manifest.stageTimesMs) {
            manifest = manifest.copy(
                stageTimesMs = manifest.stageTimesMs + ("view_total" to
                    (SystemClock.elapsedRealtime() - viewerStartedMs)),
            ).also(workspace::save)
        }
        val width = surface.width
        val height = surface.height
        if (!viewerReady || width <= 0 || height <= 0) {
            onDone()
            return
        }
        setBusy(true, "Viewer: Measuring frame time")
        val deadline = SystemClock.elapsedRealtime() + VIEWER_SWEEP_TIMEOUT_MS
        val dispatched = runCatching {
            viewerExecutor.execute {
                val samples = ArrayList<Long>(VIEWER_SWEEP_FRAMES)
                runCatching {
                    for (frame in 0 until VIEWER_SWEEP_FRAMES) {
                        if (SystemClock.elapsedRealtime() > deadline) break
                        val yaw = VIEWER_SWEEP_YAW_SPAN *
                            (frame.toFloat() / (VIEWER_SWEEP_FRAMES - 1) - 0.5f)
                        val started = System.nanoTime()
                        val rendered = NativeBridge.renderOrbit(
                            yaw, 0f, 1f, 0f, 0f, width, height)
                        val elapsedUs = (System.nanoTime() - started) / 1_000L
                        if (rendered) samples.add(elapsedUs)
                    }
                    // Leave the surface on the evidence screenshot's pose.
                    NativeBridge.renderOrbit(0f, 0f, 1f, 0f, 0f, width, height)
                }.onFailure { error ->
                    appendLog("Viewer frame sweep failed: ${error.message}")
                }
                val sweep = samples.takeIf { it.isNotEmpty() }?.let {
                    ViewerFrameSweep(it, width, height, VIEWER_SWEEP_YAW_SPAN)
                }
                runOnUiThread {
                    if (sweep != null) {
                        manifest = manifest.copy(viewerFrameSweep = sweep)
                            .also(workspace::save)
                        appendLog(
                            "Viewer frame sweep: ${sweep.frameTimesUs.size} " +
                                "frames, median %.1f ms".format(
                                    sweep.medianFrameMs))
                    } else {
                        appendLog("Viewer frame sweep recorded no frames")
                    }
                    onDone()
                }
            }
        }
        if (dispatched.isFailure) onDone()
    }

    private fun finalizeCompletedRun() {
        if (!firstViewEvidencePending.compareAndSet(true, false)) return
        captureViewAfterNextRender.set(false)
        mainHandler.removeCallbacks(firstViewRenderTimeout)
        val finalizerKey = workspace.root.canonicalPath
        if (!ACTIVE_FINAL_INVENTORIES.add(finalizerKey)) {
            observeActiveFinalizer(finalizerKey)
            return
        }
        val completionBase = runCatching {
            val completedViewMs = manifest.stageTimesMs["view_total"]
                ?: (SystemClock.elapsedRealtime() - viewerStartedMs)
            manifest = manifest.copy(
                stageTimesMs = manifest.stageTimesMs +
                    ("view_total" to completedViewMs),
            ).also(workspace::save)
            appendLog("Initial viewer render and screenshot verified; finalizing inventory")
            ProjectTelemetry.stop()
            synchronized(workspace.pipelineLog) {
                evidenceInventoryFinalizing.set(true)
            }
            setBusy(true, "Finalizing evidence inventory")
            manifest
        }.getOrElse { error ->
            try {
                evidenceInventoryFinalizing.set(false)
                recordFailure(error)
                setBusy(false)
                refresh()
            } finally {
                ACTIVE_FINAL_INVENTORIES.remove(finalizerKey)
            }
            return
        }
        thread {
            val result = runCatching {
                check(completionBase.stage == PipelineStage.VIEW) {
                    "completion inventory requires stage=view"
                }
                val recordedStages = completionBase.screenshots.keys
                check(recordedStages.containsAll(REQUIRED_SCREENSHOT_STAGES) &&
                    (recordedStages - REQUIRED_SCREENSHOT_STAGES)
                        .all(OPTIONAL_SCREENSHOT_STAGES::contains)) {
                    "complete run requires the five stage screenshots and " +
                        "allows only $OPTIONAL_SCREENSHOT_STAGES besides"
                }
                val expectedScreenshots = completionBase.screenshots.values.toSet()
                check(expectedScreenshots.size == recordedStages.size) {
                    "screenshot stages must use distinct artifacts"
                }
                val actualScreenshots = workspace.screenshotsDir.walkTopDown()
                    .filter(File::isFile)
                    .map { it.relativeTo(workspace.root).invariantSeparatorsPath }
                    .toSet()
                check(actualScreenshots == expectedScreenshots) {
                    "screenshot artifact set does not match the stage records"
                }
                REQUIRED_SCREENSHOT_STAGES.forEach { requiredStage ->
                    val relative = completionBase.screenshots[requiredStage]
                        ?: error("missing $requiredStage screenshot record")
                    check(File(workspace.root, relative).isFile) {
                        "missing $requiredStage screenshot artifact: $relative"
                    }
                }
                val inventory = ProjectEvidence.inventory(workspace)
                REQUIRED_SCREENSHOT_STAGES.forEach { requiredStage ->
                    val relative = completionBase.screenshots.getValue(requiredStage)
                    check(inventory.containsKey(relative)) {
                        "$requiredStage screenshot missing from final inventory"
                    }
                }
                completionBase.copy(
                    stage = PipelineStage.COMPLETE,
                    artifactSha256 = inventory,
                    lastError = null,
                ).also(workspace::save)
            }
            runOnUiThread {
                try {
                    result.fold(
                        onSuccess = { completed ->
                            manifest = completed
                            evidenceInventoryFinalizing.set(false)
                            setBusy(false)
                            refresh()
                        },
                        onFailure = { error ->
                            evidenceInventoryFinalizing.set(false)
                            recordFailure(error)
                            setBusy(false)
                            refresh()
                        },
                    )
                } finally {
                    ACTIVE_FINAL_INVENTORIES.remove(finalizerKey)
                }
            }
        }
    }

    private fun observeActiveFinalizer(finalizerKey: String) {
        evidenceInventoryFinalizing.set(true)
        setBusy(true, "Final evidence inventory is already running")
        mainHandler.postDelayed(object : Runnable {
            override fun run() {
                if (isFinishing || isDestroyed) return
                val saved = runCatching { workspace.load() }.getOrNull()
                when (saved?.stage) {
                    PipelineStage.COMPLETE, PipelineStage.FAILED -> {
                        manifest = saved
                        evidenceInventoryFinalizing.set(false)
                        setBusy(false)
                        refresh()
                    }
                    PipelineStage.VIEW -> {
                        if (ACTIVE_FINAL_INVENTORIES.contains(finalizerKey)) {
                            mainHandler.postDelayed(this, FINALIZER_POLL_MS)
                        } else {
                            manifest = saved
                            firstViewEvidencePending.set(true)
                            viewerStartedMs = SystemClock.elapsedRealtime()
                            finalizeCompletedRun()
                        }
                    }
                    else -> mainHandler.postDelayed(this, FINALIZER_POLL_MS)
                }
            }
        }, FINALIZER_POLL_MS)
    }

    private fun recordedScreenshotIsValid(stage: String): Boolean {
        val relative = manifest.screenshots[stage] ?: return false
        val expected = manifest.artifactSha256[relative] ?: return false
        val file = File(workspace.root, relative)
        return file.isFile && file.length() > 0L && runCatching {
            ProjectEvidence.sha256(file).equals(expected, ignoreCase = true)
        }.getOrDefault(false)
    }

    private fun failFirstViewEvidence(error: Throwable) {
        if (!firstViewEvidencePending.compareAndSet(true, false)) return
        captureViewAfterNextRender.set(false)
        mainHandler.removeCallbacks(firstViewRenderTimeout)
        evidenceInventoryFinalizing.set(false)
        recordFailure(error)
        setBusy(false)
        refresh()
    }

    private fun startImmutableCapture() {
        if (refuseWhileRunning()) return
        val capture = captureStore.create()
        pendingCapture = capture
        ProjectTelemetry.setStage("capture")
        startActivityForResult(
            Intent(this, ArCaptureActivity::class.java)
                .putExtra(ArCaptureActivity.EXTRA_CAPTURE_ROOT,
                    capture.root.absolutePath),
            REQUEST_CAPTURE,
        )
    }

    private fun chooseSavedCapture() {
        if (refuseWhileRunning()) return
        val captures = captureStore.listComplete()
        if (captures.isEmpty()) {
            Toast.makeText(this, "No completed captures", Toast.LENGTH_LONG).show()
            return
        }
        AlertDialog.Builder(this)
            .setTitle("Select immutable capture")
            .setItems(captures.map(CaptureRecord::id).toTypedArray()) { _, index ->
                beginRunFromCapture(captures[index], "saved_capture")
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    /**
     * Offer the sealed runs that still carry a checkpoint, so a finished
     * reconstruction can be looked at again without recomputing it. Only
     * stage=complete runs are listed: a run still at stage=view has evidence
     * left to record, and merely browsing it must not write to it.
     */
    private fun chooseFinishedRun() {
        if (refuseWhileRunning()) return
        val finished = runsRoot.listFiles(File::isDirectory).orEmpty()
            .filter { it != workspace.root }
            .map(ProjectWorkspace::inspect)
            .filter { candidate ->
                candidate.manifestFile.isFile &&
                    runCatching {
                        candidate.load().stage == PipelineStage.COMPLETE &&
                            File(candidate.output, "checkpoint.d3gs").isFile
                    }.getOrDefault(false)
            }
            // Run ids are a mix of timestamps and hand-chosen names, so the
            // manifest's own mtime is the only ordering that stays newest-first
            // for both.
            .sortedByDescending { it.manifestFile.lastModified() }
        if (finished.isEmpty()) {
            Toast.makeText(this, "No finished runs", Toast.LENGTH_LONG).show()
            return
        }
        val labels = finished.map { candidate ->
            val loaded = runCatching { candidate.load() }.getOrNull()
                ?: return@map candidate.root.name
            "${candidate.root.name}\n${loaded.preset.name} · " +
                "${loaded.optimizationConfig.iterations} it · " +
                "${loaded.frameTargetCount} frames"
        }.toTypedArray()
        AlertDialog.Builder(this)
            .setTitle("Open finished run")
            .setItems(labels) { _, index -> openFinishedRun(finished[index]) }
            .setNegativeButton("Cancel", null)
            .show()
    }

    /**
     * Swap the activity onto a finished run and show it.
     *
     * The workspace is replaced in place rather than by relaunching the
     * activity with the run id. A relaunch races the surface lifecycle: the
     * incoming instance claims the native surface before the outgoing one
     * tears its own down, so the outgoing `surfaceDestroyed` clears the
     * surface the new viewer is about to draw into. Rendering then silently
     * does nothing and the screen keeps presenting the previous run's last
     * frame, which looks exactly like a viewer that has stopped responding.
     */
    private fun openFinishedRun(candidate: ProjectWorkspace) {
        val loaded = runCatching { candidate.load() }.getOrElse { error ->
            Toast.makeText(
                this, "Cannot open run: ${error.message}", Toast.LENGTH_LONG,
            ).show()
            return
        }
        ProjectTelemetry.stop()
        mainHandler.removeCallbacks(firstViewRenderTimeout)
        viewerReady = false
        firstViewEvidencePending.set(false)
        captureViewAfterNextRender.set(false)
        evidenceInventoryFinalizing.set(false)
        synchronized(viewerLock) {
            orbitYaw = 0f
            orbitPitch = 0f
            orbitZoom = 1f
            orbitPanX = 0f
            orbitPanY = 0f
        }
        workspace = candidate
        manifest = loaded
        preset = loaded.preset
        frameTargetCount = loaded.frameTargetCount
        optimizationConfig = loaded.optimizationConfig
        log.text = ""
        refresh()
        prepareViewerAndRender()
    }

    private fun chooseFrozenReplay() {
        if (refuseWhileRunning()) return
        val replayable = runsRoot.listFiles(File::isDirectory).orEmpty()
            .filter { it != workspace.root }
            .map(ProjectWorkspace::inspect)
            .filter { candidate ->
                candidate.manifestFile.isFile &&
                    runCatching { candidate.load().stage == PipelineStage.COMPLETE }
                        .getOrDefault(false) &&
                    candidate.framePriorsFile.isFile &&
                    (candidate.sourceImages.listFiles()?.count(File::isFile) ?: 0) >= 8
            }
            .sortedByDescending { it.root.name }
        if (replayable.isEmpty()) {
            Toast.makeText(this, "No frozen SfM inputs", Toast.LENGTH_LONG).show()
            return
        }
        AlertDialog.Builder(this)
            .setTitle("Select hash-verified SfM input")
            .setItems(replayable.map { it.root.name }.toTypedArray()) { _, index ->
                beginRunFromFrozenReplay(replayable[index], "saved_run_replay")
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun beginRunFromFrozenReplay(source: ProjectWorkspace, sourceLabel: String) {
        verifyImportedProject(source.root)
        val sourceManifest = source.load()
        check(sourceManifest.extractedFrames >= 8) { "source run has no valid extraction" }
        if (manifest.captureId.isNotEmpty() || manifest.stage != PipelineStage.CAPTURE) {
            switchToFreshRun()
        }
        workspace.requireEmpty(workspace.captureDir, "run capture input")
        workspace.requireEmpty(workspace.sourceImages, "frozen frame replay")
        check(!workspace.framePriorsFile.exists()) { "frame priors already exist" }
        copyTreeNoOverwrite(source.captureDir, workspace.captureDir)
        copyTreeNoOverwrite(source.sourceImages, workspace.sourceImages)
        source.framePriorsFile.copyTo(workspace.framePriorsFile, overwrite = false)
        val sourceStageScreenshots = listOf("capture", "extract").associateWith { stage ->
            val relative = sourceManifest.screenshots[stage]
                ?: error("source run has no $stage screenshot record")
            File(source.root, relative).also { screenshot ->
                check(screenshot.isFile && screenshot.length() > 0L) {
                    "source run $stage screenshot is missing: $relative"
                }
                val expected = sourceManifest.artifactSha256[relative]
                    ?: error("source run has no hash for $stage screenshot")
                check(ProjectEvidence.sha256(screenshot).equals(
                    expected, ignoreCase = true)) {
                    "source run $stage screenshot hash mismatch"
                }
            }
        }
        sourceStageScreenshots.getValue("capture").let { screenshot ->
            val frozenCaptureScreenshot = workspace.captureDir.resolve("capture.png")
            if (!frozenCaptureScreenshot.exists()) {
                screenshot.copyTo(frozenCaptureScreenshot, overwrite = false)
            }
        }
        manifest = manifest.copy(
            captureId = sourceManifest.captureId,
            runId = workspace.root.name,
            autoRunToEnd = true,
            stage = PipelineStage.SFM,
            preset = preset,
            optimizationConfig = optimizationConfig,
            frameTargetCount = sourceManifest.frameTargetCount,
            pipelineConfig = ProjectEvidence.pipelineConfig(
                sourceManifest.frameTargetCount, optimizationConfig, preset,
                TILE_BIN_CAPACITY, sfmPreviewEnabled, sfmThreads, sfmCpus),
            captureSource = sourceLabel,
            extractedFrames = sourceManifest.extractedFrames,
            positionPriorCount = sourceManifest.positionPriorCount,
            gravityPriorCount = sourceManifest.gravityPriorCount,
            intrinsicsPriorPresent = sourceManifest.intrinsicsPriorPresent,
            lastError = null,
        ).also(workspace::save)
        sourceStageScreenshots.forEach { (stage, screenshot) ->
            val target = if (stage == "capture") {
                workspace.screenshotsDir.resolve("capture.png")
            } else {
                workspace.screenshotsDir.resolve(screenshot.name)
            }
            screenshot.copyTo(target, overwrite = false)
            recordScreenshot(stage, target)
        }
        refresh()
        runSfm()
    }

    /**
     * Start a run from a finished run's structure from motion, so that a
     * repeat changes only the optimization stage.
     *
     * Replaying from the capture or from frozen frames still re-runs COLMAP,
     * which is not bit-deterministic: the sparse geometry and the held-out
     * camera poses come out slightly different, and a failed or partial
     * registration would change the held-out split outright. Copying the
     * source's sparse model removes both, at the cost of not measuring the
     * stage again -- so the extract and SfM times are carried over as the
     * source measured them, and the manifest records which run they came
     * from rather than presenting them as this run's own.
     */
    private fun beginRunFromFrozenSfm(source: ProjectWorkspace, sourceLabel: String) {
        verifyImportedProject(source.root)
        val sourceManifest = source.load()
        check(sourceManifest.registeredImages >= 8) {
            "source run registered ${sourceManifest.registeredImages} images"
        }
        listOf("cameras.bin", "images.bin", "points3D.bin").forEach { name ->
            check(File(source.trainingSparse, name).isFile) {
                "source run has no $name to reuse"
            }
        }
        check(source.trainingImages.listFiles()?.isNotEmpty() == true) {
            "source run has no training images to reuse"
        }
        if (manifest.captureId.isNotEmpty() || manifest.stage != PipelineStage.CAPTURE) {
            switchToFreshRun()
        }
        workspace.requireEmpty(workspace.captureDir, "run capture input")
        workspace.requireEmpty(workspace.sourceImages, "frozen frame replay")
        workspace.requireEmpty(workspace.sfmWorkspace, "frozen SfM replay")
        workspace.requireEmpty(workspace.trainingRoot, "training scene")
        check(!workspace.framePriorsFile.exists()) { "frame priors already exist" }
        copyTreeNoOverwrite(source.captureDir, workspace.captureDir)
        copyTreeNoOverwrite(source.sourceImages, workspace.sourceImages)
        source.framePriorsFile.copyTo(workspace.framePriorsFile, overwrite = false)
        copyTreeNoOverwrite(source.sfmWorkspace, workspace.sfmWorkspace)
        copyTreeNoOverwrite(source.trainingRoot, workspace.trainingRoot)
        val sourceStageScreenshots = listOf("capture", "extract", "sfm")
            .associateWith { stage ->
                val relative = sourceManifest.screenshots[stage]
                    ?: error("source run has no $stage screenshot record")
                File(source.root, relative).also { screenshot ->
                    check(screenshot.isFile && screenshot.length() > 0L) {
                        "source run $stage screenshot is missing: $relative"
                    }
                    val expected = sourceManifest.artifactSha256[relative]
                        ?: error("source run has no hash for $stage screenshot")
                    check(ProjectEvidence.sha256(screenshot).equals(
                        expected, ignoreCase = true)) {
                        "source run $stage screenshot hash mismatch"
                    }
                }
            }
        sourceStageScreenshots.getValue("capture").let { screenshot ->
            val frozenCaptureScreenshot = workspace.captureDir.resolve("capture.png")
            if (!frozenCaptureScreenshot.exists()) {
                screenshot.copyTo(frozenCaptureScreenshot, overwrite = false)
            }
        }
        // The recorded model path points into the source run, so rebase it on
        // this workspace; leaving it would name a directory this run does not
        // own and the trainer would read a model it never copied.
        val selectedModel = sourceManifest.selectedSfmModel
            .takeIf(String::isNotEmpty)
            ?.let { File(it).relativeToOrNull(source.root) }
            ?.let { File(workspace.root, it.path).absolutePath }
            ?: error("source run records no selected SfM model")
        val carriedStageTimes = sourceManifest.stageTimesMs.filterKeys { key ->
            key == "extract" || key.startsWith("sfm")
        }
        check("extract" in carriedStageTimes && "sfm_total" in carriedStageTimes) {
            "source run has no extract and SfM stage times to carry"
        }
        manifest = manifest.copy(
            captureId = sourceManifest.captureId,
            runId = workspace.root.name,
            autoRunToEnd = true,
            stage = PipelineStage.OPTIMIZE,
            preset = preset,
            optimizationConfig = optimizationConfig,
            frameTargetCount = sourceManifest.frameTargetCount,
            pipelineConfig = ProjectEvidence.pipelineConfig(
                sourceManifest.frameTargetCount, optimizationConfig, preset,
                TILE_BIN_CAPACITY, sfmPreviewEnabled, sfmThreads, sfmCpus) + mapOf(
                "sfm_reused_from_run" to source.root.name,
                "sfm_stage_times_measured_in" to source.root.name),
            captureSource = sourceLabel,
            selectedSfmModel = selectedModel,
            extractedFrames = sourceManifest.extractedFrames,
            registeredImages = sourceManifest.registeredImages,
            meanReprojectionError = sourceManifest.meanReprojectionError,
            positionPriorCount = sourceManifest.positionPriorCount,
            gravityPriorCount = sourceManifest.gravityPriorCount,
            intrinsicsPriorPresent = sourceManifest.intrinsicsPriorPresent,
            stageTimesMs = manifest.stageTimesMs + carriedStageTimes,
            lastError = null,
        ).also(workspace::save)
        sourceStageScreenshots.forEach { (stage, screenshot) ->
            val target = if (stage == "capture") {
                workspace.screenshotsDir.resolve("capture.png")
            } else {
                workspace.screenshotsDir.resolve(screenshot.name)
            }
            screenshot.copyTo(target, overwrite = false)
            recordScreenshot(stage, target)
        }
        refresh()
        runOptimization()
    }

    private fun copyTreeNoOverwrite(source: File, destination: File) {
        check(source.isDirectory) { "source directory is missing: $source" }
        source.walkTopDown().filter(File::isFile).forEach { file ->
            val relative = file.relativeTo(source)
            val target = File(destination, relative.path)
            target.parentFile?.mkdirs()
            file.copyTo(target, overwrite = false)
        }
    }

    private fun beginRunFromCapture(
        capture: CaptureRecord,
        source: String,
        captureMs: Long = 0L,
        poseSamples: Int = 0,
        sensorSamples: Int = 0,
    ) {
        capture.requireComplete()
        sealCapture(capture)
        if (manifest.captureId.isNotEmpty() || manifest.stage != PipelineStage.CAPTURE) {
            switchToFreshRun()
        }
        workspace.requireEmpty(workspace.captureDir, "run capture input")
        capture.videoFile.copyTo(workspace.videoFile, overwrite = false)
        capture.rawArcorePosesFile.copyTo(
            workspace.rawArcorePosesFile, overwrite = false)
        if (capture.sensorFile.isFile) {
            capture.sensorFile.copyTo(workspace.sensorFile, overwrite = false)
        }
        if (capture.screenshotFile.isFile) {
            capture.screenshotFile.copyTo(
                File(workspace.captureDir, "capture.png"), overwrite = false)
        }
        capture.manifestFile.copyTo(
            File(workspace.captureDir, "capture_manifest.json"), overwrite = false)
        if (capture.screenshotFile.isFile) {
            val target = File(workspace.screenshotsDir, "capture.png")
            capture.screenshotFile.copyTo(target, overwrite = false)
            recordScreenshot("capture", target)
        }
        manifest = manifest.copy(
            captureId = capture.id,
            runId = workspace.root.name,
            autoRunToEnd = true,
            stage = PipelineStage.EXTRACT,
            captureSource = source,
            preset = preset,
            optimizationConfig = optimizationConfig,
            frameTargetCount = frameTargetCount,
            pipelineConfig = ProjectEvidence.pipelineConfig(
                frameTargetCount, optimizationConfig, preset,
                TILE_BIN_CAPACITY, sfmPreviewEnabled, sfmThreads, sfmCpus),
            stageTimesMs = if (captureMs > 0) {
                manifest.stageTimesMs + ("capture" to captureMs)
            } else manifest.stageTimesMs,
            capturePoseSamples = poseSamples,
            captureSensorSamples = sensorSamples,
            lastError = null,
        )
        workspace.save(manifest)
        refresh()
        extractFrames()
    }

    private fun switchToFreshRun() {
        ProjectTelemetry.stop()
        val runId = uniqueArtifactId("run")
        workspace = ProjectWorkspace.open(runsRoot, runId)
        manifest = ProjectManifest.create(runId).copy(
            runId = runId,
            preset = preset,
            optimizationConfig = optimizationConfig,
            frameTargetCount = frameTargetCount,
            environment = ProjectEvidence.environment(this),
            pipelineConfig = ProjectEvidence.pipelineConfig(
                frameTargetCount, optimizationConfig, preset,
                TILE_BIN_CAPACITY, sfmPreviewEnabled, sfmThreads, sfmCpus),
        ).also(workspace::save)
        ProjectTelemetry.start(this, workspace.telemetryFile, "capture")
        viewerReady = false
        firstViewEvidencePending.set(false)
        captureViewAfterNextRender.set(false)
        evidenceInventoryFinalizing.set(false)
        mainHandler.removeCallbacks(firstViewRenderTimeout)
        progress.progress = 0
        log.text = ""
    }

    private fun sealCapture(capture: CaptureRecord) {
        if (capture.manifestFile.isFile) {
            capture.verifySealed()
            return
        }
        val files = listOf(
            capture.videoFile,
            capture.rawArcorePosesFile,
            capture.sensorFile,
            capture.screenshotFile,
        ).filter(File::isFile)
        val hashes = org.json.JSONObject()
        files.forEach { file ->
            hashes.put(file.name, ProjectEvidence.sha256(file))
        }
        val json = org.json.JSONObject().apply {
            put("schema_version", 1)
            put("capture_id", capture.id)
            put("sealed_at_epoch_ms", System.currentTimeMillis())
            put("artifact_sha256", hashes)
        }
        val pending = File(capture.root, "capture_manifest.json.pending")
        check(!pending.exists() && !capture.manifestFile.exists())
        pending.writeText(json.toString(2) + "\n")
        check(pending.renameTo(capture.manifestFile)) {
            "failed to seal immutable capture manifest"
        }
        capture.verifySealed()
    }

    @Deprecated("Legacy result API keeps the benchmark module dependency-free")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQUEST_CAPTURE || resultCode != RESULT_OK) return
        val root = data?.getStringExtra(ArCaptureActivity.EXTRA_CAPTURE_ROOT)
            ?.let(::File) ?: return
        val capture = pendingCapture?.takeIf {
            it.root.canonicalFile == root.canonicalFile
        } ?: CaptureRecord(root)
        pendingCapture = null
        val captureMs = data.getLongExtra(
            ArCaptureActivity.EXTRA_CAPTURE_DURATION_MS, 0L)
        beginRunFromCapture(
            capture,
            "on_device_arcore_capture",
            captureMs,
            data.getIntExtra(ArCaptureActivity.EXTRA_POSE_COUNT, 0),
            data.getIntExtra(ArCaptureActivity.EXTRA_SENSOR_COUNT, 0),
        )
    }

    private fun extractFrames() {
        check(manifest.stage == PipelineStage.EXTRACT) {
            "frame extraction requires stage=extract, got ${manifest.stage}"
        }
        ProjectTelemetry.setStage("extract")
        setBusy(true, "Preprocessing: Extracting sharp frames")
        stageThread {
            val started = SystemClock.elapsedRealtime()
            val succeeded = runCatching {
                workspace.requireEmpty(workspace.sourceImages, "frame extraction")
                check(!workspace.framePriorsFile.exists()) {
                    "frame priors already exist; create a new run instead of overwriting"
                }
                val poses = workspace.rawArcorePosesFile.takeIf(File::isFile)
                    ?.let(PosesFile::parse)
                val count = VideoFrameExtractor.extract(
                    this,
                    android.net.Uri.fromFile(workspace.videoFile),
                    workspace.sourceImages,
                    manifest.frameTargetCount,
                    poses,
                    workspace.framePriorsFile,
                ) { done, total ->
                    runOnUiThread {
                        progress.max = total
                        progress.progress = done
                        status.text = "Preprocessing: Extracting sharp frames $done/$total"
                    }
                    // Show the frame this window just picked (frame_%04d.jpg
                    // of window done - 1), if it picked one.
                    val picked = File(workspace.sourceImages, "frame_%04d.jpg".format(done - 1))
                    if (picked.isFile && framePreviewBusy.compareAndSet(false, true)) {
                        framePreviewExecutor.execute {
                            try {
                                runCatching { showSourceFrame(picked) }
                            } finally {
                                framePreviewBusy.set(false)
                            }
                        }
                    }
                }
                check(count >= 8) { "only $count frames extracted" }
                val priorStats = readPriorStats(workspace.framePriorsFile)
                manifest = manifest.copy(
                    stage = PipelineStage.SFM,
                    extractedFrames = count,
                    positionPriorCount = priorStats.first,
                    gravityPriorCount = priorStats.second,
                    intrinsicsPriorPresent = priorStats.third,
                    stageTimesMs = manifest.stageTimesMs +
                        ("extract" to (SystemClock.elapsedRealtime() - started)),
                )
                workspace.save(manifest)
            }.onFailure(::recordFailure).isSuccess
            // On the preview executor, so it lands after any frame still
            // being shown.
            if (succeeded) {
                runCatching { framePreviewExecutor.submit { showMiddleSourceFrame() }.get() }
            }
            runOnUiThread {
                holdStatusForScreenshot()
                captureStageScreenshot(
                    "extract",
                    onSuccess = {
                        if (succeeded && manifest.autoRunToEnd) {
                            runSfm()
                        } else {
                            setBusy(false)
                            refresh()
                        }
                    },
                )
            }
        }
    }

    /**
     * Shows the SfM library's live preview (SIFT keypoints, pair matrix,
     * growing sparse model) on the surface and its caption in the status
     * line. Display only: the preview never changes the SfM result, and the
     * stage screenshot records its last frame as shown.
     */
    private fun startSfmPreview() {
        val width = surface.width.takeIf { it > 0 } ?: 1600
        val height = surface.height.takeIf { it > 0 } ?: 900
        val scale = minOf(1f, SFM_PREVIEW_MAX_EXTENT / maxOf(width, height))
        SfmNativeBridge.previewListener = { frame ->
            NativeBridge.showRgba(frame.rgba, frame.width, frame.height)
            runOnUiThread {
                status.text = sfmStatus(frame.caption)
                capturePreviewScreenshotIfDue(frame)
            }
        }
        SfmNativeBridge.setPreviewSize(
            (width * scale).roundToInt(), (height * scale).roundToInt())
    }

    /**
     * Saves the optional SfM screenshots: the SIFT keypoints and the pair
     * matrix, each at the first preview frame at least half way through its
     * step. A failed copy is only logged. UI thread.
     */
    private fun capturePreviewScreenshotIfDue(frame: SfmNativeBridge.PreviewFrame) {
        val shot = PREVIEW_SCREENSHOTS[frame.stage] ?: return
        // A second capture would cancel the pending one (screenshotGeneration);
        // a later frame of the step tries again.
        if (shot.stage in previewShotsRequested || previewShotPending) return
        val progress = shot.caption.find(frame.caption) ?: return
        val done = progress.groupValues[1].replace(",", "").toLong()
        val total = progress.groupValues[2].replace(",", "").toLong()
        if (total <= 0 || done * 2 < total) return
        previewShotsRequested += shot.stage
        previewShotPending = true
        // Delayed so the new status line is drawn before the copy.
        mainHandler.postDelayed({
            captureStageScreenshot(
                shot.stage,
                onSuccess = { previewShotPending = false },
                onFailure = { error ->
                    previewShotPending = false
                    appendLog("${shot.stage} screenshot failed (${error.message}); continuing")
                },
                onSaved = { previewShots[shot.stage] = it },
            )
        }, 250L)
    }

    /** The SfM preview caption with its step spelled out, after "SfM: ". */
    private fun sfmStatus(caption: String): String {
        val step = SFM_STEP_NAMES.entries.firstOrNull { caption.startsWith(it.key) }
        return "SfM: " + (step?.let { it.value + caption.removePrefix(it.key) } ?: caption)
    }

    private fun stopSfmPreview() {
        SfmNativeBridge.setPreviewSize(0, 0)
        SfmNativeBridge.previewListener = null
    }

    private fun runSfm() {
        check(manifest.stage == PipelineStage.SFM) {
            "SfM requires stage=sfm, got ${manifest.stage}"
        }
        ProjectTelemetry.setStage("sfm")
        setBusy(true, "SfM: Running incremental reconstruction")
        startService(Intent(this, RunKeepAliveService::class.java))
        previewShotsRequested.clear()
        previewShots.clear()
        previewShotPending = false
        // The run's frozen configuration decides, not this launch's intent.
        if (manifest.pipelineConfig["sfm_live_preview"] != "disabled") startSfmPreview()
        val threads = manifest.pipelineConfig["sfm_threads"]
            ?.split(',')?.mapNotNull(String::toIntOrNull)?.takeIf { it.size == 3 }
            ?: listOf(0, 0, 0)
        SfmNativeBridge.setThreads(threads[0], threads[1], threads[2])
        SfmNativeBridge.setCpus(manifest.pipelineConfig["sfm_cpus"] ?: "all")
        stageThread {
            val started = SystemClock.elapsedRealtime()
            synchronized(sfmTimingLock) {
                sfmStageTimes.clear()
                sfmSubstage = null
                sfmSubstageStartedMs = 0L
            }
            val succeeded = runCatching {
                manifest = ProjectPipeline(workspace, ::appendLog)
                    .runSfm(manifest.copy(stage = PipelineStage.SFM))
                manifest = manifest.copy(
                    stageTimesMs = manifest.stageTimesMs +
                        ("sfm_total" to (SystemClock.elapsedRealtime() - started)) +
                        synchronized(sfmTimingLock) { sfmStageTimes.toMap() },
                ).also(workspace::save)
            }.onFailure(::recordFailure).isSuccess
            // No preview frame arrives after the run returns; the surface
            // keeps the final sparse-model frame until training draws.
            stopSfmPreview()
            stopService(Intent(this, RunKeepAliveService::class.java))
            runOnUiThread {
                // Recorded only now, because the SfM thread replaced the
                // manifest while it ran.
                previewShots.forEach { (stage, file) ->
                    if (file.isFile) recordScreenshot(stage, file)
                }
                previewShots.clear()
                holdStatusForScreenshot()
                captureStageScreenshot(
                    "sfm",
                    onSuccess = {
                        if (succeeded && manifest.autoRunToEnd) {
                            runOptimization()
                        } else {
                            setBusy(false)
                            refresh()
                        }
                    },
                )
            }
        }
    }

    private fun runOptimization() {
        check(manifest.stage == PipelineStage.OPTIMIZE) {
            "optimization requires stage=optimize, got ${manifest.stage}"
        }
        ProjectTelemetry.setStage("optimize")
        workspace.requireEmpty(workspace.output, "optimization")
        manifest = manifest.copy(
            stage = PipelineStage.OPTIMIZE,
            preset = preset,
            optimizationConfig = optimizationConfig,
            lastError = null,
        )
        workspace.save(manifest)
        optimizationStartedMs = SystemClock.elapsedRealtime()
        trainingShot = TrainingShot.NONE
        afterTrainingShot = null
        val runConfig = manifest.optimizationConfig
        val runPreset = manifest.preset
        progress.max = runConfig.iterations
        progress.progress = 0
        startService(Intent(this, RunKeepAliveService::class.java))
        // Captures are trained at the extraction resolution, which is far
        // larger than the benchmark scenes' images_4, so the historical 512
        // per-tile pair budget can be exhausted by a scene whose splats each
        // cover many tiles. Raising it is what keeps the rasterizer from
        // silently dropping pairs; the run is rejected below if it still does.
        check(NativeBridge.setTileBinCapacity(TILE_BIN_CAPACITY)) {
            "failed to set the tile-Gaussian pair budget"
        }
        check(NativeBridge.setTrainingPreviewPanels(true)) {
            "failed to enable the training preview panels"
        }
        val started = NativeBridge.startTraining(
            workspace.root.absolutePath,
            workspace.output.absolutePath,
            manifest.imageDirectory,
            manifest.sparseDirectory,
            runConfig.iterations,
            0,
            25,
            0,
            0.2f,
            "",
            runConfig.shDegree,
            runConfig.densityStrategy,
            runConfig.gaussianCapacity,
            "random",
            "fp32",
            runPreset.fullImageSsim,
            runPreset.mcmcCloneOffset,
            runPreset.mcmcGradientSampling,
            false,
            "auto",
        )
        if (!started) {
            stopService(Intent(this, RunKeepAliveService::class.java))
            recordFailure(IllegalStateException("optimizer already running"))
        } else {
            // Optimization runs on a native thread, so it is counted here and
            // released in onFinished rather than through stageThread.
            activeStages.incrementAndGet()
            setBusy(true, "3DGS optimization: ${runConfig.iterations} updates / " +
                "${runConfig.gaussianCapacity} max Gaussians")
        }
    }

    private fun recordFailure(error: Throwable) {
        manifest = manifest.copy(
            stage = PipelineStage.FAILED,
            lastError = error.message ?: error.javaClass.simpleName,
        )
        workspace.save(manifest)
        appendLog("ERROR: ${manifest.lastError}")
    }

    /**
     * Run [body] on a stage thread that this process counts, so a later
     * activity can tell that work is still going.
     */
    private fun stageThread(body: () -> Unit) {
        activeStages.incrementAndGet()
        thread {
            try {
                body()
            } finally {
                activeStages.decrementAndGet()
            }
        }
    }

    /**
     * True when a stage is already working, in which case the caller must not
     * start another: they would share the cores and, if the new one takes over
     * this activity's workspace, the old one would keep writing into a run
     * nothing is watching.
     */
    private fun refuseWhileRunning(): Boolean {
        if (activeStages.get() == 0) return false
        Toast.makeText(
            this,
            "A run is still working. Stop it from the notification first.",
            Toast.LENGTH_LONG,
        ).show()
        return true
    }

    private fun setBusy(busy: Boolean, message: String? = null) {
        runOnUiThread {
            captureButton.isEnabled = !busy
            savedCaptureButton.isEnabled = !busy
            frozenReplayButton.isEnabled = !busy
            openFinishedButton.isEnabled = !busy
            extractButton.isEnabled = !busy
            sfmButton.isEnabled = !busy
            optimizeButton.isEnabled = !busy
            presetButton.isEnabled = !busy
            frameTargetButton.isEnabled = !busy
            optimizationSettingsButton.isEnabled = !busy
            viewButton.isEnabled = !busy
            message?.let { status.text = it }
        }
    }

    private fun refresh() {
        presetButton.text = "Preset: ${preset.name}"
        frameTargetButton.text = "Extract frames: $frameTargetCount"
        optimizationSettingsButton.text =
            "3DGS ${optimizationConfig.iterations}it " +
                "${optimizationConfig.gaussianCapacity / 1000}k " +
                "SH${optimizationConfig.shDegree} " +
                optimizationConfig.densityStrategy.uppercase()
        status.text = "Project ${manifest.projectId} | stage ${manifest.stage.name}" +
            (manifest.lastError?.let { " | $it" } ?: "")
        extractButton.isEnabled = manifest.stage == PipelineStage.EXTRACT &&
            workspace.videoFile.isFile
        sfmButton.isEnabled = manifest.stage == PipelineStage.SFM &&
            workspace.sourceImages.listFiles()?.isNotEmpty() == true
        optimizeButton.isEnabled = manifest.stage == PipelineStage.OPTIMIZE &&
            workspace.trainingSparse.isDirectory
        viewButton.isEnabled = (manifest.stage == PipelineStage.VIEW ||
            manifest.stage == PipelineStage.COMPLETE) &&
            File(workspace.output, "checkpoint.d3gs").isFile
    }

    private fun appendLog(line: String) {
        val stamped = "${System.currentTimeMillis()} $line"
        synchronized(workspace.pipelineLog) {
            if (manifest.stage != PipelineStage.COMPLETE &&
                !evidenceInventoryFinalizing.get()) {
                workspace.pipelineLog.appendText(stamped + "\n")
            }
        }
        runOnUiThread {
            log.append(stamped + "\n")
            // Follow the newest line so a long stage visibly progresses.
            // Posted, because the scroll has to happen after the text view
            // has laid out the line that was just appended.
            logScroll.post { logScroll.fullScroll(View.FOCUS_DOWN) }
        }
    }

    private fun onSfmLogLine(line: String) {
        val now = SystemClock.elapsedRealtime()
        val next = when {
            line.contains("=== Feature extraction ===") -> "sfm_feature_extraction"
            line.contains("=== Feature matching") -> "sfm_feature_matching"
            line.contains("=== Incremental mapping ===") ||
                line.contains("=== Global mapping ===") -> "sfm_mapping"
            line.contains("=== Done:") -> "done"
            else -> null
        }
        if (next != null) synchronized(sfmTimingLock) {
            sfmSubstage?.let { previous ->
                sfmStageTimes[previous] = now - sfmSubstageStartedMs
            }
            if (next == "done") {
                sfmSubstage = null
                sfmSubstageStartedMs = 0L
            } else {
                sfmSubstage = next
                sfmSubstageStartedMs = now
            }
        }
        appendLog(line)
    }

    private fun readPriorStats(file: File): Triple<Int, Int, Boolean> {
        if (!file.isFile) return Triple(0, 0, false)
        var positions = 0
        var gravity = 0
        var intrinsics = false
        file.forEachLine { raw ->
            val line = raw.trim()
            if (line.startsWith("# intrinsics ")) intrinsics = true
            if (line.isEmpty() || line.startsWith("#")) return@forEachLine
            val fields = line.split(Regex("\\s+"))
            if (fields.size >= 8) {
                positions++
                val g = fields.slice(4..6).mapNotNull(String::toDoubleOrNull)
                if (g.size == 3 && kotlin.math.sqrt(g.sumOf { it * it }) > 0.5) {
                    gravity++
                }
            }
        }
        return Triple(positions, gravity, intrinsics)
    }

    /**
     * [relaunched]: the activity was recreated or reopened from Recents.
     * Android then re-delivers the original launch intent, so its run-starting
     * extras (capture_bundle, auto_start) would start yet another replay run
     * every time the user returns to the app; they act only on a fresh launch.
     */
    private fun configureAutomation(relaunched: Boolean) {
        intent.getStringExtra("capture_bundle")?.takeUnless { relaunched }?.let { path ->
            val bundle = File(path)
            require(bundle.isDirectory) { "capture_bundle is not a directory: $path" }
            verifyImportedProject(bundle)
            // Opt-in, because replaying frozen frames and re-running COLMAP is
            // what an existing caller asks for by naming a finished run here.
            if (intent.getBooleanExtra("reuse_sfm", false)) {
                require(File(bundle, "training/sparse/0/images.bin").isFile) {
                    "reuse_sfm needs a source run with a sparse model: $path"
                }
                status.post {
                    beginRunFromFrozenSfm(
                        ProjectWorkspace.inspect(bundle), "imported_frozen_sfm")
                }
                return
            }
            val frozenImages = File(bundle, "extract/images")
            val frozenPriors = File(bundle, "extract/frame_priors.txt")
            if (frozenImages.isDirectory && frozenPriors.isFile) {
                status.post {
                    beginRunFromFrozenReplay(
                        ProjectWorkspace.inspect(bundle),
                        "imported_frozen_run",
                    )
                }
                return
            }
            val imported = captureStore.importProjectBundle(bundle)
            sealCapture(imported)
            status.post { beginRunFromCapture(imported, "imported_capture_bundle") }
            return
        }
        if (manifest.stage == PipelineStage.VIEW &&
            manifest.screenshots.containsKey("view")) {
            status.post { prepareViewerAndRender() }
            return
        }
        if (relaunched || !intent.getBooleanExtra("auto_start", false)) return
        manifest = manifest.copy(autoRunToEnd = true).also(workspace::save)
        status.post {
            when (manifest.stage) {
                PipelineStage.EXTRACT -> extractFrames()
                PipelineStage.SFM -> runSfm()
                PipelineStage.OPTIMIZE -> runOptimization()
                PipelineStage.VIEW, PipelineStage.COMPLETE -> prepareViewerAndRender()
                else -> Unit
            }
        }
    }

    private fun verifyImportedProject(bundle: File) {
        val sourceManifest = File(bundle, "project_manifest.json")
        if (!sourceManifest.isFile) {
            val captureRoot = if (File(bundle, "capture_manifest.json").isFile) {
                bundle
            } else {
                File(bundle, "input/capture")
            }
            val captureManifest = File(captureRoot, "capture_manifest.json")
            check(captureManifest.isFile) {
                "capture_bundle is neither a hashed run nor a sealed capture"
            }
            val captureJson = org.json.JSONObject(captureManifest.readText())
            val hashes = captureJson.optJSONObject("artifact_sha256")
                ?: error("sealed capture has no artifact_sha256 inventory")
            check(hashes.length() >= 4) { "sealed capture inventory is incomplete" }
            hashes.keys().forEach { name ->
                check(!name.contains('/') && !name.contains('\\')) {
                    "capture artifact must be a direct child: $name"
                }
                val file = File(captureRoot, name)
                check(file.isFile) { "capture artifact missing: $name" }
                check(ProjectEvidence.sha256(file).equals(
                    hashes.getString(name), ignoreCase = true)) {
                    "capture artifact SHA-256 mismatch: $name"
                }
            }
            return
        }
        val imported = ProjectManifest.fromJson(
            org.json.JSONObject(sourceManifest.readText()))
        check(imported.artifactSha256.isNotEmpty()) {
            "project capture_bundle has no artifact_sha256 inventory"
        }
        imported.artifactSha256.forEach { (relative, expected) ->
            val file = File(bundle, relative)
            check(file.isFile) { "capture_bundle artifact missing: $relative" }
            val actual = ProjectEvidence.sha256(file)
            check(actual.equals(expected, ignoreCase = true)) {
                "capture_bundle SHA-256 mismatch: $relative"
            }
        }
    }

    override fun onLog(line: String) = appendLog(line)

    override fun onProgress(
        iteration: Int,
        total: Int,
        loss: Float,
        elapsedSeconds: Double,
    ) {
        runOnUiThread {
            progress.max = total
            progress.progress = iteration
            status.text = "3DGS optimization: $iteration/$total | loss %.6f | %.1f s".format(
                loss, elapsedSeconds)
            placeTrainingLabels()
            // The optimize screenshot shows training in progress: the first
            // report past 90% of the updates, with the trainer's render of
            // that report, its status line and the progress bar.
            if (trainingShot == TrainingShot.NONE && total > 0 &&
                iteration * 10L >= total * 9L) {
                trainingShot = TrainingShot.PENDING
                // Delayed so the new status line is drawn before the copy.
                mainHandler.postDelayed({
                    captureStageScreenshot(
                        "optimize",
                        onSuccess = { settleTrainingShot(TrainingShot.DONE) },
                        onFailure = { error ->
                            appendLog("Optimize screenshot during training failed " +
                                "(${error.message}); taking it after training")
                            settleTrainingShot(TrainingShot.FAILED)
                        },
                    )
                }, 250L)
            }
        }
    }

    /**
     * Updates the controls for the stage just finished but leaves the status
     * line as the stage last set it, so the stage screenshot shows the stage
     * at work rather than a bookkeeping message. UI thread.
     */
    private fun holdStatusForScreenshot() {
        val shown = status.text.toString()
        refresh()
        setBusy(true, shown)
    }

    private fun settleTrainingShot(result: TrainingShot) {
        if (trainingShot != TrainingShot.PENDING) return
        trainingShot = result
        val next = afterTrainingShot
        afterTrainingShot = null
        next?.invoke()
    }

    /**
     * Dropped tile-Gaussian pairs, or -1 when the count cannot be read.
     *
     * A run that exceeds the pair buffer still exits zero, but its renders are
     * not what its model represents, so the result must not be presented as a
     * finished reconstruction.
     */
    private fun droppedPairs(): Long = runCatching {
        val bench = File(workspace.output, "bench.json")
        if (!bench.isFile) return@runCatching -1L
        org.json.JSONObject(bench.readText()).optLong("overflow_count", -1L)
    }.getOrDefault(-1L)

    override fun onFinished(exitCode: Int, message: String, plyPath: String) {
        if (exitCode != 0) runOnUiThread { trainingLabels.visibility = View.GONE }
        activeStages.decrementAndGet()
        stopService(Intent(this, RunKeepAliveService::class.java))
        val dropped = if (exitCode == 0) droppedPairs() else 0L
        val verdict = when {
            dropped > 0L -> "$message; run rejected: the rasterizer dropped " +
                "$dropped tile-Gaussian pairs, so the reconstruction is " +
                "incomplete. Retry with a tile bin capacity above " +
                "$TILE_BIN_CAPACITY, or a smaller Gaussian capacity."
            dropped < 0L -> "$message; run rejected: bench.json carries no " +
                "overflow_count, so pair completeness cannot be established."
            else -> message
        }
        val verdictCode = if (dropped != 0L) 1 else exitCode
        val optimizeMs = SystemClock.elapsedRealtime() - optimizationStartedMs
        // The manifest changes on the UI thread, where the optimize screenshot
        // taken during training records itself.
        runOnUiThread {
            manifest = if (verdictCode == 0) {
                manifest.copy(
                    stage = PipelineStage.VIEW,
                    stageTimesMs = manifest.stageTimesMs + ("optimize_total" to optimizeMs),
                    lastError = null,
                )
            } else {
                manifest.copy(stage = PipelineStage.FAILED, lastError = verdict)
            }
            workspace.save(manifest)
            appendLog(verdict)
            if (verdictCode != 0) {
                setBusy(false)
                refresh()
                return@runOnUiThread
            }
            val proceed: () -> Unit = {
                if (manifest.autoRunToEnd) {
                    prepareViewerAndRender()
                } else {
                    setBusy(false)
                    refresh()
                }
            }
            val afterShot: () -> Unit = afterShot@{
                if (trainingShot == TrainingShot.DONE) {
                    proceed()
                    return@afterShot
                }
                // No screenshot from training: take it now, with the last
                // progress line still on screen.
                holdStatusForScreenshot()
                mainHandler.postDelayed({
                    captureStageScreenshot("optimize", onSuccess = proceed)
                }, 750L)
            }
            if (trainingShot == TrainingShot.PENDING) {
                afterTrainingShot = afterShot
            } else {
                afterShot()
            }
        }
    }

    /**
     * Saves a full-window screenshot for [stage] and records it in the
     * manifest, or hands the saved file to [onSaved] instead when the caller
     * records it later.
     */
    private fun captureStageScreenshot(
        stage: String,
        onSuccess: () -> Unit = {},
        onFailure: (Throwable) -> Unit = { error ->
            recordFailure(error)
            setBusy(false)
            refresh()
        },
        onSaved: ((File) -> Unit)? = null,
    ) {
        val generation = screenshotGeneration.incrementAndGet()
        val target = runCatching {
            File(
                workspace.screenshotsDir,
                "${uniqueArtifactId(stage)}.png",
            ).also {
                check(!it.exists()) { "screenshot target already exists: $it" }
            }
        }.getOrElse { error ->
            onFailure(error)
            return
        }
        val staging = File(
            cacheDir,
            "capture_to_view_screenshots/${workspace.root.name}/${target.name}",
        )
        staging.parentFile?.mkdirs()
        if (staging.exists()) {
            onFailure(IllegalStateException(
                "screenshot staging target already exists: $staging"))
            return
        }
        // The screenshot copies what the surface shows: the middle extracted
        // frame (extract), the last SfM preview frame (sfm), the trainer's
        // progress render (optimize), or the viewer (view). These files stand in
        // only when the surface copy fails or comes back blank.
        val surfaceFallback = when (stage) {
            "extract", "sfm" -> middleSourceImage()
            "optimize" -> File(workspace.output, "render")
                .listFiles { file -> file.isFile && file.extension.equals("png", true) }
                .orEmpty()
                .sortedBy(File::getName)
                .firstOrNull()
            else -> null
        }
        ProjectEvidence.captureWindow(
            this,
            staging,
            SCREENSHOT_TIMEOUT_MS,
            surfaceView = surface,
            fallbackSurfaceImage = surfaceFallback,
            requireSurfaceContent = surfaceFallback != null ||
                stage == "optimize" || stage == "view" || stage in OPTIONAL_SCREENSHOT_STAGES,
            overlayViews = listOf(trainingLabels),
        ) { ok ->
            val active = generation == screenshotGeneration.get() &&
                !isFinishing && !isDestroyed
            if (!active) {
                runCatching { staging.delete() }
                return@captureWindow
            }
            if (!ok) {
                onFailure(IllegalStateException(
                    "automatic screenshot timed out or failed for stage=$stage"))
                return@captureWindow
            }
            val recorded = runCatching {
                check(staging.renameTo(target)) {
                    "failed to publish screenshot from staging: $staging"
                }
                if (onSaved != null) onSaved(target) else recordScreenshot(stage, target)
                appendLog(
                    "Automatic screenshot saved: ${target.relativeTo(workspace.root)}")
            }
            recorded.onFailure(onFailure)
            if (recorded.isSuccess) runCatching(onSuccess).onFailure(onFailure)
            runCatching { staging.delete() }
        }
    }

    private fun middleSourceImage(): File? = workspace.sourceImages
        .listFiles { file -> file.isFile && file.extension.equals("jpg", true) }
        .orEmpty()
        .sortedBy(File::getName)
        .let { images -> images.getOrNull(images.size / 2) }

    /**
     * Shows the middle extracted frame on the surface when extraction ends, so
     * the extract screen and its screenshot show the same frame. Worker thread.
     */
    private fun showMiddleSourceFrame() {
        middleSourceImage()?.let(::showSourceFrame)
    }

    /** Shows one extracted frame on the surface. Worker thread. */
    private fun showSourceFrame(file: File) {
        val longest = maxOf(surface.width, surface.height).coerceAtLeast(1)
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= longest) sample *= 2
        val bitmap = BitmapFactory.decodeFile(
            file.absolutePath,
            BitmapFactory.Options().apply {
                inSampleSize = sample
                inPreferredConfig = Bitmap.Config.ARGB_8888
            },
        ) ?: return
        try {
            // ARGB_8888 pixels are stored as R, G, B, A bytes.
            val rgba = ByteArray(bitmap.byteCount)
            bitmap.copyPixelsToBuffer(ByteBuffer.wrap(rgba))
            NativeBridge.showRgba(rgba, bitmap.width, bitmap.height)
        } finally {
            bitmap.recycle()
        }
    }

    private fun recordScreenshot(stage: String, file: File) {
        manifest = manifest.copy(
            screenshots = manifest.screenshots +
                (stage to file.relativeTo(workspace.root).invariantSeparatorsPath),
            artifactSha256 = manifest.artifactSha256 +
                (file.relativeTo(workspace.root).invariantSeparatorsPath to
                    ProjectEvidence.sha256(file)),
        )
        workspace.save(manifest)
    }

    override fun onDestroy() {
        screenshotGeneration.incrementAndGet()
        mainHandler.removeCallbacks(firstViewRenderTimeout)
        firstViewEvidencePending.set(false)
        captureViewAfterNextRender.set(false)
        NativeBridge.setSurface(null)
        NativeBridge.setListener(null)
        if (SfmNativeBridge.isAvailable()) {
            SfmNativeBridge.logListener = null
        }
        viewerExecutor.shutdownNow()
        framePreviewExecutor.shutdownNow()
        if (isFinishing) ProjectTelemetry.stop()
        super.onDestroy()
    }

    private enum class TrainingShot { NONE, PENDING, DONE, FAILED }

    private class PreviewShot(val stage: String, val caption: Regex)

    companion object {
        /**
         * Pipeline stage threads running anywhere in this process.
         *
         * Static, because the keep-alive service deliberately outlives this
         * screen: a stage started from an activity that has since been
         * destroyed keeps computing, and a freshly created activity knows
         * nothing about it. Starting a second run on top of the first is what
         * leaves two structure-from-motion passes competing for the cores.
         */
        private val activeStages = AtomicInteger(0)

        /**
         * Per-tile tile-Gaussian pair budget requested before every run.
         *
         * pair_capacity is this times the tile count. At the 1920x1080
         * extraction resolution that is 8,160 tiles. 2048 matches the
         * desktop path (16.7 M pairs, ~250 MB over the 512 default at
         * 20 bytes per pair), so a full-preset run does not spend hours
         * training and then fail the overflow gate.
         */
        private const val TILE_BIN_CAPACITY = 2048
        private const val REQUEST_CAPTURE = 41
        private const val DEFAULT_FRAME_TARGET_COUNT = 60
        private const val MIN_FRAME_TARGET_COUNT = 8
        private const val MAX_FRAME_TARGET_COUNT = 240
        private const val MIN_TRAINING_ITERATIONS = 1
        private const val MAX_TRAINING_ITERATIONS = 100_000
        private const val MIN_GAUSSIAN_CAPACITY = 10_000
        private const val MAX_GAUSSIAN_CAPACITY = 2_000_000
        private const val SCREENSHOT_TIMEOUT_MS = 12_000L
        private const val FIRST_VIEW_RENDER_TIMEOUT_MS = 20_000L
        private const val VIEW_RENDER_RETRY_MS = 250L
        private const val FINALIZER_POLL_MS = 500L
        private const val VIEWER_SWEEP_FRAMES = 60
        /** Radians the view turns for a drag across the whole surface. */
        private const val ORBIT_TURN_PER_SWEEP = 3.14159f
        private const val VIEWER_SWEEP_YAW_SPAN = 0.7f
        /** Longest side of the SfM live preview, like the viewer's render cap. */
        private const val SFM_PREVIEW_MAX_EXTENT = 960f
        private const val VIEWER_SWEEP_TIMEOUT_MS = 30_000L
        private val REQUIRED_SCREENSHOT_STAGES =
            setOf("capture", "extract", "sfm", "optimize", "view")
        /**
         * Optional screenshots during SfM, by preview stage
         * (CROSS3DGS_SFM_STAGE_* in cross3dgs_sfm.h), with the progress at the
         * start of that stage's caption (sfm_preview.cc). Absent when the live
         * preview is off.
         */
        private val PREVIEW_SCREENSHOTS = mapOf(
            1 to PreviewShot("sfm_features", Regex("""^SIFT ([\d,]+)/([\d,]+)""")),
            2 to PreviewShot("sfm_matching", Regex("""^Matching ([\d,]+)/([\d,]+) pairs""")),
        )
        private val OPTIONAL_SCREENSHOT_STAGES =
            PREVIEW_SCREENSHOTS.values.map(PreviewShot::stage).toSet()
        /** Step names for the leading words of the SfM preview captions. */
        private val SFM_STEP_NAMES = mapOf(
            "SIFT " to "SIFT extraction ",
            "Matching " to "Feature matching ",
            "Incremental SfM " to "Incremental mapping ",
            "Sparse model " to "Incremental mapping done ",
        )
        private val ACTIVE_FINAL_INVENTORIES =
            ConcurrentHashMap.newKeySet<String>()
    }
}
