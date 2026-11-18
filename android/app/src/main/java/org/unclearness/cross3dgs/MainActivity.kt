package org.unclearness.cross3dgs

import android.annotation.SuppressLint
import android.app.Activity
import android.app.AlertDialog
import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.text.InputType
import android.view.Gravity
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.WindowInsets
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.concurrent.thread

@SuppressLint("SetTextI18n")
class MainActivity : Activity(), NativeBridge.Listener {
    private lateinit var capsText: TextView
    private lateinit var statusText: TextView
    private lateinit var sceneSpinner: Spinner
    private lateinit var viewsText: EditText
    private lateinit var iterationsText: EditText
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var backwardButton: Button
    private lateinit var progress: ProgressBar
    private lateinit var resultBar: LinearLayout
    private lateinit var modeButton: Button
    private lateinit var outputText: TextView
    private lateinit var exportButton: Button
    private lateinit var shareButton: Button
    private lateinit var howToViewButton: Button
    private lateinit var logText: TextView
    private lateinit var logScroll: ScrollView

    private val scenes = listOf(
        "bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump")
    private var dataRoot: File? = null
    private var selectedScene = "bicycle"
    private var lastPlyFile: File? = null
    private var sceneDir: File? = null
    private var selectedView = 0
    private var showTarget = false
    private var backwardSamples = 0
    private var progressEvery = 25
    private var ssimWeight = 0.0f
    private var initCheckpoint = ""
    private var shDegree = 0
    private var densityStrategy = "fixed"
    private var gaussianCapacity = 0
    private var viewOrder = "cyclic"
    private var adamMoments = "bf16"
    private var fullImageSsim = false
    private var mcmcCloneOffset = 0.0f
    private var mcmcGradientSampling = 0.0f
    private var forceAtomicFreeExact = false
    private var backwardCandidate = "auto"
    private var trainingViewSeed = 42
    private var initSeed = 42
    private var tileBinCapacity = 2048
    private val resultButtons = mutableListOf<Button>()
    private val logLines = ArrayDeque<String>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
        val requestedIterations =
            intent.getIntExtra("iterations", 1000).coerceAtLeast(1)
        val autoStart = intent.getBooleanExtra("auto_start", false)
        backwardSamples = intent.getIntExtra("sobol_samples", 0)
            .takeIf { it == 0 || it == 1 || it == 2 || it == 4 || it == 8 }
            ?: 0
        progressEvery = intent.getIntExtra("progress_every", 25)
            .coerceAtLeast(1)
        ssimWeight = intent.getFloatExtra("ssim_weight", 0.0f)
            .coerceIn(0.0f, 1.0f)
        initCheckpoint = intent.getStringExtra("init_checkpoint") ?: ""
        shDegree = intent.getIntExtra("sh_degree", 0).coerceIn(0, 3)
        densityStrategy = intent.getStringExtra("density_strategy")
            ?.takeIf { it == "fixed" || it == "mcmc" } ?: "fixed"
        gaussianCapacity = intent.getIntExtra("gaussian_capacity", 0)
            .coerceAtLeast(0)
        viewOrder = intent.getStringExtra("view_order")
            ?.takeIf { it == "cyclic" || it == "random" } ?: "cyclic"
        adamMoments = intent.getStringExtra("adam_moments")
            ?.takeIf { it == "bf16" || it == "fp32" } ?: "bf16"
        fullImageSsim = intent.getBooleanExtra("full_image_ssim", false)
        mcmcCloneOffset = intent.getFloatExtra("mcmc_clone_offset", 0.0f)
            .coerceAtLeast(0.0f)
        mcmcGradientSampling = intent.getFloatExtra(
            "mcmc_gradient_sampling", 0.0f).coerceAtLeast(0.0f)
        forceAtomicFreeExact = intent.getBooleanExtra(
            "force_atomic_free_exact", false)
        // 42 reproduces every historical run; other values enable on-device
        // seed-spread studies with the same recipe.
        trainingViewSeed = intent.getIntExtra("training_view_seed", 42)
            .coerceAtLeast(0)
        initSeed = intent.getIntExtra("init_seed", 42).coerceAtLeast(0)
        tileBinCapacity = intent.getIntExtra("tile_bin_capacity", 2048)
        backwardCandidate = intent.getStringExtra("backward_candidate")
            ?.takeIf {
                it == "auto" || it == "exact_pair_gather" ||
                    it == "pixel_subgroup_cas" ||
                    it == "per_splat_cas" ||
                    it == "per_splat_pair_reduce"
            } ?: "auto"
        val probeMode = intent.getStringExtra("probe_mode") ?: ""
        val requestedScene = intent.getStringExtra("scene") ?: "bicycle"
        if (requestedScene in scenes) {
            selectedScene = requestedScene
        } else {
            appendLog("Unknown scene \"$requestedScene\"; using bicycle")
        }
        // Before the first layout pass, or the Spinner's spurious initial
        // onItemSelected(position=0) resets selectedScene to bicycle.
        sceneSpinner.setSelection(scenes.indexOf(selectedScene))
        // Headless runs default to all installed views (pre-view-limit
        // behavior, keeps scripts/benchmark_android.py results comparable);
        // interactive runs default to the fast 8-view demo.
        val requestedViews = intent
            .getIntExtra("views", if (autoStart) 0 else 8)
            .coerceAtLeast(0)
        viewsText.setText(requestedViews.toString())
        updateBackwardButton()
        iterationsText.setText(requestedIterations.toString())
        NativeBridge.setListener(this)
        thread {
            runCatching {
                val data = AssetInstaller.install(this) { message ->
                    runOnUiThread { statusText.text = message }
                }
                Pair(data, NativeBridge.queryDeviceCaps())
            }.onSuccess { (data, caps) ->
                runOnUiThread {
                    dataRoot = data
                    capsText.text = formatCaps(caps)
                    sceneSpinner.setSelection(scenes.indexOf(selectedScene))
                    applyScene()
                    if (probeMode.isNotEmpty()) {
                        if (!NativeBridge.isRunning()) {
                            val probeOut = File(filesDir,
                                "out/probe_$probeMode").absolutePath
                            appendLog("Launching schedule probe " +
                                "$probeMode -> $probeOut")
                            statusText.text = "Schedule probe: $probeMode"
                            NativeBridge.runScheduleProbe(probeMode,
                                probeOut)
                        }
                    } else if (autoStart && !NativeBridge.isRunning()) {
                        startTraining()
                    }
                }
            }.onFailure { error ->
                runOnUiThread {
                    statusText.text = "Asset setup failed: ${error.message}"
                    appendLog(error.stackTraceToString())
                }
            }
        }
    }

    /** Points the UI at [selectedScene] and re-derives per-scene state. */
    private fun applyScene() {
        val data = dataRoot ?: return
        val scene = File(data, selectedScene)
        sceneDir = scene
        resultButtons.forEach { it.isEnabled = false }
        modeButton.isEnabled = false
        val imageCount = File(scene, "images_4").list()?.size ?: 0
        // Mirrors LoadColmapScene's split: index % 8 == 0 is validation.
        val trainCount = imageCount - (imageCount + 7) / 8
        if (!scene.isDirectory || imageCount == 0) {
            startButton.isEnabled = false
            statusText.text = "Scene data missing: ${scene.absolutePath}"
        } else {
            startButton.isEnabled = !NativeBridge.isRunning()
            statusText.text =
                "Ready: $selectedScene, $trainCount train views " +
                    "(Views: 8-$trainCount, 0=all)"
        }
        lastPlyFile = File(filesDir, "out/$selectedScene/splat.ply")
            .takeIf { it.isFile }
        updateOutputPanel()
    }

    /** Shows the output location for the selected scene, if any. */
    private fun updateOutputPanel() {
        val ply = lastPlyFile
        val exportable = ply != null && !NativeBridge.isRunning()
        exportButton.isEnabled = exportable
        shareButton.isEnabled = exportable
        if (ply == null) {
            outputText.visibility = View.GONE
            outputText.text = ""
        } else {
            outputText.visibility = View.VISIBLE
            outputText.text =
                "Output: files/out/$selectedScene/ " +
                    "(splat.ply, bench.json, render/, target/)\n" +
                    "Full path: ${ply.parentFile?.absolutePath}"
        }
    }

    override fun onDestroy() {
        if (NativeBridge.isRunning()) NativeBridge.requestStop()
        NativeBridge.setSurface(null)
        NativeBridge.setListener(null)
        super.onDestroy()
    }

    private fun buildUi() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.rgb(250, 250, 250))
        }
        val title = TextView(this).apply {
            text = "Cross 3DGS — Vulkan optimizer"
            textSize = 19f
            setTypeface(null, Typeface.BOLD)
            setPadding(20, 16, 20, 8)
        }
        capsText = TextView(this).apply {
            text = "Installing dataset and querying Vulkan device..."
            typeface = Typeface.MONOSPACE
            textSize = 11f
            setPadding(20, 4, 20, 8)
        }
        statusText = TextView(this).apply {
            text = "Preparing..."
            setPadding(20, 4, 20, 4)
        }
        sceneSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_item, scenes).apply {
                setDropDownViewResource(
                    android.R.layout.simple_spinner_dropdown_item)
            }
            onItemSelectedListener = object :
                AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: AdapterView<*>?, view: View?,
                    position: Int, id: Long,
                ) {
                    val scene = scenes[position]
                    if (scene == selectedScene) return
                    selectedScene = scene
                    if (dataRoot != null) applyScene()
                }
                override fun onNothingSelected(parent: AdapterView<*>?) {}
            }
        }
        viewsText = EditText(this).apply {
            setText("8")
            hint = "views (0=all)"
            inputType = InputType.TYPE_CLASS_NUMBER
        }
        val sceneRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(20, 0, 12, 0)
            addView(TextView(this@MainActivity).apply { text = "Scene" })
            addView(
                sceneSpinner,
                LinearLayout.LayoutParams(
                    0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
            addView(TextView(this@MainActivity).apply { text = "Views" })
            addView(
                viewsText,
                LinearLayout.LayoutParams(
                    0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        }
        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(12, 0, 12, 4)
        }
        iterationsText = EditText(this).apply {
            setText("1000")
            hint = "iterations"
            inputType = InputType.TYPE_CLASS_NUMBER
        }
        startButton = Button(this).apply {
            text = "Optimize"
            isEnabled = false
            setOnClickListener { startTraining() }
        }
        stopButton = Button(this).apply {
            text = "Stop"
            isEnabled = false
            setOnClickListener {
                NativeBridge.requestStop()
                isEnabled = false
                statusText.text = "Stopping after the current iteration..."
            }
        }
        controls.addView(
            iterationsText,
            LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        controls.addView(
            startButton,
            LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        controls.addView(
            stopButton,
            LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        backwardButton = Button(this).apply {
            setOnClickListener {
                backwardSamples = when (backwardSamples) {
                    0 -> 1
                    1 -> 2
                    2 -> 4
                    4 -> 8
                    else -> 0
                }
                updateBackwardButton()
            }
        }

        progress = ProgressBar(
            this, null, android.R.attr.progressBarStyleHorizontal).apply {
            max = 1000
            progress = 0
            setPadding(20, 0, 20, 2)
        }
        val surface = SurfaceView(this)
        surface.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                NativeBridge.setSurface(holder.surface)
            }
            override fun surfaceChanged(
                holder: SurfaceHolder, format: Int, width: Int, height: Int,
            ) {
                NativeBridge.setSurface(holder.surface)
            }
            override fun surfaceDestroyed(holder: SurfaceHolder) {
                NativeBridge.setSurface(null)
            }
        })

        resultBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        for (view in 0 until 8) {
            val button = Button(this).apply {
                text = "${view + 1}"
                isEnabled = false
                textSize = 11f
                setOnClickListener {
                    selectedView = view
                    selectResult()
                }
            }
            resultButtons += button
            resultBar.addView(button)
        }
        modeButton = Button(this).apply {
            text = "Showing: render"
            isEnabled = false
            setOnClickListener {
                showTarget = !showTarget
                selectResult()
            }
        }
        val results = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(
                HorizontalScrollView(this@MainActivity).apply {
                    addView(resultBar)
                },
                LinearLayout.LayoutParams(
                    0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
            addView(modeButton)
        }

        outputText = TextView(this).apply {
            typeface = Typeface.MONOSPACE
            textSize = 10f
            setTextIsSelectable(true)
            setPadding(20, 2, 20, 2)
            visibility = View.GONE
        }
        exportButton = Button(this).apply {
            text = "Export PLY"
            isEnabled = false
            textSize = 11f
            setOnClickListener { exportOutputs(thenShare = false) }
        }
        shareButton = Button(this).apply {
            text = "Share PLY"
            isEnabled = false
            textSize = 11f
            setOnClickListener { exportOutputs(thenShare = true) }
        }
        howToViewButton = Button(this).apply {
            text = "How to view"
            textSize = 11f
            setOnClickListener { showHowToView() }
        }
        val exportRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(
                exportButton,
                LinearLayout.LayoutParams(
                    0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
            addView(
                shareButton,
                LinearLayout.LayoutParams(
                    0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
            addView(
                howToViewButton,
                LinearLayout.LayoutParams(
                    0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        }

        logText = TextView(this).apply {
            typeface = Typeface.MONOSPACE
            textSize = 10f
            setTextColor(Color.rgb(45, 45, 45))
            setPadding(12, 4, 12, 4)
        }
        logScroll = ScrollView(this).apply { addView(logText) }

        root.addView(title)
        root.addView(capsText)
        root.addView(statusText)
        root.addView(sceneRow)
        root.addView(controls)
        root.addView(backwardButton)
        root.addView(progress)
        root.addView(
            surface,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 5f))
        root.addView(results)
        root.addView(outputText)
        root.addView(exportRow)
        root.addView(
            logScroll,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 2f))
        setContentView(root)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        root.setOnApplyWindowInsetsListener { view, insets ->
            @Suppress("DEPRECATION")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                val bars = insets.getInsets(WindowInsets.Type.systemBars())
                view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
            } else {
                view.setPadding(
                    insets.systemWindowInsetLeft,
                    insets.systemWindowInsetTop,
                    insets.systemWindowInsetRight,
                    insets.systemWindowInsetBottom)
            }
            insets
        }
    }

    private fun startTraining() {
        val scene = sceneDir ?: return
        val iterations = iterationsText.text.toString().toIntOrNull()
        if (iterations == null || iterations <= 0) {
            Toast.makeText(
                this, "Iterations must be positive", Toast.LENGTH_SHORT).show()
            return
        }
        val viewLimit = (viewsText.text.toString().toIntOrNull() ?: 0)
            .coerceAtLeast(0)
        val output = File(filesDir, "out/$selectedScene")
        logLines.clear()
        logText.text = ""
        progress.max = iterations
        progress.progress = 0
        resultButtons.forEach { it.isEnabled = false }
        modeButton.isEnabled = false
        if (!NativeBridge.setTrainingViewSeed(trainingViewSeed)) {
            Toast.makeText(
                this, "Failed to set training view seed", Toast.LENGTH_SHORT)
                .show()
            return
        }
        if (!NativeBridge.setInitSeed(initSeed) ||
            !NativeBridge.setTileBinCapacity(tileBinCapacity)) {
            Toast.makeText(
                this, "Failed to set seed or pair capacity", Toast.LENGTH_SHORT)
                .show()
            return
        }
        if (!NativeBridge.startTraining(
                scene.absolutePath, output.absolutePath,
                "images_4", "sparse/0", iterations,
                viewLimit, progressEvery,
                backwardSamples, ssimWeight, initCheckpoint, shDegree,
                densityStrategy, gaussianCapacity, viewOrder, adamMoments,
                fullImageSsim, mcmcCloneOffset, mcmcGradientSampling,
                forceAtomicFreeExact,
                backwardCandidate)) {
            Toast.makeText(
                this, "Training is already running", Toast.LENGTH_SHORT).show()
            return
        }
        startButton.isEnabled = false
        stopButton.isEnabled = true
        iterationsText.isEnabled = false
        viewsText.isEnabled = false
        backwardButton.isEnabled = false
        sceneSpinner.isEnabled = false
        exportButton.isEnabled = false
        shareButton.isEnabled = false
        statusText.text = "Building Vulkan pipelines..."
    }

    /**
     * Stable per-run export prefix: scene, training end time (splat.ply
     * mtime, so re-exports of the same run overwrite), view count, and
     * iteration count from bench.json, e.g.
     * "garden_YYYYMMDD_HHMMSS_v169_it1000". Runs with different parameters
     * on the same scene get distinct export names.
     */
    private fun exportPrefix(ply: File, bench: File): String {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
            .format(Date(ply.lastModified()))
        val benchText = if (bench.isFile) bench.readText() else ""
        fun field(name: String): String? =
            Regex("\"$name\"\\s*:\\s*(\\d+)")
                .find(benchText)?.groupValues?.get(1)
        val views = field("sampled_train_image_count")?.let { "_v$it" } ?: ""
        val iterations = field("completed_iterations")?.let { "_it$it" } ?: ""
        return "${selectedScene}_$stamp$views$iterations"
    }

    private fun exportOutputs(thenShare: Boolean) {
        val ply = lastPlyFile ?: return
        exportButton.isEnabled = false
        shareButton.isEnabled = false
        thread {
            val result = runCatching {
                val bench = File(ply.parentFile, "bench.json")
                val prefix = exportPrefix(ply, bench)
                val uri = PlyExporter.exportToDownloads(
                    this, ply, "${prefix}_splat.ply")
                if (bench.isFile) {
                    PlyExporter.exportToDownloads(
                        this, bench, "${prefix}_bench.json")
                }
                Pair(uri, prefix)
            }
            runOnUiThread {
                updateOutputPanel()
                result.onSuccess { (uri, prefix) ->
                    appendLog(
                        "Exported ${PlyExporter.RELATIVE_PATH}" +
                            "${prefix}_splat.ply")
                    Toast.makeText(
                        this,
                        "Saved to ${PlyExporter.RELATIVE_PATH}" +
                            "${prefix}_splat.ply",
                        Toast.LENGTH_LONG).show()
                    if (thenShare) sharePly(uri, prefix)
                }.onFailure { error ->
                    appendLog("Export failed: ${error.message}")
                    Toast.makeText(
                        this, "Export failed: ${error.message}",
                        Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun sharePly(uri: Uri, prefix: String) {
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "application/octet-stream"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_TITLE, "${prefix}_splat.ply")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(send, "Share ${prefix}_splat.ply"))
    }

    private fun showHowToView() {
        val scene = selectedScene
        val text = TextView(this).apply {
            typeface = Typeface.MONOSPACE
            textSize = 12f
            setTextIsSelectable(true)
            setPadding(40, 24, 40, 24)
            this.text =
                """
                The trained model is a standard 3DGS PLY:
                files/out/$scene/splat.ply
                (app-private; use Export/Share to take it out)

                View on a PC (recommended)
                1. Pull the file (bash, not PowerShell):
                   adb exec-out run-as org.unclearness.cross3dgs \
                     cat files/out/$scene/splat.ply > splat.ply
                2. Open https://superspl.at/editor in a browser
                   and drag splat.ply in.

                View on this device
                1. Tap "Export PLY" — the file is saved to
                   Download/cross_3dgs/
                   ${scene}_<date>_it<iterations>_splat.ply.
                2. Open it with a 3DGS viewer app (e.g. "A3DGS3D"
                   on the Play Store), or load it in a web viewer
                   such as https://splat.host in Chrome, or upload
                   it to Drive and open it on a PC.

                "Share PLY" sends the exported file directly
                to another app (Drive, Gmail, a viewer, ...).
                """.trimIndent()
        }
        AlertDialog.Builder(this)
            .setTitle("How to view the result")
            .setView(ScrollView(this).apply { addView(text) })
            .setPositiveButton("Close", null)
            .show()
    }

    private fun updateBackwardButton() {
        backwardButton.text = if (backwardSamples == 0) {
            "Backward: exact"
        } else {
            "Backward: Sobol gather K=$backwardSamples"
        }
    }

    private fun selectResult() {
        modeButton.text =
            if (showTarget) "Showing: target" else "Showing: render"
        resultButtons.forEachIndexed { index, button ->
            button.setBackgroundColor(
                if (index == selectedView) Color.rgb(30, 136, 229)
                else Color.rgb(224, 224, 224))
            button.setTextColor(
                if (index == selectedView) Color.WHITE
                else Color.rgb(48, 48, 48))
        }
        if (!NativeBridge.selectResult(selectedView, showTarget)) {
            Toast.makeText(
                this, "Result is not ready", Toast.LENGTH_SHORT).show()
        }
    }

    private fun formatCaps(json: String): String = runCatching {
        val value = JSONObject(json)
        if (value.has("error")) {
            return "Vulkan error: ${value.getString("error")}"
        }
        "${value.optString("deviceName")} | " +
            "Vulkan ${value.optString("apiVersion")} | " +
            "subgroup ${value.optInt("subgroupSize")} | " +
            "float atomic ${value.optBoolean("float32AtomicAdd")} | " +
            "workgroup ${value.optInt("maxComputeWorkGroupInvocations")}"
    }.getOrElse { "Caps parse failed: $json" }

    private fun appendLog(line: String) {
        logLines.addLast(line)
        while (logLines.size > 160) logLines.removeFirst()
        logText.text = logLines.joinToString("\n")
        logScroll.post { logScroll.fullScroll(View.FOCUS_DOWN) }
    }

    override fun onLog(line: String) {
        runOnUiThread { appendLog(line) }
    }

    override fun onProgress(
        iteration: Int,
        total: Int,
        loss: Float,
        elapsedSeconds: Double,
    ) {
        runOnUiThread {
            progress.max = total
            progress.progress = iteration
            statusText.text =
                "$iteration / $total | L1 %.6f | %.2f s".format(
                    loss, elapsedSeconds)
        }
    }

    override fun onFinished(exitCode: Int, message: String, plyPath: String) {
        runOnUiThread {
            stopButton.isEnabled = false
            startButton.isEnabled = true
            iterationsText.isEnabled = true
            viewsText.isEnabled = true
            backwardButton.isEnabled = true
            sceneSpinner.isEnabled = true
            statusText.text = message
            appendLog(message)
            if (plyPath.isNotEmpty()) {
                lastPlyFile = File(plyPath).takeIf { it.isFile }
                lastPlyFile?.parentFile?.let {
                    appendLog("Output dir: ${it.absolutePath}")
                    appendLog("  splat.ply / bench.json / render/ / target/")
                }
            }
            updateOutputPanel()
            if (exitCode == 0) {
                resultButtons.forEach { it.isEnabled = true }
                modeButton.isEnabled = true
                selectedView = 0
                showTarget = false
                selectResult()
            }
            Toast.makeText(
                this,
                if (exitCode == 0) "Optimization finished" else message,
                Toast.LENGTH_LONG).show()
        }
    }
}
