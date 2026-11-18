package org.unclearness.cross3dgs

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

enum class PipelineStage {
    CAPTURE,
    EXTRACT,
    SFM,
    OPTIMIZE,
    VIEW,
    COMPLETE,
    FAILED,
}

enum class OptimizationPreset(
    val iterations: Int,
    val gaussianCapacity: Int,
    val fullImageSsim: Boolean,
    val mcmcCloneOffset: Float,
    val mcmcGradientSampling: Float,
) {
    // The recipe flags match the standard-quality preset
    // (configs/standard_quality.json: full_image_ssim with clone offset 0.5
    // and gradient sampling 1.5), so every preset trains with the same
    // recipe. Only the budget differs by preset.
    // Capacities are multiples of 32 so the trainer can use the reordered
    // SH-rest layout, which it gates on capacity % 32 == 0; the
    // demonstration budget is 250,000 rounded up to 7,813 x 32.
    DEMO(
        iterations = 3_000,
        gaussianCapacity = 250_016,
        fullImageSsim = true,
        mcmcCloneOffset = 0.5f,
        mcmcGradientSampling = 1.5f,
    ),
    FULL(
        iterations = 30_000,
        gaussianCapacity = 1_000_000,
        fullImageSsim = true,
        mcmcCloneOffset = 0.5f,
        mcmcGradientSampling = 1.5f,
    ),
}

data class OptimizationConfig(
    val iterations: Int = OptimizationPreset.DEMO.iterations,
    val gaussianCapacity: Int = OptimizationPreset.DEMO.gaussianCapacity,
    val shDegree: Int = 3,
    val densityStrategy: String = "mcmc",
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("iterations", iterations)
        put("gaussian_capacity", gaussianCapacity)
        put("sh_degree", shDegree)
        put("density_strategy", densityStrategy)
    }

    companion object {
        fun fromPreset(preset: OptimizationPreset): OptimizationConfig =
            OptimizationConfig(
                iterations = preset.iterations,
                gaussianCapacity = preset.gaussianCapacity,
            )

        fun fromJson(
            json: JSONObject?,
            fallbackPreset: OptimizationPreset,
        ): OptimizationConfig {
            val fallback = fromPreset(fallbackPreset)
            if (json == null) return fallback
            return OptimizationConfig(
                iterations = json.optInt("iterations", fallback.iterations),
                gaussianCapacity = json.optInt(
                    "gaussian_capacity", fallback.gaussianCapacity),
                shDegree = json.optInt("sh_degree", fallback.shDegree),
                densityStrategy = json.optString(
                    "density_strategy", fallback.densityStrategy),
            )
        }
    }
}

/**
 * Raw per-frame viewer timings from the deterministic post-evidence orbit
 * sweep, plus the derived summary. Times cover one `renderOrbit` host call
 * (execute, color readback, surface show) and therefore carry no present
 * completion guarantee, matching the optimizer's preview timing scope.
 */
data class ViewerFrameSweep(
    val frameTimesUs: List<Long>,
    val requestedWidth: Int,
    val requestedHeight: Int,
    val yawSpanRadians: Float,
) {
    private fun percentileUs(fraction: Double): Long {
        val sorted = frameTimesUs.sorted()
        val position = fraction * (sorted.size - 1)
        return sorted[position.toInt().coerceIn(0, sorted.size - 1)]
    }

    val medianFrameMs: Double
        get() = if (frameTimesUs.isEmpty()) 0.0 else percentileUs(0.5) / 1000.0

    fun toJson(): JSONObject = JSONObject().apply {
        put("frame_times_us", JSONArray(frameTimesUs))
        put("frames", frameTimesUs.size)
        put("requested_width", requestedWidth)
        put("requested_height", requestedHeight)
        put("yaw_span_radians", yawSpanRadians.toDouble())
        if (frameTimesUs.isNotEmpty()) {
            put("median_frame_ms", medianFrameMs)
            put("p90_frame_ms", percentileUs(0.9) / 1000.0)
            put("min_frame_ms", (frameTimesUs.min()) / 1000.0)
            put("max_frame_ms", (frameTimesUs.max()) / 1000.0)
            put("fps_from_median",
                if (medianFrameMs > 0.0) 1_000.0 / medianFrameMs else 0.0)
        }
        put("scope", "render_execute_download_color_and_surface_show_host_call")
        put("semantics", "no_present_completion_guarantee_deterministic_yaw_sweep")
    }

    companion object {
        fun fromJson(json: JSONObject?): ViewerFrameSweep? {
            if (json == null) return null
            val times = json.optJSONArray("frame_times_us") ?: return null
            return ViewerFrameSweep(
                frameTimesUs = (0 until times.length()).map(times::getLong),
                requestedWidth = json.optInt("requested_width", 0),
                requestedHeight = json.optInt("requested_height", 0),
                yawSpanRadians = json.optDouble("yaw_span_radians", 0.0).toFloat(),
            )
        }
    }
}

/** Persistent, resumable state for one capture-to-view project. */
data class ProjectManifest(
    val schemaVersion: Int = 6,
    val projectId: String,
    val captureId: String = "",
    val runId: String = projectId,
    val autoRunToEnd: Boolean = false,
    val stage: PipelineStage = PipelineStage.CAPTURE,
    val preset: OptimizationPreset = OptimizationPreset.DEMO,
    val optimizationConfig: OptimizationConfig = OptimizationConfig(),
    val captureSource: String = "",
    val imageDirectory: String = "training/images",
    val sparseDirectory: String = "training/sparse/0",
    val selectedSfmModel: String = "",
    val frameTargetCount: Int = 60,
    val extractedFrames: Int = 0,
    val registeredImages: Int = 0,
    val meanReprojectionError: Double? = null,
    val positionPriorCount: Int = 0,
    val gravityPriorCount: Int = 0,
    val intrinsicsPriorPresent: Boolean = false,
    val capturePoseSamples: Int = 0,
    val captureSensorSamples: Int = 0,
    val stageTimesMs: Map<String, Long> = emptyMap(),
    val viewerFrameSweep: ViewerFrameSweep? = null,
    val artifactSha256: Map<String, String> = emptyMap(),
    val screenshots: Map<String, String> = emptyMap(),
    val environment: Map<String, String> = emptyMap(),
    val pipelineConfig: Map<String, String> = emptyMap(),
    val lastError: String? = null,
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("schema_version", schemaVersion)
        put("project_id", projectId)
        put("capture_id", captureId)
        put("run_id", runId)
        put("auto_run_to_end", autoRunToEnd)
        put("stage", stage.name.lowercase())
        put("preset", preset.name.lowercase())
        put("optimization_config", optimizationConfig.toJson())
        put("capture_source", captureSource)
        put("image_directory", imageDirectory)
        put("sparse_directory", sparseDirectory)
        put("selected_sfm_model", selectedSfmModel)
        put("frame_target_count", frameTargetCount)
        put("extracted_frames", extractedFrames)
        put("registered_images", registeredImages)
        put("mean_reprojection_error", meanReprojectionError ?: JSONObject.NULL)
        put("position_prior_count", positionPriorCount)
        put("gravity_prior_count", gravityPriorCount)
        put("intrinsics_prior_present", intrinsicsPriorPresent)
        put("capture_pose_samples", capturePoseSamples)
        put("capture_sensor_samples", captureSensorSamples)
        put("stage_times_ms", JSONObject(stageTimesMs))
        put("viewer_frame_sweep", viewerFrameSweep?.toJson() ?: JSONObject.NULL)
        put("artifact_sha256", JSONObject(artifactSha256))
        put("screenshots", JSONObject(screenshots))
        put("environment", JSONObject(environment))
        put("pipeline_config", JSONObject(pipelineConfig))
        put("last_error", lastError ?: JSONObject.NULL)
    }

    companion object {
        fun create(projectId: String = UUID.randomUUID().toString()): ProjectManifest =
            ProjectManifest(projectId = projectId)

        fun fromJson(json: JSONObject): ProjectManifest {
            val preset = OptimizationPreset.valueOf(
                json.optString("preset", "demo").uppercase())
            return ProjectManifest(
                schemaVersion = json.optInt("schema_version", 1),
                projectId = json.getString("project_id"),
                captureId = json.optString("capture_id", ""),
                runId = json.optString(
                    "run_id", json.getString("project_id")),
                autoRunToEnd = json.optBoolean("auto_run_to_end", false),
                stage = PipelineStage.valueOf(
                    json.optString("stage", "capture").uppercase()),
                preset = preset,
                optimizationConfig = OptimizationConfig.fromJson(
                    json.optJSONObject("optimization_config"), preset),
                captureSource = json.optString("capture_source", ""),
                imageDirectory = json.optString(
                    "image_directory", "training/images"),
                sparseDirectory = json.optString(
                    "sparse_directory", "training/sparse/0"),
                selectedSfmModel = json.optString("selected_sfm_model", ""),
                frameTargetCount = json.optInt("frame_target_count", 60),
                extractedFrames = json.optInt("extracted_frames", 0),
                registeredImages = json.optInt("registered_images", 0),
                meanReprojectionError = if (
                    json.isNull("mean_reprojection_error")
                ) {
                    null
                } else {
                    json.getDouble("mean_reprojection_error")
                },
                positionPriorCount = json.optInt("position_prior_count", 0),
                gravityPriorCount = json.optInt("gravity_prior_count", 0),
                intrinsicsPriorPresent = json.optBoolean(
                    "intrinsics_prior_present", false),
                capturePoseSamples = json.optInt("capture_pose_samples", 0),
                captureSensorSamples = json.optInt("capture_sensor_samples", 0),
                stageTimesMs = json.optJSONObject("stage_times_ms")?.let { times ->
                    times.keys().asSequence().associateWith { times.getLong(it) }
                }.orEmpty(),
                viewerFrameSweep = ViewerFrameSweep.fromJson(
                    json.optJSONObject("viewer_frame_sweep")),
                artifactSha256 = json.stringMap("artifact_sha256"),
                screenshots = json.stringMap("screenshots"),
                environment = json.stringMap("environment"),
                pipelineConfig = json.stringMap("pipeline_config"),
                lastError = if (json.isNull("last_error")) {
                    null
                } else {
                    json.getString("last_error")
                },
            )
        }
    }
}

class ProjectWorkspace private constructor(val root: File) {
    val captureDir = File(root, "input/capture")
    val videoFile = File(captureDir, "video.mp4")
    val sensorFile = File(captureDir, "sensors.csv")
    /** Raw, per-ARCore-frame trajectory. Never overwrite with derived priors. */
    val rawArcorePosesFile = File(captureDir, "arcore_poses.txt")
    /** Selected-frame priors consumed by COLMAP; derived from rawArcorePosesFile. */
    val framePriorsFile = File(root, "extract/frame_priors.txt")
    val sourceImages = File(root, "extract/images")
    val sfmWorkspace = File(root, "sfm")
    val sfmSparse = File(sfmWorkspace, "sparse")
    val trainingImages = File(root, "training/images")
    val trainingRoot = File(root, "training")
    val trainingSparse = File(root, "training/sparse/0")
    val output = File(root, "output")
    val screenshotsDir = File(root, "screenshots")
    val manifestFile = File(root, "project_manifest.json")
    val pipelineLog = File(root, "pipeline.log")
    val telemetryFile = File(root, "telemetry.csv")

    fun ensureLayout() {
        listOf(
            captureDir, sourceImages, sfmSparse, trainingImages,
            trainingSparse, output, framePriorsFile.parentFile, screenshotsDir,
        ).forEach(File::mkdirs)
    }

    fun save(manifest: ProjectManifest) {
        ensureLayout()
        val pending = File(root, "project_manifest.json.pending")
        pending.writeText(manifest.toJson().toString(2) + "\n")
        try {
            Files.move(
                pending.toPath(), manifestFile.toPath(),
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING,
            )
        } catch (_: AtomicMoveNotSupportedException) {
            Files.move(
                pending.toPath(), manifestFile.toPath(),
                StandardCopyOption.REPLACE_EXISTING,
            )
        }
    }

    fun load(): ProjectManifest =
        ProjectManifest.fromJson(JSONObject(manifestFile.readText()))

    fun requireEmpty(directory: File, label: String) {
        check(!directory.exists() || directory.walkTopDown().none(File::isFile)) {
            "$label output already exists; create a new run instead of overwriting"
        }
    }

    companion object {
        fun open(projectsRoot: File, projectId: String): ProjectWorkspace =
            ProjectWorkspace(File(projectsRoot, projectId)).also {
                it.ensureLayout()
            }

        fun inspect(root: File): ProjectWorkspace = ProjectWorkspace(root).also {
            check(root.isDirectory) { "run directory is missing: $root" }
        }
    }
}

class CaptureRecord(val root: File) {
    val id: String = root.name
    val videoFile = File(root, "video.mp4")
    val rawArcorePosesFile = File(root, "arcore_poses.txt")
    val sensorFile = File(root, "sensors.csv")
    val screenshotFile = File(root, "capture.png")
    val manifestFile = File(root, "capture_manifest.json")

    fun requireComplete() {
        check(videoFile.isFile && videoFile.length() > 0) {
            "capture $id has no video.mp4"
        }
        check(rawArcorePosesFile.isFile && rawArcorePosesFile.length() > 0) {
            "capture $id has no arcore_poses.txt"
        }
        check(sensorFile.isFile && sensorFile.length() > 0) {
            "capture $id has no sensors.csv"
        }
        check(screenshotFile.isFile && screenshotFile.length() > 0) {
            "capture $id has no capture.png"
        }
    }

    fun verifySealed() {
        requireComplete()
        check(manifestFile.isFile) { "capture $id is not sealed" }
        val json = JSONObject(manifestFile.readText())
        check(json.getString("capture_id") == id) {
            "capture manifest id does not match directory: $id"
        }
        val hashes = json.optJSONObject("artifact_sha256")
            ?: error("capture $id has no SHA-256 inventory")
        check(hashes.length() >= 2) { "capture $id has an incomplete inventory" }
        hashes.keys().forEach { name ->
            check(!name.contains('/') && !name.contains('\\')) {
                "capture artifact must be a direct child: $name"
            }
            val file = File(root, name)
            check(file.isFile) { "capture artifact is missing: $name" }
            check(ProjectEvidence.sha256(file).equals(
                hashes.getString(name), ignoreCase = true)) {
                "capture artifact SHA-256 mismatch: $name"
            }
        }
        listOf(videoFile, rawArcorePosesFile, sensorFile, screenshotFile).forEach { required ->
            check(hashes.has(required.name)) {
                "capture inventory is missing ${required.name}"
            }
        }
    }
}

class CaptureStore(private val root: File) {
    init { root.mkdirs() }

    fun create(): CaptureRecord {
        val record = CaptureRecord(File(root, uniqueArtifactId("capture")))
        check(record.root.mkdir()) { "failed to create unique capture directory" }
        return record
    }

    fun listComplete(): List<CaptureRecord> = root.listFiles(File::isDirectory)
        .orEmpty().map(::CaptureRecord).filter {
            runCatching { it.verifySealed() }.isSuccess
        }.sortedByDescending { it.id }

    fun importProjectBundle(bundle: File): CaptureRecord {
        check(bundle.isDirectory) { "capture bundle is not a directory: $bundle" }
        val sourceVideo = firstFile(bundle, "video.mp4", "capture/video.mp4", "input/capture/video.mp4")
            ?: error("capture bundle has no video.mp4")
        val sourcePoses = firstFile(
            bundle, "arcore_poses.txt", "capture/arcore_poses.txt",
            "input/capture/arcore_poses.txt",
        ) ?: error("capture bundle has no arcore_poses.txt")
        val record = create()
        sourceVideo.copyTo(record.videoFile, overwrite = false)
        sourcePoses.copyTo(record.rawArcorePosesFile, overwrite = false)
        firstFile(bundle, "sensors.csv", "capture/sensors.csv", "input/capture/sensors.csv")
            ?.copyTo(record.sensorFile, overwrite = false)
        firstFile(bundle, "capture.png", "capture/capture.png", "input/capture/capture.png")
            ?.copyTo(record.screenshotFile, overwrite = false)
        return record.also(CaptureRecord::requireComplete)
    }

    private fun firstFile(root: File, vararg relative: String): File? =
        relative.asSequence().map { File(root, it) }.firstOrNull(File::isFile)
}

fun uniqueArtifactId(prefix: String): String {
    val time = SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())
    return "${prefix}_${time}_${UUID.randomUUID().toString().substring(0, 8)}"
}

private fun JSONObject.stringMap(name: String): Map<String, String> =
    optJSONObject(name)?.let { values ->
        values.keys().asSequence().associateWith { values.getString(it) }
    }.orEmpty()
