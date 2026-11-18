package org.unclearness.cross3dgs

import java.io.File

/** Fail-closed orchestration for the resumable capture-to-view application. */
class ProjectPipeline(
    private val workspace: ProjectWorkspace,
    private val onStatus: (String) -> Unit,
) {
    @Volatile
    private var canceled = false

    fun cancel() {
        canceled = true
        SfmNativeBridge.cancel()
        NativeBridge.requestStop()
    }

    fun runSfm(manifest: ProjectManifest): ProjectManifest {
        check(manifest.stage == PipelineStage.SFM) {
            "SfM requires stage=sfm, got ${manifest.stage}"
        }
        check(workspace.sourceImages.listFiles()?.isNotEmpty() == true) {
            "no extracted source images"
        }
        check(SfmNativeBridge.isAvailable()) {
            "Android COLMAP library unavailable: ${SfmNativeBridge.loadError()}"
        }
        workspace.requireEmpty(workspace.sfmWorkspace, "SfM")
        workspace.sfmSparse.mkdirs()
        onStatus("Running incremental COLMAP with ARCore priors")
        val priors = workspace.framePriorsFile.takeIf(File::isFile)?.absolutePath
        val exit = SfmNativeBridge.runIncremental(
            workspace.sourceImages.absolutePath,
            workspace.sfmWorkspace.absolutePath,
            priors,
        )
        check(!canceled) { "SfM canceled" }
        check(exit == 0) { "COLMAP failed with exit code $exit" }
        val selected = selectLargestSparseModel(workspace.sfmSparse)
        val registered = prepareTrainingScene(selected)
        val meanError = ColmapBinary.meanPointReprojectionError(
            File(selected, "points3D.bin"))
        return manifest.copy(
            stage = PipelineStage.OPTIMIZE,
            selectedSfmModel = selected.absolutePath,
            registeredImages = registered,
            meanReprojectionError = meanError,
            lastError = null,
        ).also(workspace::save)
    }

    private fun selectLargestSparseModel(root: File): File {
        val models = root.listFiles { file -> file.isDirectory }
            ?.filter { File(it, "images.bin").isFile }
            .orEmpty()
        check(models.isNotEmpty()) { "COLMAP produced no sparse model" }
        return models.maxBy { File(it, "images.bin").length() }
    }

    private fun prepareTrainingScene(selected: File): Int {
        val registeredNames = ColmapBinary.registeredImageNames(
            File(selected, "images.bin"))
        check(registeredNames.size >= 8) {
            "SfM registered only ${registeredNames.size} images"
        }
        workspace.requireEmpty(workspace.trainingRoot, "training scene")
        workspace.trainingSparse.mkdirs()
        listOf("cameras.bin", "images.bin", "points3D.bin").forEach { name ->
            File(selected, name).copyTo(
                File(workspace.trainingSparse, name), overwrite = false)
        }
        check(workspace.trainingImages.mkdirs() || workspace.trainingImages.isDirectory) {
            "failed to create training image directory"
        }
        registeredNames.forEach { name ->
            val source = File(workspace.sourceImages, name)
            check(source.isFile) { "registered image missing: $name" }
            File(workspace.trainingImages, name).parentFile?.mkdirs()
            source.copyTo(
                    File(workspace.trainingImages, name),
                    overwrite = false,
                )
        }
        return registeredNames.size
    }
}
