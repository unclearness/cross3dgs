package org.unclearness.cross3dgs

import android.view.Surface

object NativeBridge {
    init {
        System.loadLibrary("cross3dgs_android")
    }

    interface Listener {
        fun onLog(line: String)
        fun onProgress(
            iteration: Int,
            total: Int,
            loss: Float,
            elapsedSeconds: Double,
        )
        fun onFinished(exitCode: Int, message: String, plyPath: String)
    }

    external fun queryDeviceCaps(): String
    external fun startTraining(
        sceneDir: String,
        outputDir: String,
        imageDir: String,
        sparseDir: String,
        iterations: Int,
        viewLimit: Int,
        progressEvery: Int,
        stochasticSamples: Int,
        ssimWeight: Float,
        initCheckpoint: String,
        shDegree: Int,
        densityStrategy: String,
        gaussianCapacity: Int,
        viewOrder: String,
        adamMoments: String,
        fullImageSsim: Boolean,
        mcmcCloneOffset: Float,
        mcmcGradientSampling: Float,
        forceAtomicFreeExact: Boolean,
        backwardCandidate: String,
    ): Boolean
    /**
     * Sets the per-tile tile-Gaussian pair budget for the next run.
     *
     * pair_capacity is this value times the tile count, so a capture trained
     * at full extraction resolution can exhaust the 512 default and silently
     * drop pairs during rasterization. Returns false while a run is active.
     */
    external fun setTileBinCapacity(value: Int): Boolean

    /**
     * Shows the photo, the render and the loss map of the current training view
     * instead of the render alone. Returns false while a run is active.
     */
    external fun setTrainingPreviewPanels(enabled: Boolean): Boolean

    /**
     * Image width and height, then x, y, width, height of each panel in the
     * order of [trainingPreviewLabels]. Empty until a run has its image size.
     */
    external fun trainingPreviewLayout(): IntArray

    /** Captions of the photo, render and loss panels. */
    external fun trainingPreviewLabels(): Array<String>

    external fun requestStop()
    external fun isRunning(): Boolean
    external fun setSurface(surface: Surface?)
    external fun selectResult(viewIndex: Int, target: Boolean): Boolean
    external fun prepareViewer(
        sceneDir: String,
        imageDir: String,
        sparseDir: String,
        checkpointPath: String,
        referenceImageName: String,
        // frame_priors.txt, or "" when the run has none; with gravity priors
        // the free view is gravity-aligned (cross3dgs/orbit_view.h).
        framePriorsPath: String,
        // Where to write the gravity-aligned PLY, or "" to write nothing. A
        // run without usable gravity priors writes nothing either.
        gravityUpPlyPath: String,
        shDegree: Int,
    ): Boolean
    /**
     * Shows an RGBA8 frame (e.g. the SfM live preview) on the surface,
     * letterboxed like the training preview.
     */
    external fun showRgba(rgba: ByteArray, width: Int, height: Int): Boolean
    /** {min, max} pitch of the prepared viewer in radians; contains 0. */
    external fun orbitPitchRange(): FloatArray
    external fun renderOrbit(
        yawRadians: Float,
        pitchRadians: Float,
        zoom: Float,
        panX: Float,
        panY: Float,
        width: Int,
        height: Int,
    ): Boolean
    external fun setListener(listener: Listener?)
}
