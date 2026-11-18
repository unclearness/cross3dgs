package org.unclearness.cross3dgs

/**
 * Lazy bridge to libcross3dgs_sfm_jni.so: the shared CPU COLMAP SfM
 * (libs/cross3dgs_sfm, the same code the iOS app links) behind a thin JNI
 * wrapper, built by scripts/build_android_sfm.ps1.
 */
object SfmNativeBridge {
    private val loadResult: Result<Unit> by lazy {
        runCatching { System.loadLibrary("cross3dgs_sfm_jni") }
    }

    /** Receives every COLMAP log line, on COLMAP worker threads. */
    @Volatile
    var logListener: ((String) -> Unit)? = null

    fun isAvailable(): Boolean = loadResult.isSuccess

    fun loadError(): String? = loadResult.exceptionOrNull()?.message

    fun runIncremental(
        imagePath: String,
        workspacePath: String,
        priorsPath: String?,
    ): Int {
        loadResult.getOrThrow()
        return nativeRun(imagePath, workspacePath, priorsPath, false, true)
    }

    fun cancel() {
        if (isAvailable()) {
            nativeCancel()
        }
    }

    /** A live preview frame: RGBA8 pixels, size, stage (1..4) and caption. */
    class PreviewFrame(
        val rgba: ByteArray,
        val width: Int,
        val height: Int,
        val stage: Int,
        val caption: String,
    )

    /** Receives live preview frames on the library's preview thread. */
    @Volatile
    var previewListener: ((PreviewFrame) -> Unit)? = null

    /**
     * Enables the library-rendered live preview for later runs at
     * [width] x [height]; 0 disables it.
     */
    fun setPreviewSize(width: Int, height: Int) {
        if (isAvailable()) {
            nativeSetPreview(width, height)
        }
    }

    /** Called from native code (JNI_OnLoad caches this method). */
    @JvmStatic
    fun onLog(line: String) {
        logListener?.invoke(line)
    }

    /** Called from native code on the preview thread. */
    @JvmStatic
    fun onPreview(rgba: ByteArray, width: Int, height: Int, stage: Int, caption: String) {
        previewListener?.invoke(PreviewFrame(rgba, width, height, stage, caption))
    }

    private external fun nativeRun(
        imagePath: String,
        workspacePath: String,
        priorsPath: String?,
        useGlobal: Boolean,
        singleCamera: Boolean,
    ): Int

    private external fun nativeCancel()

    private external fun nativeSetPreview(width: Int, height: Int)

    private external fun nativeSetThreads(extraction: Int, matching: Int, mapping: Int)

    private external fun nativeSetCpuMask(mask: Long)

    /**
     * Restricts later runs to the CPUs in [cpus] ("4-7", "4,5,6,7"; "all" or
     * unparsable = no restriction). Returns the mask applied (0 = none).
     */
    fun setCpus(cpus: String): Long {
        val mask = parseCpuMask(cpus)
        if (isAvailable()) nativeSetCpuMask(mask)
        return mask
    }

    fun parseCpuMask(cpus: String): Long {
        var mask = 0L
        for (part in cpus.split(',')) {
            val range = part.trim().split('-').map { it.trim().toIntOrNull() ?: return 0L }
            val (first, last) = when (range.size) {
                1 -> range[0] to range[0]
                2 -> range[0] to range[1]
                else -> return 0L
            }
            if (first !in 0..63 || last !in first..63) return 0L
            for (cpu in first..last) mask = mask or (1L shl cpu)
        }
        return mask
    }

    /**
     * Worker threads for extraction, matching and mapping in later runs; 0
     * keeps the library default (one per core).
     */
    fun setThreads(extraction: Int, matching: Int, mapping: Int) {
        if (isAvailable()) {
            nativeSetThreads(extraction, matching, mapping)
        }
    }
}
