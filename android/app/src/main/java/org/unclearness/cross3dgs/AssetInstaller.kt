package org.unclearness.cross3dgs

import android.content.Context
import android.content.res.AssetManager
import java.io.File

object AssetInstaller {
    fun install(
        context: Context,
        onProgress: (String) -> Unit = {},
    ): File {
        val dataDir = File(context.filesDir, "data")
        val marker = File(
            context.filesDir,
            ".cross3dgs_assets_v${BuildConfig.VERSION_CODE}")
        if (marker.exists()) return dataDir
        dataDir.deleteRecursively()
        val scenes = context.assets.list("data").orEmpty().sorted()
        scenes.forEachIndexed { index, scene ->
            onProgress(
                "Installing $scene (${index + 1}/${scenes.size})...")
            copyAssetPath(
                context.assets, "data/$scene", File(dataDir, scene))
        }
        context.filesDir.listFiles {
            file -> file.name.startsWith(".cross3dgs_assets_v")
        }?.forEach { it.delete() }
        marker.createNewFile()
        return dataDir
    }

    private fun copyAssetPath(
        assets: AssetManager,
        path: String,
        destination: File,
    ) {
        val children = assets.list(path) ?: return
        if (children.isEmpty()) {
            destination.parentFile?.mkdirs()
            assets.open(path).use { input ->
                destination.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
        } else {
            children.forEach { child ->
                copyAssetPath(
                    assets, "$path/$child", File(destination, child))
            }
        }
    }
}
