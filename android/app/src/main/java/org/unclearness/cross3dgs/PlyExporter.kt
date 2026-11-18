package org.unclearness.cross3dgs

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.provider.MediaStore
import java.io.File
import java.io.IOException

/**
 * Copies training outputs from the app-private files dir into the shared
 * Downloads collection so external viewers and other apps can reach them.
 * MediaStore inserts into Downloads need no storage permission on API 29+.
 */
object PlyExporter {
    const val RELATIVE_PATH = "Download/cross_3dgs/"

    /**
     * Inserts [source] as Download/cross_3dgs/[displayName], overwriting the
     * app's previous export of the same name instead of creating
     * "name (1).ply" duplicates. Returns the MediaStore item Uri.
     */
    fun exportToDownloads(
        context: Context,
        source: File,
        displayName: String,
    ): Uri {
        if (!source.isFile) {
            throw IOException("missing source file: ${source.absolutePath}")
        }
        val resolver = context.contentResolver
        val collection = MediaStore.Downloads.EXTERNAL_CONTENT_URI
        val existing = resolver.query(
            collection,
            arrayOf(MediaStore.MediaColumns._ID),
            "${MediaStore.MediaColumns.RELATIVE_PATH} = ? AND " +
                "${MediaStore.MediaColumns.DISPLAY_NAME} = ?",
            arrayOf(RELATIVE_PATH, displayName),
            null,
        )?.use { cursor ->
            if (cursor.moveToFirst()) {
                Uri.withAppendedPath(collection, cursor.getLong(0).toString())
            } else {
                null
            }
        }
        val uri = existing ?: resolver.insert(
            collection,
            ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, displayName)
                put(MediaStore.MediaColumns.RELATIVE_PATH, RELATIVE_PATH)
                put(MediaStore.MediaColumns.MIME_TYPE,
                    "application/octet-stream")
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            },
        ) ?: throw IOException("MediaStore insert failed for $displayName")
        try {
            resolver.openOutputStream(uri, "wt")?.use { output ->
                source.inputStream().use { input -> input.copyTo(output) }
            } ?: throw IOException("cannot open output stream for $uri")
        } catch (error: Exception) {
            if (existing == null) resolver.delete(uri, null, null)
            throw error
        }
        resolver.update(
            uri,
            ContentValues().apply {
                put(MediaStore.MediaColumns.IS_PENDING, 0)
            },
            null, null)
        return uri
    }
}
