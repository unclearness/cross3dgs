package org.unclearness.cross3dgs

import java.io.BufferedInputStream
import java.io.EOFException
import java.io.File
import java.io.InputStream

/** Minimal little-endian COLMAP images.bin reader used for scene handoff. */
object ColmapBinary {
    fun registeredImageNames(imagesBin: File): List<String> {
        BufferedInputStream(imagesBin.inputStream()).use { input ->
            val count = input.readU64()
            check(count <= 1_000_000uL) { "implausible COLMAP image count $count" }
            return buildList(count.toInt()) {
                repeat(count.toInt()) {
                    input.skipExact(4 + 7 * 8 + 4)
                    add(input.readCString())
                    val points = input.readU64()
                    check(points <= 100_000_000uL) {
                        "implausible COLMAP observation count $points"
                    }
                    input.skipExact(points.toLong() * 24)
                }
            }
        }
    }

    /** Mean of COLMAP's stored per-point reprojection errors. */
    fun meanPointReprojectionError(pointsBin: File): Double? {
        BufferedInputStream(pointsBin.inputStream()).use { input ->
            val count = input.readU64()
            if (count == 0uL) return null
            check(count <= 100_000_000uL) {
                "implausible COLMAP point count $count"
            }
            var errorSum = 0.0
            repeat(count.toInt()) {
                input.skipExact(8 + 3 * 8 + 3)
                errorSum += Double.fromBits(input.readU64().toLong())
                val trackLength = input.readU64()
                check(trackLength <= 100_000_000uL) {
                    "implausible COLMAP track length $trackLength"
                }
                input.skipExact(trackLength.toLong() * 8L)
            }
            return errorSum / count.toDouble()
        }
    }

    private fun InputStream.readU64(): ULong {
        var value = 0uL
        repeat(8) { shift ->
            val byte = read()
            if (byte < 0) throw EOFException("truncated COLMAP binary")
            value = value or (byte.toULong() shl (8 * shift))
        }
        return value
    }

    private fun InputStream.readCString(): String {
        val bytes = ArrayList<Byte>()
        while (true) {
            val value = read()
            if (value < 0) throw EOFException("truncated COLMAP image name")
            if (value == 0) break
            bytes += value.toByte()
        }
        return bytes.toByteArray().toString(Charsets.UTF_8)
    }

    private fun InputStream.skipExact(bytes: Long) {
        var remaining = bytes
        while (remaining > 0) {
            val skipped = skip(remaining)
            if (skipped > 0) {
                remaining -= skipped
            } else if (read() >= 0) {
                remaining--
            } else {
                throw EOFException("truncated COLMAP binary")
            }
        }
    }
}
