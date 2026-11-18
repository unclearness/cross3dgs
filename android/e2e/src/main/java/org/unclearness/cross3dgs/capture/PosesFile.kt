package org.unclearness.cross3dgs.capture

import java.io.File
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Parser for `*.arposes.txt` capture files.
 *
 * Body lines are TUM trajectory format: `timestamp_ns tx ty tz qx qy qz qw`,
 * world_from_camera in meters, ARCore camera convention (+X right, +Y up,
 * -Z forward). Header comments carry `video_start_ns`, `intrinsics`
 * (PINHOLE fx fy cx cy width height) and `gravity_world`.
 */
class PosesFile private constructor(
    val videoStartNs: Long,
    val intrinsics: DoubleArray?, // fx fy cx cy
    val intrinsicsSize: IntArray?, // width height
    private val gravityWorld: DoubleArray,
    private val times: LongArray,
    private val positions: Array<DoubleArray>,
    private val quats: Array<DoubleArray>, // x y z w
) {

    companion object {
        private const val MAX_GAP_NS = 150_000_000L // 150 ms
        private const val STABLE_BRACKET_NS = 66_000_000L // ~2 VIO frames
        private const val SIGMA_STABLE = 0.08
        private const val SIGMA_UNSTABLE = 0.20

        fun parse(file: File): PosesFile? {
            if (!file.exists()) return null
            var videoStartNs = 0L
            var intr: DoubleArray? = null
            var intrSize: IntArray? = null
            var gravity = doubleArrayOf(0.0, -1.0, 0.0)
            val times = ArrayList<Long>()
            val positions = ArrayList<DoubleArray>()
            val quats = ArrayList<DoubleArray>()
            file.forEachLine { raw ->
                val line = raw.trim()
                if (line.isEmpty()) return@forEachLine
                if (line.startsWith("#")) {
                    val tok = line.removePrefix("#").trim().split(Regex("\\s+"))
                    when (tok.getOrNull(0)) {
                        "video_start_ns" -> videoStartNs = tok[1].toLong()
                        "intrinsics" -> {
                            intr = doubleArrayOf(tok[1].toDouble(), tok[2].toDouble(),
                                tok[3].toDouble(), tok[4].toDouble())
                            intrSize = intArrayOf(tok[5].toInt(), tok[6].toInt())
                        }
                        "gravity_world" -> gravity = doubleArrayOf(
                            tok[1].toDouble(), tok[2].toDouble(), tok[3].toDouble())
                    }
                    return@forEachLine
                }
                val tok = line.split(Regex("\\s+"))
                if (tok.size < 8) return@forEachLine
                times.add(tok[0].toLong())
                positions.add(doubleArrayOf(tok[1].toDouble(), tok[2].toDouble(), tok[3].toDouble()))
                quats.add(doubleArrayOf(tok[4].toDouble(), tok[5].toDouble(),
                    tok[6].toDouble(), tok[7].toDouble()))
            }
            if (times.isEmpty()) return null
            return PosesFile(videoStartNs, intr, intrSize, gravity,
                times.toLongArray(), positions.toTypedArray(), quats.toTypedArray())
        }
    }

    data class Prior(
        val position: DoubleArray,
        val gravityCam: DoubleArray,
        /** Position std-dev [m]; inflated when the VIO stream had a gap near
         *  this timestamp (tracking instability). */
        val sigma: Double,
    )

    /**
     * Interpolated prior at absolute capture time [tNs] (ARCore clock), or
     * null when no pose sample is within range. Gravity is returned in the
     * COLMAP camera frame (+X right, +Y down, +Z forward).
     */
    fun priorAt(tNs: Long): Prior? {
        val idx = times.binarySearch(tNs).let { if (it >= 0) it else -it - 2 }
        if (idx < 0 || idx >= times.size) {
            // outside the sampled range; accept only if very close to an end
            val j = if (idx < 0) 0 else times.size - 1
            if (abs(tNs - times[j]) > MAX_GAP_NS) return null
            return makePrior(positions[j], quats[j], SIGMA_UNSTABLE)
        }
        if (idx == times.size - 1) {
            if (abs(tNs - times[idx]) > MAX_GAP_NS) return null
            return makePrior(positions[idx], quats[idx], SIGMA_UNSTABLE)
        }
        val t0 = times[idx]
        val t1 = times[idx + 1]
        if (tNs - t0 > MAX_GAP_NS && t1 - tNs > MAX_GAP_NS) return null
        // VIO normally samples every ~33 ms; a larger bracket indicates the
        // tracker dropped frames around this time -> weaker prior.
        val sigma = if (t1 - t0 > STABLE_BRACKET_NS) SIGMA_UNSTABLE else SIGMA_STABLE
        val a = (tNs - t0).toDouble() / (t1 - t0).toDouble()
        val pos = DoubleArray(3) { positions[idx][it] * (1 - a) + positions[idx + 1][it] * a }
        val q = slerp(quats[idx], quats[idx + 1], a)
        return makePrior(pos, q, sigma)
    }

    private fun makePrior(pos: DoubleArray, qWorldFromCam: DoubleArray, sigma: Double): Prior {
        // gravity in ARCore camera frame: g_cam = R_wc^T * g_world
        val gCamAr = rotateByInverse(qWorldFromCam, gravityWorld)
        // ARCore camera (+X right, +Y up, -Z fwd) -> COLMAP camera
        // (+X right, +Y down, +Z fwd): negate Y and Z.
        val gCam = doubleArrayOf(gCamAr[0], -gCamAr[1], -gCamAr[2])
        return Prior(pos.copyOf(), gCam, sigma)
    }

    // --- minimal quaternion helpers (x y z w) ---

    private fun rotateByInverse(q: DoubleArray, v: DoubleArray): DoubleArray {
        // R(q)^T v = R(conj(q)) v
        val cq = doubleArrayOf(-q[0], -q[1], -q[2], q[3])
        return rotate(cq, v)
    }

    private fun rotate(q: DoubleArray, v: DoubleArray): DoubleArray {
        val (x, y, z, w) = listOf(q[0], q[1], q[2], q[3])
        // t = 2 q_vec x v; v' = v + w t + q_vec x t
        val tx = 2 * (y * v[2] - z * v[1])
        val ty = 2 * (z * v[0] - x * v[2])
        val tz = 2 * (x * v[1] - y * v[0])
        return doubleArrayOf(
            v[0] + w * tx + (y * tz - z * ty),
            v[1] + w * ty + (z * tx - x * tz),
            v[2] + w * tz + (x * ty - y * tx))
    }

    private fun slerp(qa: DoubleArray, qbIn: DoubleArray, a: Double): DoubleArray {
        var qb = qbIn
        var dot = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3]
        if (dot < 0) { qb = doubleArrayOf(-qb[0], -qb[1], -qb[2], -qb[3]); dot = -dot }
        if (dot > 0.9995) {
            val out = DoubleArray(4) { qa[it] * (1 - a) + qb[it] * a }
            val n = sqrt(out.sumOf { it * it })
            return DoubleArray(4) { out[it] / n }
        }
        val theta = acos(dot.coerceIn(-1.0, 1.0))
        val s = sin(theta)
        val wa = sin((1 - a) * theta) / s
        val wb = sin(a * theta) / s
        return DoubleArray(4) { qa[it] * wa + qb[it] * wb }
    }
}

