package org.unclearness.cross3dgs

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Debug
import android.os.PowerManager
import android.os.Process
import android.os.SystemClock
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/** Raw, scope-labeled process/battery/thermal samples across the E2E pipeline. */
object ProjectTelemetry {
    private val lock = Any()
    private var executor = Executors.newSingleThreadScheduledExecutor()
    private var writer: BufferedWriter? = null
    private var appContext: Context? = null
    private var outputFile: File? = null
    @Volatile private var stage = "capture"

    fun start(context: Context, output: File, initialStage: String) = synchronized(lock) {
        if (outputFile == output && writer != null) {
            stage = initialStage
            return@synchronized
        }
        stopLocked()
        if (executor.isShutdown) executor = Executors.newSingleThreadScheduledExecutor()
        output.parentFile?.mkdirs()
        val append = output.isFile && output.length() > 0
        writer = BufferedWriter(
            OutputStreamWriter(FileOutputStream(output, append), Charsets.UTF_8),
            DEFAULT_BUFFER_SIZE,
        ).also {
            if (!append) {
                it.write(
                    "elapsed_realtime_ns,wall_time_ms,stage,pid,process_total_pss_kb," +
                        "native_heap_allocated_bytes,thermal_status,battery_level_percent," +
                        "battery_temperature_deci_c,battery_voltage_mv,battery_status," +
                        "battery_current_now_ua,battery_charge_counter_uah,battery_energy_counter_nwh\n")
                it.flush()
            }
        }
        appContext = context.applicationContext
        outputFile = output
        stage = initialStage
        executor.scheduleAtFixedRate(::sample, 0, 1, TimeUnit.SECONDS)
    }

    fun setStage(value: String) {
        stage = value
    }

    fun stop() = synchronized(lock) { stopLocked() }

    private fun stopLocked() {
        executor.shutdownNow()
        runCatching { writer?.flush() }
        runCatching { writer?.close() }
        writer = null
        appContext = null
        outputFile = null
    }

    private fun sample() {
        val context = appContext ?: return
        val activity = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memory = activity.getProcessMemoryInfo(intArrayOf(Process.myPid())).firstOrNull()
        val power = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = battery?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
        val levelPercent = if (level >= 0 && scale > 0) 100.0 * level / scale else Double.NaN
        fun property(id: Int): Long = batteryManager.getLongProperty(id)
        val row = listOf(
            SystemClock.elapsedRealtimeNanos(),
            System.currentTimeMillis(),
            stage,
            Process.myPid(),
            memory?.totalPss ?: -1,
            Debug.getNativeHeapAllocatedSize(),
            power.currentThermalStatus,
            levelPercent,
            battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1,
            battery?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1) ?: -1,
            battery?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1,
            property(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW),
            property(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER),
            property(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER),
        ).joinToString(",")
        synchronized(lock) {
            runCatching {
                writer?.write(row + "\n")
                writer?.flush()
            }
        }
    }
}
