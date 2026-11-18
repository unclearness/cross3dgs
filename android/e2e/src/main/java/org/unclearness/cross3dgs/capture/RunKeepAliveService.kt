package org.unclearness.cross3dgs.capture

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder

/**
 * Foreground "keep-alive" service held while a reconstruction runs. The heavy
 * work stays in the app process (JNI thread started from MainActivity); this
 * service only raises the process priority so aggressive OEM task killers
 * (MIUI) don't kill a long CPU-bound run when the app loses foreground.
 */
class RunKeepAliveService : Service() {

    companion object {
        private const val CHANNEL_ID = "sfm_run"
        private const val NOTIFICATION_ID = 1
        const val ACTION_CANCEL = "org.unclearness.cross3dgs.capture.CANCEL_RUN"

        /**
         * How to ask the running stage to stop, installed by whoever owns the
         * work. The notification outlives the screen that started the run --
         * that is the point of the service -- so without this there is no way
         * back to a stop button once that screen is gone.
         *
         * Must not capture an Activity: it is held for the life of the
         * process, which continues past the Activity it was set from.
         */
        @Volatile
        @JvmStatic
        var onCancelRequested: (() -> Unit)? = null
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_CANCEL) {
            onCancelRequested?.invoke()
            // The stage still has to unwind, so the service stays up until the
            // work that started it stops it; the label says the request landed.
            updateNotification("Stopping", "Cancelling the current stage")
            return START_NOT_STICKY
        }
        val cancel = PendingIntent.getService(
            this, 0,
            Intent(this, RunKeepAliveService::class.java).setAction(ACTION_CANCEL),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        val notification = buildNotification(
            "Reconstruction running", "Computing the 3D reconstruction", cancel)
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(NOTIFICATION_ID, notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
        return START_NOT_STICKY
    }

    private fun buildNotification(
        title: String, text: String, cancel: PendingIntent?,
    ): Notification {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(CHANNEL_ID, "Reconstruction",
                NotificationManager.IMPORTANCE_LOW))
        val builder = Notification.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setContentTitle(title)
            .setContentText(text)
            .setOngoing(true)
        if (cancel != null) {
            builder.addAction(
                Notification.Action.Builder(
                    null, "Stop", cancel).build())
        }
        return builder.build()
    }

    private fun updateNotification(title: String, text: String) {
        getSystemService(NotificationManager::class.java).notify(
            NOTIFICATION_ID, buildNotification(title, text, null))
    }

    override fun onBind(intent: Intent?): IBinder? = null
}

