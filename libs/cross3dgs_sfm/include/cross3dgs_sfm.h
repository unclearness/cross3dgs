// C interface to the CPU COLMAP sparse SfM shared by the Android and iOS
// capture-to-view apps. iOS calls it directly; Android goes through the thin
// JNI wrapper in jni/cross3dgs_sfm_jni.cc. See src/cross3dgs_sfm.cc.
#ifndef CROSS3DGS_SFM_H
#define CROSS3DGS_SFM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*cross3dgs_sfm_log_fn)(const char* line, void* user_data);

// Exit codes; the Android JNI wrapper returns them unchanged.
enum {
    CROSS3DGS_SFM_OK = 0,
    CROSS3DGS_SFM_FAILED = 1,
    CROSS3DGS_SFM_NO_MODEL = 2,
    CROSS3DGS_SFM_CANCELED = 3,
};

// Runs feature extraction, matching and mapping on the JPEGs in image_path and
// writes workspace/database.db plus workspace/sparse/<n>/. priors_path may be
// NULL or empty; otherwise it is the frame_priors.txt written by frame
// extraction. camera_model may be NULL (COLMAP's default, or PINHOLE when the
// priors carry intrinsics) or a COLMAP model name for inputs without AR
// priors; the optimizer consumes PINHOLE cameras only. log receives every
// COLMAP log line (from worker threads).
int cross3dgs_sfm_run(const char* image_path,
                      const char* workspace_path,
                      const char* priors_path,
                      const char* camera_model,
                      int use_global,
                      int single_camera,
                      cross3dgs_sfm_log_fn log,
                      void* log_user_data);

// Requests cancellation of the active run; safe from any thread.
void cross3dgs_sfm_cancel(void);

// Worker-thread counts for later runs: feature extraction, matching (and
// geometric verification) and mapping (bundle adjustment, triangulation).
// 0 keeps the library default (one per core; on iOS bounded by available
// memory, which an override never exceeds). For per-device tuning; thread
// counts change timing, not features, so apps record the values they set.
void cross3dgs_sfm_set_threads(int extraction, int matching, int mapping);

// Explicit seeds for benchmark runs; -1 retains ordinary application defaults.
// FAISS's clustering seed remains its pinned upstream default (1234).
void cross3dgs_sfm_set_seed(int seed);
// Reports the actual stage thread options chosen for the most recent run.
void cross3dgs_sfm_get_effective_threads(int* extraction, int* matching, int* mapping);

// Restricts later runs to the CPUs in `mask` (bit i = CPU i; 0 = no
// restriction). Applied to the calling thread for the duration of
// cross3dgs_sfm_run, so every worker COLMAP/Ceres starts inherits it; e.g.
// the big cores of a heterogeneous Android SoC, so no worker becomes a
// straggler on an efficiency core. Linux/Android only (no-op elsewhere).
void cross3dgs_sfm_set_cpu_mask(unsigned long long mask);

// ---- Live preview ----
//
// While cross3dgs_sfm_run works, the library can render what it is doing so
// both apps show the same picture: the latest image with its SIFT keypoints,
// the pair matrix filling in during matching, and the growing sparse model
// (points and camera frusta, turning about gravity) during mapping. Frames are
// rendered from read-only database queries and copies of the reconstruction
// on a separate library thread, so they never change the SfM result.

enum {
    CROSS3DGS_SFM_STAGE_FEATURES = 1,
    CROSS3DGS_SFM_STAGE_MATCHING = 2,
    CROSS3DGS_SFM_STAGE_MAPPING = 3,
    CROSS3DGS_SFM_STAGE_DONE = 4,
};

typedef struct cross3dgs_sfm_preview {
    const unsigned char* rgba;  // width * height * 4 bytes, row-major, opaque
    int width;
    int height;
    int stage;                  // CROSS3DGS_SFM_STAGE_*
    const char* caption;        // one line describing the frame
} cross3dgs_sfm_preview;

// Called on the preview thread, at most about four times per second; the
// frame is valid only during the call. No call happens after
// cross3dgs_sfm_run returns.
typedef void (*cross3dgs_sfm_preview_fn)(const cross3dgs_sfm_preview* frame,
                                         void* user_data);

// Enables the preview for subsequent runs at width x height (clamped to
// 64..1920); fn == NULL disables it. Call it before cross3dgs_sfm_run.
void cross3dgs_sfm_set_preview(cross3dgs_sfm_preview_fn fn,
                               void* user_data,
                               int width,
                               int height);

#ifdef __cplusplus
}
#endif

#endif
