// Native optimizer and free-view renderer for the iOS capture-to-view app.
//
// The iOS counterpart of the training/viewer half of
// android/app/src/main/cpp/jni_bridge.cpp, which the Android :e2e module
// drives through NativeBridge.startTraining / prepareViewer / renderOrbit.
// The optimizer configuration, held-out evaluation, artifacts
// (splat.ply, checkpoint.d3gs, render/, target/, step_times_ms.csv) and the
// bench.json schema are kept the same so a run is comparable across
// platforms; only the JNI/ANativeWindow plumbing is replaced by callbacks.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "cross3dgs/training_preview.h"

namespace cross3dgs_ios {

// 8-bit RGBA image delivered to the UI (preview frames, orbit renders).
struct RgbaImage {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> pixels;
};

struct TrainingRequest {
    std::string scene_dir;
    std::string output_dir;
    std::string image_dir = "training/images";
    std::string sparse_dir = "training/sparse/0";
    uint32_t iterations = 3000;
    uint32_t progress_every = 25;
    float ssim_weight = 0.2f;
    uint32_t sh_degree = 3;
    std::string density_strategy = "mcmc";
    uint32_t gaussian_capacity = 250016;
    std::string view_order = "random";
    std::string adam_moments = "fp32";
    bool full_image_ssim = true;
    float mcmc_clone_offset = 0.5f;
    float mcmc_gradient_sampling = 1.5f;
};

struct EngineCallbacks {
    std::function<void(const std::string&)> log;
    std::function<void(int iteration, int total, float loss,
                       double elapsed_seconds)> progress;
    std::function<void(const RgbaImage&)> frame;
    // Panel positions of the training frames, sent before the first one.
    std::function<void(const cross3dgs::TrainingPreviewLayout&)>
            training_preview_layout;
    std::function<void(int exit_code, const std::string& message,
                       const std::string& ply_path)> finished;
};

class Engine {
public:
    static Engine& instance();

    void setCallbacks(EngineCallbacks callbacks);

    // Per-tile tile-Gaussian pair budget; rejected while a run is active.
    bool setTileBinCapacity(uint32_t value);
    // Starts training on a background thread; false if one is running or the
    // request is invalid. finished() is always called once on success.
    bool startTraining(const TrainingRequest& request);
    void requestStop();
    bool isRunning() const;

    bool prepareViewer(const std::string& scene_dir,
                       const std::string& image_dir,
                       const std::string& sparse_dir,
                       const std::string& checkpoint_path,
                       const std::string& reference_image_name,
                       // frame_priors.txt (may be empty); with gravity
                       // priors the free view is gravity-aligned
                       // (cross3dgs/orbit_view.h).
                       const std::string& frame_priors_path,
                       // Where to write the gravity-aligned PLY; empty, or a
                       // run without usable gravity priors, writes nothing.
                       const std::string& gravity_up_ply_path,
                       uint32_t sh_degree);
    // Pitch range of the prepared viewer (radians, contains 0); the view
    // clamps its accumulated vertical drag to it.
    void orbitPitchRange(float* min_pitch, float* max_pitch) const;
    // Renders the orbit view and delivers it through callbacks.frame.
    bool renderOrbit(float yaw, float pitch, float zoom, float pan_x,
                     float pan_y, uint32_t width, uint32_t height);

    std::string deviceCapsJson();

private:
    Engine() = default;
};

}  // namespace cross3dgs_ios
