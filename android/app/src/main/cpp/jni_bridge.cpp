#include <android/native_window_jni.h>
#include <jni.h>
#include <sys/system_properties.h>

#include <spdlog/sinks/android_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cross3dgs/dataset.h"
#include "cross3dgs/gaussian_forward.h"
#include "cross3dgs/gaussian_state.h"
#include "cross3dgs/gpu/gpu_program.h"
#include "cross3dgs/orbit_view.h"
#include "surface_renderer.h"

namespace cross3dgs_android {
std::string QueryDeviceCapsJson();
}

namespace cross3dgs_probe {
bool RunLoweringCandidateConformance(
        const std::filesystem::path& output_dir,
        const std::string& candidate_name);
bool RunSelectedScheduleCpuOracle(
        const std::filesystem::path& output_dir,
        bool force_split_sh_rest_storage = false);
}

namespace {

using Clock = std::chrono::steady_clock;
using cross3dgs::Camera;
using cross3dgs::FloatImage;
using cross3dgs::GaussianForwardProgram;
using cross3dgs::GaussianStateCpu;

JavaVM* g_vm = nullptr;
jobject g_listener = nullptr;
std::mutex g_listener_mutex;
std::atomic<bool> g_cancel{false};
std::atomic<bool> g_running{false};
// Per-tile tile-Gaussian pair budget. pair_capacity is this times the tile
// count, so a workload whose splats cover many tiles each can exhaust it and
// silently drop pairs. 2048 matches the desktop trainer; 512 overflowed
// mid-run on bicycle at one million Gaussians.
std::atomic<uint32_t> g_tile_bin_capacity{2048};
// Training-view sampling seed. Historically this was the compile-time
// constant 42, which made every Android run a repeat of one seed and made
// seed-spread studies impossible on-device. It is now settable per run; the
// default stays 42 so an unmodified launch reproduces the historical
// behavior, and bench.json records the value actually used.
std::atomic<uint32_t> g_training_view_seed{42};
// Seed of the random initial rotations when no checkpoint is given, as the
// desktop trainer's --init-seed.
std::atomic<uint32_t> g_init_seed{42};
// Off by default: the benchmark app keeps its render-only preview.
std::atomic<bool> g_training_preview_panels{false};
std::mutex g_training_preview_mutex;
cross3dgs::TrainingPreviewLayout g_training_preview_layout;
std::thread g_thread;

struct ResultFiles {
    std::array<std::filesystem::path, 8> targets;
    std::array<std::filesystem::path, 8> renders;
    bool ready = false;
};
ResultFiles g_results;
std::mutex g_results_mutex;

struct ViewerState {
    GaussianStateCpu gaussians;
    cross3dgs::OrbitReference orbit;
    uint32_t sh_degree = 0;
    uint32_t render_width = 0;
    uint32_t render_height = 0;
    std::unique_ptr<GaussianForwardProgram> renderer;
};
ViewerState g_viewer;
std::mutex g_viewer_mutex;

JNIEnv* GetEnv(bool& detach) {
    detach = false;
    if (g_vm == nullptr) return nullptr;
    JNIEnv* env = nullptr;
    if (g_vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) ==
        JNI_OK) {
        return env;
    }
    if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) return nullptr;
    detach = true;
    return env;
}

template <typename Callback>
void WithListener(Callback&& callback) {
    bool detach = false;
    JNIEnv* env = GetEnv(detach);
    if (env == nullptr) return;
    jobject listener = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_listener_mutex);
        if (g_listener != nullptr) listener = env->NewLocalRef(g_listener);
    }
    if (listener != nullptr) {
        callback(env, listener);
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
        }
        env->DeleteLocalRef(listener);
    }
    if (detach) g_vm->DetachCurrentThread();
}

std::string FromJString(JNIEnv* env, jstring value) {
    if (value == nullptr) return {};
    const char* text = env->GetStringUTFChars(value, nullptr);
    const std::string result = text != nullptr ? text : "";
    if (text != nullptr) env->ReleaseStringUTFChars(value, text);
    return result;
}

void NotifyLog(const std::string& line) {
    spdlog::info("{}", line);
    WithListener([&](JNIEnv* env, jobject listener) {
        const auto cls = env->GetObjectClass(listener);
        const auto method = env->GetMethodID(
                cls, "onLog", "(Ljava/lang/String;)V");
        const auto text = env->NewStringUTF(line.c_str());
        env->CallVoidMethod(listener, method, text);
        env->DeleteLocalRef(text);
        env->DeleteLocalRef(cls);
    });
}

void NotifyProgress(int iteration, int total, float loss,
                    double elapsed_seconds) {
    WithListener([&](JNIEnv* env, jobject listener) {
        const auto cls = env->GetObjectClass(listener);
        const auto method = env->GetMethodID(cls, "onProgress", "(IIFD)V");
        env->CallVoidMethod(listener, method, static_cast<jint>(iteration),
                            static_cast<jint>(total),
                            static_cast<jfloat>(loss),
                            static_cast<jdouble>(elapsed_seconds));
        env->DeleteLocalRef(cls);
    });
}

void NotifyFinished(int exit_code, const std::string& message,
                    const std::string& ply_path) {
    WithListener([&](JNIEnv* env, jobject listener) {
        const auto cls = env->GetObjectClass(listener);
        const auto method = env->GetMethodID(
                cls, "onFinished",
                "(ILjava/lang/String;Ljava/lang/String;)V");
        const auto message_string = env->NewStringUTF(message.c_str());
        const auto ply_string = env->NewStringUTF(ply_path.c_str());
        env->CallVoidMethod(listener, method, static_cast<jint>(exit_code),
                            message_string, ply_string);
        env->DeleteLocalRef(message_string);
        env->DeleteLocalRef(ply_string);
        env->DeleteLocalRef(cls);
    });
}

std::string ViewFilename(size_t view) {
    std::ostringstream name;
    name << "view_" << std::setfill('0') << std::setw(3) << view << ".png";
    return name.str();
}

double Median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const double upper = values[middle];
    if (values.size() % 2 != 0) return upper;
    const double lower =
            *std::max_element(values.begin(), values.begin() + middle);
    return 0.5 * (lower + upper);
}

size_t CountNonFinite(const std::vector<glm::vec4>& values) {
    size_t count = 0;
    for (const auto& value : values) {
        for (int component = 0; component < 4; component++) {
            if (!std::isfinite(value[component])) count++;
        }
    }
    return count;
}

void PublishResults(const ResultFiles& results) {
    std::lock_guard<std::mutex> lock(g_results_mutex);
    g_results = results;
}

void PublishViewerState(GaussianStateCpu state,
                        const cross3dgs::OrbitReference& orbit,
                        uint32_t sh_degree) {
    std::lock_guard<std::mutex> lock(g_viewer_mutex);
    g_viewer.renderer.reset();
    g_viewer.gaussians = std::move(state);
    g_viewer.orbit = orbit;
    g_viewer.sh_degree = sh_degree;
    g_viewer.render_width = 0;
    g_viewer.render_height = 0;
}

struct RunResult {
    int exit_code = 2;
    std::string message;
    std::string ply_path;
};

RunResult RunTraining(const std::filesystem::path& scene_dir,
                      const std::filesystem::path& output_dir,
                      const std::string& image_dir,
                      const std::string& sparse_dir,
                      uint32_t iterations, uint32_t view_limit,
                      uint32_t progress_every,
                      uint32_t stochastic_samples, float ssim_weight,
                      const std::string& init_checkpoint,
                      uint32_t sh_degree,
                      const std::string& density_strategy,
                      size_t requested_gaussian_capacity,
                      const std::string& view_order,
                      const std::string& adam_moments,
                      bool full_image_ssim,
                      float mcmc_clone_offset,
                      float mcmc_gradient_sampling,
                      bool force_atomic_free_exact,
                      const std::string& backward_candidate) {
    const auto run_start = Clock::now();
    NotifyLog("Loading COLMAP metadata and available training views");
    if (image_dir.empty() || sparse_dir.empty()) {
        throw std::invalid_argument("image_dir and sparse_dir must be set");
    }
    const auto scene = cross3dgs::LoadColmapScene(
            scene_dir, image_dir, sparse_dir);
    std::vector<Camera> cameras;
    std::vector<Camera> validation_cameras;
    for (const auto& camera : scene.cameras) {
        if (!std::filesystem::is_regular_file(camera.image_path)) continue;
        if (camera.validation) validation_cameras.push_back(camera);
        else cameras.push_back(camera);
    }
    if (cameras.empty()) {
        throw std::runtime_error("no training images are available");
    }
    if (view_limit > 0 && cameras.size() > view_limit) {
        cameras.resize(view_limit);
    }
    NotifyLog("Training views: " + std::to_string(cameras.size()) +
              (view_limit > 0 ? " (limit " + std::to_string(view_limit) + ")"
                              : "") +
              ", validation views: " +
              std::to_string(validation_cameras.size()));

    auto state = cross3dgs::InitializeFromSparsePoints(
            scene, 0.1f, 0.5f, g_init_seed.load());
    std::string init_source = "sparse_points";
    if (!init_checkpoint.empty()) {
        const auto checkpoint_path = scene_dir / init_checkpoint;
        if (!std::filesystem::is_regular_file(checkpoint_path)) {
            throw std::runtime_error("initial checkpoint not found: " +
                                     checkpoint_path.string());
        }
        state = cross3dgs::LoadGaussianCheckpoint(checkpoint_path);
        init_source = init_checkpoint;
        NotifyLog("Initialized " + std::to_string(state.size()) +
                  " Gaussians from checkpoint " + init_checkpoint);
    } else {
        NotifyLog("Initializing fixed Gaussians from sparse points");
    }
    const size_t initial_gaussian_count = state.size();
    const bool enable_mcmc = density_strategy == "mcmc";
    const size_t resolved_gaussian_capacity =
            requested_gaussian_capacity == 0
                    ? initial_gaussian_count
                    : requested_gaussian_capacity;
    if (resolved_gaussian_capacity < initial_gaussian_count) {
        throw std::invalid_argument(
                "Gaussian capacity is below initial count");
    }
    if (enable_mcmc && resolved_gaussian_capacity == initial_gaussian_count) {
        throw std::invalid_argument(
                "MCMC requires capacity above initial count");
    }
    // The trainer allocates whole blocks of 32 Gaussians; record both.
    const size_t gaussian_capacity =
            cross3dgs::AlignGaussianCapacity(resolved_gaussian_capacity);
    const float scene_scale =
            cross3dgs::CameraExtentSceneScale(scene.cameras);
    NotifyLog("Scene scale: " + std::to_string(scene_scale) +
              ", ssim weight: " + std::to_string(ssim_weight) +
              ", density: " + density_strategy +
              ", capacity: " + std::to_string(gaussian_capacity));
    cross3dgs::gpu::CustomGpuProgram context_owner;
    const auto& caps = context_owner.deviceCaps();
    if (caps.max_workgroup_invocations < 256 ||
        caps.max_shared_memory < 16384 || caps.subgroup_size < 8 ||
        caps.subgroup_size > 128) {
        throw std::runtime_error(
                "GPU does not satisfy the 256-thread/subgroup/shared-memory "
                "requirements");
    }
    const bool force_per_splat_pair =
            backward_candidate == "per_splat_pair_reduce";
    const bool force_per_splat_cas =
            backward_candidate == "per_splat_cas";
    const bool force_pixel_cas =
            backward_candidate == "pixel_subgroup_cas";
    const bool force_exact_pair =
            backward_candidate == "exact_pair_gather";
    if (stochastic_samples != 0 &&
        (force_per_splat_pair || force_per_splat_cas || force_pixel_cas ||
         force_exact_pair)) {
        throw std::invalid_argument(
                "exact lowering candidate requires stochastic_samples=0");
    }
    // backward_candidate "auto" follows the schedule ledger's policy. On
    // Adreno that is per-splat CAS: HC-ATOMIC-SCATTER-QCOM-01 rejects the
    // pixel/subgroup, tiled and native scatter that fault there and admits
    // per-splat CAS. The gather variants remain selectable by name.
    const bool atomic_free_exact =
            stochastic_samples == 0 &&
            (force_atomic_free_exact || force_per_splat_pair ||
             force_exact_pair) &&
            !force_per_splat_cas && !force_pixel_cas;
    if (atomic_free_exact) {
        NotifyLog(force_atomic_free_exact
                          ? "Forced atomic-free exact gather"
                          : "Requested atomic-free gather: " +
                                    backward_candidate);
    }
    const auto context = context_owner.context();
    const double input_setup_seconds =
            std::chrono::duration<double>(Clock::now() - run_start).count();

    std::vector<double> step_milliseconds;
    step_milliseconds.reserve(iterations);
    std::vector<double> step_elapsed_seconds;
    step_elapsed_seconds.reserve(iterations);
    struct LossPoint {
        uint32_t iteration;
        float loss;
        double elapsed_seconds;
    };
    std::vector<LossPoint> loss_curve;
    loss_curve.reserve(iterations / std::max(progress_every, 1u) + 2);
    uint32_t completed = 0;
    float final_loss = 0.0f;
    cross3dgs::GaussianForwardStats training_stats;
    cross3dgs::GaussianScheduleIR training_schedule;
    double setup_pipeline_upload_seconds = 0.0;
    double optimization_loop_wall_seconds = 0.0;
    double progress_loss_readback_seconds = 0.0;
    double preview_readback_show_host_seconds = 0.0;
    double final_state_readback_seconds = 0.0;
    const bool preview_panels = g_training_preview_panels.load();
    const auto training_start = Clock::now();
    {
        NotifyLog("Building Vulkan training pipelines");
        NotifyLog(stochastic_samples == 0
                          ? (atomic_free_exact
                                     ? "Backward mode: atomic-free exact gather"
                                     : "Backward mode: exact scatter")
                          : "Backward mode: Sobol gather K=" +
                                    std::to_string(stochastic_samples));
        const auto pipeline_setup_start = Clock::now();
        std::filesystem::create_directories(output_dir);
        GaussianForwardProgram trainer(
                {.width = cameras[0].width,
                 .height = cameras[0].height,
                 .gaussian_capacity = gaussian_capacity,
                 .tile_size = 16,
                 .tile_bin_capacity = g_tile_bin_capacity.load(),
                 .enable_backward = true,
                 .enable_optimizer = true,
                 .resident_view_capacity =
                         static_cast<uint32_t>(cameras.size()),
                 .randomize_training_views = view_order == "random",
                 .training_view_seed = g_training_view_seed.load(),
                 .sh_degree = sh_degree,
                 // Mirror the desktop fixed-strategy learning-rate
                 // normalization (train_cli): the classic 3DGS mean LR is
                 // scaled by the camera-extent scene scale.
                 .means_lr = 0.00016f * scene_scale,
                 .means_lr_final = 0.0000016f * scene_scale,
                 .l1_weight = 1.0f - ssim_weight,
                 .ssim_weight = ssim_weight,
                 .full_image_ssim = full_image_ssim,
                 .enable_mcmc = enable_mcmc,
                 .mcmc_clone_offset = mcmc_clone_offset,
                 .mcmc_gradient_sampling = mcmc_gradient_sampling,
                 .scale_regularizer = enable_mcmc ? 0.01f : 0.0f,
                 .opacity_regularizer = enable_mcmc ? 0.01f : 0.0f,
                 .fp32_adam_moments = adam_moments == "fp32",
                 .stochastic_backward_samples = stochastic_samples,
                 .atomic_free_exact_backward = atomic_free_exact,
                 .force_pixel_subgroup_backward = force_pixel_cas,
                 .force_per_splat_backward =
                         force_per_splat_pair || force_per_splat_cas,
                 .force_cas_scatter_backward =
                         force_pixel_cas || force_per_splat_cas,
                 .byte_targets = true},
                context);
        {
            std::ofstream ledger_output(output_dir / "candidate_ledger.json",
                                        std::ios::binary);
            ledger_output << trainer.candidateLedgerJson();
            if (!ledger_output) {
                throw std::runtime_error(
                        "failed to write runtime candidate ledger");
            }
        }
        trainer.uploadState(state);
        // Stream one decoded view at a time; the GPU keeps every target
        // resident as packed RGBA8, so full desktop view counts fit.
        trainer.uploadTrainingSet(cameras, [&](size_t view) {
            if (view % 32 == 0 || view + 1 == cameras.size()) {
                NotifyLog("Uploading targets " + std::to_string(view + 1) +
                          "/" + std::to_string(cameras.size()));
            }
            return cross3dgs::LoadImageRgbaF32(
                    cameras[view].image_path);
        });
        setup_pipeline_upload_seconds = std::chrono::duration<double>(
                Clock::now() - pipeline_setup_start).count();
        {
            char diag[PROP_VALUE_MAX] = {0};
            __system_property_get("debug.cross3dgs.diag", diag);
            if (diag[0] == '1') {
                NotifyLog("DIAG: per-pass isolation (5 s/pass timeout)");
                const std::string hung =
                        trainer.firstHangingPass(5000000000ull);
                const std::string message =
                        hung.empty()
                                ? "DIAG: all passes completed, no hang"
                                : "DIAG: first hanging pass = " + hung;
                NotifyLog(message);
                return {0, message, ""};
            }
        }
        {
            const glm::uvec4 m = trainer.debugMeta();
            NotifyLog("DIAG: meta before training = (" +
                      std::to_string(m.x) + "," + std::to_string(m.y) + "," +
                      std::to_string(m.z) + "," + std::to_string(m.w) + ")");
        }
        bool trace_forward_counts = false;
        {
            char diag[PROP_VALUE_MAX] = {0};
            __system_property_get("debug.cross3dgs.diag", diag);
            trace_forward_counts = diag[0] == '2';
        }
        if (preview_panels) {
            std::lock_guard<std::mutex> lock(g_training_preview_mutex);
            g_training_preview_layout = cross3dgs::MakeTrainingPreviewLayout(
                    cameras[0].width, cameras[0].height);
        }
        NotifyLog("Optimization started");
        const auto optimization_loop_start = Clock::now();
        for (uint32_t iteration = 0; iteration < iterations; iteration++) {
            if (g_cancel.load(std::memory_order_relaxed)) break;
            const auto step_start = Clock::now();
            try {
                trainer.execute();
            } catch (const std::exception& error) {
                throw std::runtime_error(
                        "iteration " + std::to_string(iteration + 1) +
                        " view " +
                        std::to_string(iteration % cameras.size() + 1) +
                        "/" + std::to_string(cameras.size()) + ": " +
                        error.what());
            }
            const auto step_end = Clock::now();
            step_milliseconds.push_back(
                    std::chrono::duration<double, std::milli>(
                            step_end - step_start).count());
            step_elapsed_seconds.push_back(
                    std::chrono::duration<double>(
                            step_end - optimization_loop_start).count());
            completed = iteration + 1;
            const bool report = completed == 1 ||
                    completed == iterations ||
                    completed % std::max(progress_every, 1u) == 0;
            if (report) {
                const auto loss_readback_start = Clock::now();
                final_loss = trainer.downloadMeanL1();
                progress_loss_readback_seconds +=
                        std::chrono::duration<double>(
                                Clock::now() - loss_readback_start).count();
                if (trace_forward_counts) {
                    const glm::uvec4 f = trainer.debugForwardCounts();
                    const glm::uvec4 m = trainer.debugMeta();
                    const glm::uvec4 p = trainer.debugPairIntegrity();
                    NotifyLog("DIAG: iteration=" + std::to_string(completed) +
                              " loss=" + std::to_string(final_loss) +
                              " coverage=" + std::to_string(f.x) +
                              " total_pairs=" + std::to_string(f.z) +
                              " meta=(" + std::to_string(m.x) + "," +
                              std::to_string(m.y) + "," +
                              std::to_string(m.z) + "," +
                              std::to_string(m.w) + ") pairs=(" +
                              std::to_string(p.x) + "," +
                              std::to_string(p.y) + "," +
                              std::to_string(p.z) + "," +
                              std::to_string(p.w) + ")");
                }
                const auto preview_start = Clock::now();
                cross3dgs_android::SurfaceRenderer::instance().show(
                        preview_panels
                                ? cross3dgs::ComposeTrainingPreview(
                                          trainer.downloadTrainingPreview())
                                : trainer.downloadColor());
                preview_readback_show_host_seconds +=
                        std::chrono::duration<double>(
                                Clock::now() - preview_start).count();
                const double seconds =
                        std::chrono::duration<double>(
                                Clock::now() - training_start).count();
                loss_curve.push_back({completed, final_loss, seconds});
                const auto progress_stats = trainer.stats();
                spdlog::info("progress iteration={}/{} l1={:f} "
                             "elapsed_s={:.2f} tracked_current_bytes={} "
                             "tracked_peak_bytes={}",
                             completed, iterations, final_loss, seconds,
                             progress_stats.current_memory_bytes,
                             progress_stats.peak_memory_bytes);
                NotifyProgress(static_cast<int>(completed),
                               static_cast<int>(iterations), final_loss,
                               seconds);
            }
        }
        optimization_loop_wall_seconds = std::chrono::duration<double>(
                Clock::now() - optimization_loop_start).count();
        training_stats = trainer.stats();
        training_schedule = trainer.schedule();
        {
            const glm::uvec4 f = trainer.debugForwardCounts();
            NotifyLog("DIAG: coverage=" + std::to_string(f.x) +
                      " total_pairs=" + std::to_string(f.z));
        }
        const auto final_state_readback_start = Clock::now();
        state = trainer.downloadState();
        final_state_readback_seconds = std::chrono::duration<double>(
                Clock::now() - final_state_readback_start).count();
        NotifyLog("DIAG: state count after training = " +
                  std::to_string(state.size()));
    }
    if (completed == 0) {
        throw std::runtime_error("optimization stopped before the first step");
    }
    const double training_seconds =
            std::chrono::duration<double>(Clock::now() - training_start)
                    .count();
    const size_t non_finite_means = CountNonFinite(state.means);
    const size_t non_finite_scales_opacity =
            CountNonFinite(state.scales_opacity);
    const size_t non_finite_rotations = CountNonFinite(state.rotations);
    const size_t non_finite_sh_dc = CountNonFinite(state.sh_dc);
    const size_t non_finite_sh_rest = CountNonFinite(state.sh_rest);
    const size_t non_finite =
            non_finite_means + non_finite_scales_opacity +
            non_finite_rotations + non_finite_sh_dc + non_finite_sh_rest;
    if (non_finite != 0) {
        NotifyLog("DIAG: non-finite means=" +
                  std::to_string(non_finite_means) +
                  " scales_opacity=" +
                  std::to_string(non_finite_scales_opacity) +
                  " rotations=" + std::to_string(non_finite_rotations) +
                  " sh_dc=" + std::to_string(non_finite_sh_dc) +
                  " sh_rest=" + std::to_string(non_finite_sh_rest));
        throw std::runtime_error(
                "optimization produced non-finite Gaussian parameters");
    }

    const auto artifact_start = Clock::now();
    std::filesystem::create_directories(output_dir);
    const auto ply_path = output_dir / "splat.ply";
    cross3dgs::SaveGaussianPly(ply_path, state);
    NotifyLog("Saved " + ply_path.string());
    cross3dgs::SaveGaussianCheckpoint(
            output_dir / "checkpoint.d3gs", state);

    {
        std::ofstream steps(output_dir / "step_times_ms.csv");
        steps << "iteration,elapsed_seconds,host_step_ms,step_ms\n";
        for (size_t index = 0; index < step_milliseconds.size(); index++) {
            steps << index + 1 << ',' << step_elapsed_seconds[index] << ','
                  << step_milliseconds[index] << ','
                  << step_milliseconds[index] << '\n';
        }
    }
    const double artifact_io_seconds = std::chrono::duration<double>(
            Clock::now() - artifact_start).count();

    // Evaluate held-out validation views when their images are installed;
    // fall back to the training views otherwise. Every evaluated render is
    // saved so the host-side common evaluator can recompute quality.
    const bool has_validation = !validation_cameras.empty();
    const auto& evaluation_cameras =
            has_validation ? validation_cameras : cameras;
    NotifyLog(std::string("Evaluating ") +
              std::to_string(evaluation_cameras.size()) +
              (has_validation ? " validation" : " training") + " views");
    const auto evaluation_start = Clock::now();
    double evaluation_artifact_io_seconds = 0.0;
    ResultFiles result_files;
    struct ViewMetric {
        std::string name;
        double l1 = 0.0;
        double mse = 0.0;
    };
    std::vector<ViewMetric> view_metrics;
    view_metrics.reserve(evaluation_cameras.size());
    double absolute_error = 0.0;
    double squared_error = 0.0;
    size_t value_count = 0;
    double psnr_per_view_sum = 0.0;
    bool trace_evaluation = false;
    {
        char diag[PROP_VALUE_MAX] = {0};
        __system_property_get("debug.cross3dgs.diag", diag);
        trace_evaluation = diag[0] == '2';
    }
    {
        GaussianForwardProgram evaluator(
                {.width = evaluation_cameras[0].width,
                 .height = evaluation_cameras[0].height,
                 .gaussian_capacity = state.size(),
                 .tile_size = 16,
                 .tile_bin_capacity = g_tile_bin_capacity.load(),
                 .sh_degree = sh_degree},
                context);
        evaluator.uploadState(state);
        for (size_t view = 0; view < evaluation_cameras.size(); view++) {
            if (trace_evaluation) {
                NotifyLog("DIAG: evaluation view " + std::to_string(view) +
                          " begin");
            }
            evaluator.setCamera(evaluation_cameras[view]);
            evaluator.execute();
            const auto rendered = evaluator.downloadColor();
            const auto target = cross3dgs::LoadImageRgbaF32(
                    evaluation_cameras[view].image_path);
            const auto filename = ViewFilename(view);
            const auto render_path = output_dir / "render" / filename;
            const auto target_path = output_dir / "target" / filename;
            const auto evaluation_artifact_start = Clock::now();
            cross3dgs::SaveImagePng(render_path, rendered);
            cross3dgs::SaveImagePng(target_path, target);
            evaluation_artifact_io_seconds +=
                    std::chrono::duration<double>(
                            Clock::now() - evaluation_artifact_start).count();
            if (view < result_files.targets.size()) {
                result_files.targets[view] = target_path;
                result_files.renders[view] = render_path;
            }
            const size_t pixels =
                    static_cast<size_t>(rendered.width) * rendered.height;
            double view_absolute = 0.0;
            double view_squared = 0.0;
            for (size_t pixel = 0; pixel < pixels; pixel++) {
                for (size_t channel = 0; channel < 3; channel++) {
                    const double difference =
                            rendered.pixels[pixel * 4 + channel] -
                            target.pixels[pixel * 4 + channel];
                    view_absolute += std::abs(difference);
                    view_squared += difference * difference;
                }
            }
            const double view_values = static_cast<double>(pixels) * 3.0;
            view_metrics.push_back(
                    {evaluation_cameras[view].image_name,
                     view_absolute / view_values,
                     view_squared / view_values});
            psnr_per_view_sum +=
                    -10.0 * std::log10(view_squared / view_values);
            absolute_error += view_absolute;
            squared_error += view_squared;
            value_count += pixels * 3;
            // With panels the surface stays on training until the viewer.
            if (view == 0 && !preview_panels) {
                cross3dgs_android::SurfaceRenderer::instance().show(
                        rendered);
            }
        }
        if (trace_evaluation) {
            NotifyLog("DIAG: evaluation loop complete");
        }
    }
    result_files.ready = true;
    PublishResults(result_files);
    const double mean_l1 = absolute_error / value_count;
    const double mse = squared_error / value_count;
    const double psnr = -10.0 * std::log10(mse);
    const double mean_view_psnr =
            psnr_per_view_sum / static_cast<double>(view_metrics.size());
    const double evaluation_seconds = std::chrono::duration<double>(
            Clock::now() - evaluation_start).count();
    double step_host_sum_seconds = 0.0;
    for (const double value : step_milliseconds) {
        step_host_sum_seconds += value / 1000.0;
    }
    const size_t warmup =
            std::min<size_t>(100, step_milliseconds.size() / 10);
    std::vector<double> steady(
            step_milliseconds.begin() + warmup,
            step_milliseconds.end());
    std::sort(steady.begin(), steady.end());
    const auto percentile = [&](double fraction) {
        const double position = fraction * double(steady.size() - 1);
        const size_t lower = static_cast<size_t>(position);
        const size_t upper = std::min(lower + 1, steady.size() - 1);
        const double weight = position - double(lower);
        return steady[lower] * (1.0 - weight) + steady[upper] * weight;
    };
    const double run_wall_seconds = std::chrono::duration<double>(
            Clock::now() - run_start).count();

    std::ofstream benchmark(output_dir / "bench.json");
    if (!benchmark) {
        throw std::runtime_error("failed to create bench.json");
    }
    benchmark << "{\n"
              << "  \"schema_version\": 3,\n"
              << "  \"backend\": \"Cross3DGS Vulkan Android\",\n"
              << "  \"scene\": \"" << scene_dir.filename().string()
              << "\",\n"
              << "  \"image_dir\": \"" << image_dir << "\",\n"
              << "  \"sparse_dir\": \"" << sparse_dir << "\",\n"
              << "  \"sampled_train_image_count\": " << cameras.size()
              << ",\n"
              << "  \"view_limit\": " << view_limit << ",\n"
              << "  \"validation_image_count\": "
              << validation_cameras.size() << ",\n"
              << "  \"evaluation_split\": \""
              << (has_validation ? "validation" : "train") << "\",\n"
              << "  \"init_source\": \"" << init_source << "\",\n"
              << "  \"init_seed\": " << g_init_seed.load() << ",\n"
              << "  \"tile_bin_capacity\": " << g_tile_bin_capacity.load()
              << ",\n"
              << "  \"scene_scale\": " << scene_scale << ",\n"
              << "  \"sh_degree\": " << sh_degree << ",\n"
              << "  \"l1_weight\": " << (1.0f - ssim_weight) << ",\n"
              << "  \"ssim_weight\": " << ssim_weight << ",\n"
              << "  \"full_image_ssim\": "
              << (full_image_ssim ? "true" : "false") << ",\n"
              << "  \"density_strategy\": \"" << density_strategy
              << "\",\n"
              << "  \"view_order\": \"" << view_order << "\",\n"
              << "  \"adam_moments\": \"" << adam_moments << "\",\n"
              << "  \"mcmc_clone_offset\": " << mcmc_clone_offset
              << ",\n"
              << "  \"mcmc_gradient_sampling\": "
              << mcmc_gradient_sampling << ",\n"
              << "  \"training_view_seed\": "
              << g_training_view_seed.load() << ",\n"
              << "  \"alpha_cap\": 0.999,\n"
              << "  \"requested_gaussian_capacity\": "
              << resolved_gaussian_capacity << ",\n"
              << "  \"gaussian_capacity\": " << gaussian_capacity
              << ",\n"
              << "  \"initial_gaussian_count\": "
              << initial_gaussian_count << ",\n"
              << "  \"gaussian_count\": " << state.size() << ",\n"
              << "  \"requested_iterations\": " << iterations << ",\n"
              << "  \"completed_iterations\": " << completed << ",\n"
              << "  \"backward_mode\": \""
              << (stochastic_samples == 0
                          ? (atomic_free_exact ? "exact_gather"
                                               : "exact_scatter")
                          : "sobol_gather")
              << "\",\n"
              << "  \"requested_backward_candidate\": \""
              << backward_candidate << "\",\n"
              << "  \"stochastic_samples\": " << stochastic_samples
              << ",\n"
              << "  \"loss_curve_interval\": " << progress_every << ",\n"
              << "  \"first_iteration_ms\": " << step_milliseconds.front()
              << ",\n"
              << "  \"median_step_ms\": " << percentile(0.5) << ",\n"
              << "  \"p10_step_ms\": " << percentile(0.1) << ",\n"
              << "  \"p90_step_ms\": " << percentile(0.9) << ",\n"
              << "  \"p99_step_ms\": " << percentile(0.99) << ",\n"
              << "  \"total_train_seconds\": " << training_seconds << ",\n"
              << "  \"total_train_seconds_semantics\": "
                 "\"training_scope_including_pipeline_upload_progress_"
                 "preview_final_readback_and_teardown\",\n"
              << "  \"input_setup_seconds\": " << input_setup_seconds
              << ",\n"
              << "  \"setup_pipeline_upload_seconds\": "
              << setup_pipeline_upload_seconds << ",\n"
              << "  \"optimization_loop_wall_seconds\": "
              << optimization_loop_wall_seconds << ",\n"
              << "  \"step_host_sum_seconds\": "
              << step_host_sum_seconds << ",\n"
              << "  \"progress_loss_readback_seconds\": "
              << progress_loss_readback_seconds << ",\n"
              << "  \"preview_readback_show_host_seconds\": "
              << preview_readback_show_host_seconds << ",\n"
              << "  \"training_preview_panels\": "
              << (preview_panels ? "true" : "false") << ",\n"
              << "  \"preview_readback_show_host_semantics\": "
              << (preview_panels
                          ? "\"downloadTrainingPreview_compose_and_"
                            "SurfaceRenderer_show_host_call_"
                            "without_present_completion_guarantee\""
                          : "\"downloadColor_and_SurfaceRenderer_show_host_"
                            "call_without_present_completion_guarantee\"")
              << ",\n"
              << "  \"final_state_readback_seconds\": "
              << final_state_readback_seconds << ",\n"
              << "  \"artifact_io_seconds\": " << artifact_io_seconds
              << ",\n"
              << "  \"evaluation_seconds\": " << evaluation_seconds
              << ",\n"
              << "  \"evaluation_artifact_io_seconds\": "
              << evaluation_artifact_io_seconds << ",\n"
              << "  \"evaluation_compute_readback_present_seconds\": "
              << evaluation_seconds - evaluation_artifact_io_seconds
              << ",\n"
              << "  \"run_wall_seconds\": " << run_wall_seconds << ",\n"
              << "  \"run_wall_scope\": "
                 "\"native_run_start_through_evaluation_before_bench_write\",\n"
              << "  \"step_times_csv\": \"step_times_ms.csv\",\n"
              << "  \"step_times_elapsed_semantics\": "
                 "\"execute_completion_wall_since_optimization_loop_start\",\n"
              << "  \"step_times_host_step_semantics\": "
                 "\"single_execute_host_wall\",\n"
              << "  \"gpu_total_training_seconds\": null,\n"
              << "  \"gpu_total_training_seconds_note\": "
                 "\"not_reported_without_reliable_whole-run_timestamps\",\n"
              << "  \"last_training_view_l1\": " << final_loss << ",\n"
              << "  \"evaluation_mean_l1\": " << mean_l1 << ",\n"
              << "  \"evaluation_psnr\": " << psnr << ",\n"
              << "  \"evaluation_mean_view_psnr\": " << mean_view_psnr
              << ",\n"
              << "  \"overflow_count\": "
              << training_stats.overflow_count << ",\n"
              << "  \"overflow_update_count\": "
              << training_stats.overflow_update_count << ",\n"
              << "  \"peak_pair_count\": "
              << training_stats.peak_pair_count << ",\n"
              << "  \"max_tile_count\": "
              << training_stats.max_tile_count << ",\n"
              << "  \"tile_reference_count\": "
              << training_stats.tile_reference_count << ",\n"
              << "  \"pair_capacity\": "
              << training_stats.pair_capacity << ",\n"
              << "  \"atomic_free_pair_scratch_used_bytes\": "
              << training_stats.atomic_free_pair_scratch_used_bytes << ",\n"
              << "  \"atomic_free_pair_scratch_bytes\": "
              << training_stats.atomic_free_pair_scratch_bytes << ",\n"
              << "  \"state_non_finite\": " << non_finite << ",\n"
              << "  \"current_memory_bytes\": "
              << training_stats.current_memory_bytes << ",\n"
              << "  \"peak_memory_bytes\": "
              << training_stats.peak_memory_bytes << ",\n"
              << "  \"schedule_ir\": {\n"
              << "    \"vendor_id\": " << training_schedule.vendor_id
              << ",\n"
              << "    \"device_id\": " << training_schedule.device_id
              << ",\n"
              << "    \"subgroup_size\": "
              << training_schedule.subgroup_size << ",\n"
              << "    \"max_shared_memory\": "
              << training_schedule.max_shared_memory << ",\n"
              << "    \"max_storage_buffer_bindings\": "
              << training_schedule.max_storage_buffer_bindings << ",\n"
              << "    \"max_storage_buffer_range\": "
              << training_schedule.max_storage_buffer_range << ",\n"
              << "    \"tile_size\": " << training_schedule.tile_size
              << ",\n"
              << "    \"subtile_size\": " << training_schedule.subtile_size
              << ",\n"
              << "    \"raster_workgroup_size\": "
              << training_schedule.raster_workgroup_size << ",\n"
              << "    \"backward_workgroup_size\": "
              << training_schedule.backward_workgroup_size << ",\n"
              << "    \"projection_optimizer_workgroup_size\": "
              << training_schedule.projection_optimizer_workgroup_size
              << ",\n"
              << "    \"command_batch_size\": "
              << training_schedule.command_batch_size << ",\n"
              << "    \"backward_work_assignment\": \""
              << training_schedule.backward_work_assignment << "\",\n"
              << "    \"backward_global_merge\": \""
              << training_schedule.backward_global_merge << "\",\n"
              << "    \"backward_tile_shape\": \""
              << training_schedule.backward_tile_shape << "\",\n"
              << "    \"backward_selection_policy\": \""
              << training_schedule.backward_selection_policy << "\",\n"
              << "    \"schedule_candidate_id\": \""
              << training_schedule.schedule_candidate_id << "\",\n"
              << "    \"policy_selected_candidate_id\": \""
              << training_schedule.policy_selected_candidate_id << "\",\n"
              << "    \"schedule_catalog_version\": \""
              << training_schedule.schedule_catalog_version << "\",\n"
              << "    \"schedule_catalog_sha256\": \""
              << training_schedule.schedule_catalog_sha256 << "\",\n"
              << "    \"schedule_policy_id\": \""
              << training_schedule.schedule_policy_id << "\",\n"
              << "    \"schedule_policy_sha256\": \""
              << training_schedule.schedule_policy_sha256 << "\",\n"
              << "    \"schedule_selection_source\": \""
              << training_schedule.schedule_selection_source << "\",\n"
              << "    \"schedule_candidate_accepted\": "
              << (training_schedule.schedule_candidate_accepted
                          ? "true" : "false")
              << ",\n"
              << "    \"tiled_backward\": "
              << (training_schedule.tiled_backward ? "true" : "false")
              << ",\n"
              << "    \"per_splat_backward\": "
              << (training_schedule.per_splat_backward ? "true" : "false")
              << ",\n"
              << "    \"projection_optimizer_fusion\": "
              << (training_schedule.projection_optimizer_fusion
                          ? "true" : "false")
              << ",\n"
              << "    \"subgroup_reordered_sh\": "
              << (training_schedule.subgroup_reordered_sh
                          ? "true" : "false")
              << ",\n"
              << "    \"split_sh_rest_storage\": "
              << (training_schedule.split_sh_rest_storage
                          ? "true" : "false")
              << ",\n"
              << "    \"native_float_atomic_add\": "
              << (training_schedule.native_float_atomic_add
                          ? "true" : "false")
              << ",\n"
              << "    \"byte_targets\": "
              << (training_schedule.byte_targets ? "true" : "false")
              << ",\n    \"fenced_execution\": "
              << (training_schedule.fenced_execution ? "true" : "false")
              << ",\n    \"atomic_free_exact_backward\": "
              << (training_schedule.atomic_free_exact_backward
                          ? "true" : "false")
              << ",\n    \"streamed_target\": "
              << (training_schedule.streamed_target ? "true" : "false")
              << ",\n    \"mcmc_indirect_dispatch\": "
              << (training_schedule.mcmc_indirect_dispatch ? "true" : "false")
              << ",\n    \"portable_radix_sort\": "
              << (training_schedule.portable_radix_sort ? "true" : "false")
              << ",\n    \"indirect_pair_dispatch\": "
              << (training_schedule.indirect_pair_dispatch ? "true" : "false")
              << "\n  },\n"
              << "  \"stage_timings_ms\": {\n";
    for (size_t index = 0; index < training_stats.timings.size(); index++) {
        const auto& timing = training_stats.timings[index];
        benchmark << "    \"" << timing.label << "\": "
                  << timing.gpu_microseconds / 1000.0
                  << (index + 1 == training_stats.timings.size()
                              ? "\n"
                              : ",\n");
    }
    benchmark << "  },\n"
              << "  \"fenced_host_pass_timings_ms\": {\n";
    for (size_t index = 0;
         index < training_stats.fenced_host_timings.size(); index++) {
        const auto& timing = training_stats.fenced_host_timings[index];
        benchmark << "    \"" << timing.label << "\": "
                  << timing.host_microseconds / 1000.0
                  << (index + 1 == training_stats.fenced_host_timings.size()
                              ? "\n"
                              : ",\n");
    }
    benchmark << "  },\n"
              << "  \"evaluation_views\": [\n";
    for (size_t index = 0; index < view_metrics.size(); index++) {
        const auto& metric = view_metrics[index];
        benchmark << "    {\"view\": " << index
                  << ", \"image\": \"" << metric.name << "\""
                  << ", \"l1\": " << metric.l1
                  << ", \"mse\": " << metric.mse << "}"
                  << (index + 1 == view_metrics.size() ? "\n" : ",\n");
    }
    benchmark << "  ],\n"
              << "  \"loss_curve\": [\n";
    for (size_t index = 0; index < loss_curve.size(); index++) {
        const auto& point = loss_curve[index];
        benchmark << "    {\"iteration\": " << point.iteration
                  << ", \"l1\": " << point.loss
                  << ", \"elapsed_seconds\": "
                  << point.elapsed_seconds << "}"
                  << (index + 1 == loss_curve.size() ? "\n" : ",\n");
    }
    benchmark << "  ]\n}\n";

    std::ostringstream message;
    message << (g_cancel.load() ? "Stopped" : "Finished") << " "
            << completed << "/" << iterations << " in "
            << std::fixed << std::setprecision(2) << training_seconds
            << " s; L1 " << std::setprecision(6) << mean_l1
            << ", PSNR " << std::setprecision(3) << psnr << " dB";
    if (training_stats.overflow_count != 0) {
        message << "; WARNING overflow=" << training_stats.overflow_count;
    }
    if (!g_cancel.load()) {
        // Result browsing only; the free view re-prepares through
        // prepareViewer, which adds gravity alignment.
        PublishViewerState(
                std::move(state),
                {evaluation_cameras[0],
                 cross3dgs::CameraDownAxis(evaluation_cameras[0])},
                sh_degree);
    }
    return {g_cancel.load() ? 1 : 0, message.str(), ply_path.string()};
}

}  // namespace

// ---------------------------------------------------------------------------
// Bounded schedule failure probes.
//
// Each probe builds a minimal synthetic workload and enables exactly one
// candidate that the vendor evidence normally rejects, so a raw failure
// witness (integrity counters, hanging-pass label, or fault signature) can
// be captured at most once per device. The host harness bounds the run with
// a watchdog and force-stop; these probes are never part of training.

std::string ProbeScheduleIrJson(
        const cross3dgs::GaussianScheduleIR& ir) {
    std::ostringstream json;
    json << "{\"vendor_id\": " << ir.vendor_id
         << ", \"device_id\": " << ir.device_id
         << ", \"subgroup_size\": " << ir.subgroup_size
         << ", \"max_storage_buffer_range\": "
         << ir.max_storage_buffer_range
         << ", \"schedule_candidate_id\": \""
         << ir.schedule_candidate_id << "\""
         << ", \"policy_selected_candidate_id\": \""
         << ir.policy_selected_candidate_id << "\""
         << ", \"schedule_candidate_accepted\": "
         << (ir.schedule_candidate_accepted ? "true" : "false")
         << ", \"backward_work_assignment\": \""
         << ir.backward_work_assignment << "\""
         << ", \"backward_global_merge\": \""
         << ir.backward_global_merge << "\""
         << ", \"backward_tile_shape\": \""
         << ir.backward_tile_shape << "\""
         << ", \"tiled_backward\": "
         << (ir.tiled_backward ? "true" : "false")
         << ", \"per_splat_backward\": "
         << (ir.per_splat_backward ? "true" : "false")
         << ", \"split_sh_rest_storage\": "
         << (ir.split_sh_rest_storage ? "true" : "false")
         << ", \"atomic_free_exact_backward\": "
         << (ir.atomic_free_exact_backward ? "true" : "false")
         << ", \"native_float_atomic_add\": "
         << (ir.native_float_atomic_add ? "true" : "false")
         << ", \"fenced_execution\": "
         << (ir.fenced_execution ? "true" : "false")
         << ", \"streamed_target\": "
         << (ir.streamed_target ? "true" : "false")
         << ", \"portable_radix_sort\": "
         << (ir.portable_radix_sort ? "true" : "false")
         << ", \"indirect_pair_dispatch\": "
         << (ir.indirect_pair_dispatch ? "true" : "false") << "}";
    return json.str();
}

std::string RunScheduleProbe(const std::string& mode,
                             const std::filesystem::path& output_dir) {
    std::filesystem::create_directories(output_dir);
    if (mode == "conformance_selected_cpu_oracle" ||
        mode == "conformance_split_sh_rest_cpu_oracle") {
        const bool split_storage =
                mode == "conformance_split_sh_rest_cpu_oracle";
        NotifyLog(std::string("selected-schedule CPU double/Adam oracle") +
                  (split_storage ? " with split SH-rest" : ""));
        const bool passed =
                cross3dgs_probe::RunSelectedScheduleCpuOracle(
                        output_dir, split_storage);
        std::ofstream summary(output_dir / "probe_result.json");
        summary << "{\n  \"schema\": "
                   "\"cross_3dgs.selected_cpu_oracle_android.v1\",\n"
                << "  \"candidate\": \"runtime_policy_selected\",\n"
                << "  \"storage_layout\": \""
                << (split_storage ? "sh_rest_6_plus_6" : "device_default")
                << "\",\n"
                << "  \"reference\": \"CPU double finite-difference + "
                   "independent Adam\",\n"
                << "  \"pass\": " << (passed ? "true" : "false")
                << "\n}\n";
        if (!passed) {
            throw std::runtime_error(
                    "selected-schedule CPU oracle conformance failed");
        }
        return "conformance_passed";
    }
    if (mode == "conformance_per_splat_pair_reduce" ||
        mode == "conformance_per_splat_cas" ||
        mode == "conformance_per_splat_cas_production_footprint" ||
        mode ==
        "conformance_per_splat_cas_production_footprint_negative_control" ||
        mode == "conformance_pixel_subgroup_cas" ||
        mode == "conformance_exact_pair_gather" ||
        mode == "conformance_exact_pair_gather_split_sh_rest" ||
        mode == "conformance_per_splat_pair_reduce_split_sh_rest") {
        const std::string candidate =
                mode.substr(std::string("conformance_").size());
        NotifyLog("lowering conformance candidate=" + candidate);
        const bool passed =
                cross3dgs_probe::RunLoweringCandidateConformance(
                        output_dir, candidate);
        std::ofstream summary(output_dir / "probe_result.json");
        summary << "{\n  \"schema\": "
                   "\"cross_3dgs.lowering_conformance_android.v1\",\n"
                << "  \"candidate\": \"" << candidate << "\",\n"
                << "  \"reference\": \""
                << (candidate == "exact_pair_gather"
                            ? "pixel_stream_subgroup_scatter"
                            : "exact_pair_gather") << "\",\n"
                << "  \"pass\": " << (passed ? "true" : "false")
                << "\n}\n";
        // The split-axis negative control is expected to FAIL the gate; a
        // pass would mean the corruption went undetected, which is the
        // alarming outcome for that mode.
        if (candidate ==
            "per_splat_cas_production_footprint_negative_control") {
            return passed ? "negative_control_not_detected"
                          : "negative_control_failed_as_expected";
        }
        if (!passed) {
            throw std::runtime_error(
                    candidate + " conformance failed");
        }
        return "conformance_passed";
    }
    constexpr uint32_t kProbeWidth = 256;
    constexpr uint32_t kProbeHeight = 256;
    constexpr size_t kProbeGaussians = 4096;

    // Deterministic synthetic cloud in front of an identity camera.
    GaussianStateCpu state;
    uint32_t lcg = 42u;
    const auto uniform = [&lcg]() {
        lcg = lcg * 1664525u + 1013904223u;
        return static_cast<float>(lcg >> 8) /
               static_cast<float>(1u << 24);
    };
    for (size_t i = 0; i < kProbeGaussians; i++) {
        const float z = 2.5f + 3.0f * uniform();
        state.means.emplace_back((uniform() - 0.5f) * z * 0.9f,
                                 (uniform() - 0.5f) * z * 0.9f, z, 0.0f);
        state.scales_opacity.emplace_back(0.02f + 0.05f * uniform(),
                                          0.02f + 0.05f * uniform(),
                                          0.02f + 0.05f * uniform(),
                                          0.2f + 0.75f * uniform());
        state.rotations.emplace_back(
                glm::normalize(glm::vec4(0.5f + uniform(), uniform() - 0.5f,
                                         uniform() - 0.5f,
                                         uniform() - 0.5f)));
        state.sh_dc.emplace_back(uniform() - 0.5f, uniform() - 0.5f,
                                 uniform() - 0.5f, 0.0f);
    }
    state.sh_rest.assign(kProbeGaussians * 12, glm::vec4(0.0f));
    Camera camera;
    camera.width = kProbeWidth;
    camera.height = kProbeHeight;
    camera.fx = 200.0f;
    camera.fy = 200.0f;
    camera.cx = 128.0f;
    camera.cy = 128.0f;
    camera.model = cross3dgs::CameraModel::Pinhole;

    cross3dgs::GaussianForwardConfig config{
            .width = kProbeWidth,
            .height = kProbeHeight,
            .gaussian_capacity = kProbeGaussians,
            .tile_size = 16,
            .tile_bin_capacity = 512,
            .enable_backward = true};
    std::string expectation;
    if (mode == "subgroup_radix") {
        config.atomic_free_exact_backward = true;
        config.unsafe_force_subgroup_radix = true;
        expectation = "sorted-key integrity errors "
                      "(HC-RADIX-SUBGROUP-QCOM-01)";
    } else if (mode == "cas_scatter") {
        config.force_cas_scatter_backward = true;
        config.unsafe_allow_rejected_schedule = true;
        expectation = "CAS scatter backward hang/fault "
                      "(HC-ATOMIC-SCATTER-QCOM-01)";
    } else if (mode == "per_splat_scatter") {
        config.force_per_splat_backward = true;
        config.force_cas_scatter_backward = true;
        config.unsafe_allow_rejected_schedule = true;
        expectation = "validity-only PerSplat CAS scatter completion or "
                      "bounded hang/fault witness";
    } else if (mode == "batched_hang") {
        config.atomic_free_exact_backward = true;
        config.unsafe_force_batched_submission = true;
        expectation = "batched multi-pass submit hang "
                      "(HC-SUBMISSION-BATCHED-QCOM-01)";
    } else {
        throw std::invalid_argument("unknown probe mode " + mode);
    }

    NotifyLog("schedule_probe mode=" + mode + " expectation=" +
              expectation);
    GaussianForwardProgram program(config);
    const auto schedule = program.schedule();
    NotifyLog("schedule_probe schedule=" + ProbeScheduleIrJson(schedule));
    program.uploadState(state);
    FloatImage target;
    target.width = kProbeWidth;
    target.height = kProbeHeight;
    target.pixels.assign(
            static_cast<size_t>(kProbeWidth) * kProbeHeight * 4, 0.0f);
    program.uploadTarget(target);
    program.setCamera(camera);

    std::ostringstream results;
    results << std::setprecision(9);
    // Persist the pre-execution record first so a wedged process still
    // leaves the schedule and expectation on disk.
    {
        std::ofstream partial(output_dir / "probe_result.json");
        partial << "{\n  \"schema\": \"cross_3dgs.schedule_probe.v1\",\n"
                << "  \"mode\": \"" << mode << "\",\n"
                << "  \"expectation\": \"" << expectation << "\",\n"
                << "  \"schedule_ir\": " << ProbeScheduleIrJson(schedule)
                << ",\n  \"status\": \"launched\"\n}\n";
    }

    std::string status = "completed";
    if (mode == "cas_scatter" || mode == "per_splat_scatter" ||
        mode == "batched_hang") {
        // Bounded per-pass diagnostic first: it isolates a hanging pass
        // with its own fence and a 3-second timeout.
        const std::string hanging =
                program.firstHangingPass(3'000'000'000ull);
        results << "  \"first_hanging_pass\": \"" << hanging << "\",\n";
        NotifyLog("schedule_probe first_hanging_pass=" +
                  (hanging.empty() ? std::string("none") : hanging));
        if (!hanging.empty()) {
            status = "hanging_pass_detected";
        }
    }
    if (status == "completed") {
        // One full execute: on the batched probe this is the exact failing
        // submit shape; the host watchdog bounds it and captures logcat if
        // this call never returns.
        NotifyLog("schedule_probe executing mode=" + mode);
        program.execute();
        const float loss = program.downloadMeanL1();
        const auto integrity = program.debugPairIntegrity();
        const auto stats = program.stats();
        results << "  \"mean_l1\": " << loss << ",\n"
                << "  \"loss_finite\": "
                << (std::isfinite(loss) ? "true" : "false") << ",\n"
                << "  \"pair_count\": " << integrity.x << ",\n"
                << "  \"key_order_errors\": " << integrity.y << ",\n"
                << "  \"key_value_errors\": " << integrity.z << ",\n"
                << "  \"tile_membership_errors\": " << integrity.w << ",\n"
                << "  \"overflow_count\": " << stats.overflow_count
                << ",\n";
        NotifyLog("schedule_probe integrity pairs=" +
                  std::to_string(integrity.x) + " order_errors=" +
                  std::to_string(integrity.y) + " key_value_errors=" +
                  std::to_string(integrity.z) + " tile_errors=" +
                  std::to_string(integrity.w) + " loss=" +
                  std::to_string(loss));
        if (mode == "subgroup_radix" &&
            (integrity.y != 0 || integrity.z != 0 || integrity.w != 0)) {
            status = "integrity_errors_detected";
        }
    }

    std::ofstream output(output_dir / "probe_result.json");
    output << "{\n  \"schema\": \"cross_3dgs.schedule_probe.v1\",\n"
           << "  \"mode\": \"" << mode << "\",\n"
           << "  \"expectation\": \"" << expectation << "\",\n"
           << "  \"schedule_ir\": " << ProbeScheduleIrJson(schedule)
           << ",\n" << results.str()
           << "  \"status\": \"" << status << "\"\n}\n";
    NotifyLog("schedule_probe result mode=" + mode + " status=" + status);
    return status;
}

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
    g_vm = vm;
    // Route the default spdlog logger to logcat under the tag the
    // host-side monitoring greps for. The property
    // debug.cross3dgs.loglevel selects the level (default info).
    auto logger = std::make_shared<spdlog::logger>(
            "cross3dgs",
            std::make_shared<spdlog::sinks::android_sink_mt>("cross3dgs"));
    {
        char level[PROP_VALUE_MAX] = {0};
        __system_property_get("debug.cross3dgs.loglevel", level);
        logger->set_level(level[0] != '\0'
                                  ? spdlog::level::from_str(level)
                                  : spdlog::level::info);
    }
    spdlog::set_default_logger(std::move(logger));
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNI_OnUnload(JavaVM*, void*) {
    g_cancel = true;
    if (g_thread.joinable()) g_thread.join();
    bool detach = false;
    JNIEnv* env = GetEnv(detach);
    if (env != nullptr) {
        std::lock_guard<std::mutex> lock(g_listener_mutex);
        if (g_listener != nullptr) {
            env->DeleteGlobalRef(g_listener);
            g_listener = nullptr;
        }
    }
    if (detach) g_vm->DetachCurrentThread();
    g_vm = nullptr;
}

JNIEXPORT jstring JNICALL
Java_org_unclearness_cross3dgs_NativeBridge_queryDeviceCaps(
        JNIEnv* env, jclass) {
    const auto json = cross3dgs_android::QueryDeviceCapsJson();
    return env->NewStringUTF(json.c_str());
}

JNIEXPORT jboolean JNICALL
Java_org_unclearness_cross3dgs_NativeBridge_startTraining(
        JNIEnv* env, jclass, jstring scene_dir, jstring output_dir,
        jstring image_dir, jstring sparse_dir,
        jint iterations, jint view_limit, jint progress_every,
        jint stochastic_samples, jfloat ssim_weight,
        jstring init_checkpoint, jint sh_degree, jstring density_strategy,
        jint gaussian_capacity, jstring view_order, jstring adam_moments,
        jboolean full_image_ssim, jfloat mcmc_clone_offset,
        jfloat mcmc_gradient_sampling,
        jboolean force_atomic_free_exact,
        jstring backward_candidate) {
    if (iterations <= 0 || view_limit < 0 || progress_every <= 0 ||
        !(stochastic_samples == 0 || stochastic_samples == 1 ||
          stochastic_samples == 2 || stochastic_samples == 4 ||
          stochastic_samples == 8) ||
        !(ssim_weight >= 0.0f && ssim_weight <= 1.0f) ||
        sh_degree < 0 || sh_degree > 3 ||
        gaussian_capacity < 0 ||
        !(mcmc_clone_offset >= 0.0f) ||
        !(mcmc_gradient_sampling >= 0.0f) ||
        g_running.exchange(true)) {
        return JNI_FALSE;
    }
    if (g_thread.joinable()) g_thread.join();
    const auto scene = FromJString(env, scene_dir);
    const auto output = FromJString(env, output_dir);
    const auto images = FromJString(env, image_dir);
    const auto sparse = FromJString(env, sparse_dir);
    const auto checkpoint = FromJString(env, init_checkpoint);
    const auto density = FromJString(env, density_strategy);
    const auto order = FromJString(env, view_order);
    const auto moments = FromJString(env, adam_moments);
    const auto candidate = FromJString(env, backward_candidate);
    const bool full_image_ssim_enabled = full_image_ssim == JNI_TRUE;
    const float clone_offset = static_cast<float>(mcmc_clone_offset);
    const float gradient_sampling =
            static_cast<float>(mcmc_gradient_sampling);
    const bool force_atomic_free_exact_enabled =
            force_atomic_free_exact == JNI_TRUE;
    if (images.empty() || sparse.empty() ||
        (density != "fixed" && density != "mcmc")) {
        g_running = false;
        return JNI_FALSE;
    }
    if ((order != "cyclic" && order != "random") ||
        (moments != "bf16" && moments != "fp32")) {
        g_running = false;
        return JNI_FALSE;
    }
    if (candidate != "auto" && candidate != "exact_pair_gather" &&
        candidate != "pixel_subgroup_cas" &&
        candidate != "per_splat_cas" &&
        candidate != "per_splat_pair_reduce") {
        g_running = false;
        return JNI_FALSE;
    }
    {
        std::lock_guard<std::mutex> lock(g_results_mutex);
        g_results = {};
    }
    g_cancel = false;
    g_thread = std::thread([scene, output, images, sparse,
                            iterations, view_limit,
                            progress_every, stochastic_samples, ssim_weight,
                            checkpoint, sh_degree, density,
                            gaussian_capacity, order, moments,
                            full_image_ssim_enabled, clone_offset,
                            gradient_sampling, force_atomic_free_exact_enabled,
                            candidate]() {
        RunResult result;
        try {
            result = RunTraining(
                    scene, output, images, sparse,
                    static_cast<uint32_t>(iterations),
                    static_cast<uint32_t>(view_limit),
                    static_cast<uint32_t>(progress_every),
                    static_cast<uint32_t>(stochastic_samples),
                    static_cast<float>(ssim_weight), checkpoint,
                    static_cast<uint32_t>(sh_degree), density,
                    static_cast<size_t>(gaussian_capacity),
                    order, moments,
                    full_image_ssim_enabled, clone_offset, gradient_sampling,
                    force_atomic_free_exact_enabled, candidate);
        } catch (const std::exception& error) {
            result.exit_code = 2;
            result.message =
                    std::string("Optimization failed: ") + error.what();
            NotifyLog(result.message);
        } catch (...) {
            result.exit_code = 2;
            result.message = "Optimization failed with an unknown error";
            NotifyLog(result.message);
        }
        g_running = false;
        NotifyFinished(result.exit_code, result.message, result.ply_path);
    });
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_org_unclearness_cross3dgs_NativeBridge_requestStop(
        JNIEnv*, jclass) {
    g_cancel = true;
}

JNIEXPORT jboolean JNICALL
Java_org_unclearness_cross3dgs_NativeBridge_setTrainingViewSeed(
        JNIEnv*, jclass, jint value) {
    // Rejected while a run is in flight so the recorded seed can never
    // disagree with the one the sampler was built with.
    if (value < 0 || g_running.load()) return JNI_FALSE;
    g_training_view_seed.store(static_cast<uint32_t>(value));
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_org_unclearness_cross3dgs_NativeBridge_setInitSeed(
        JNIEnv*, jclass, jint value) {
    if (value < 0 || g_running.load()) return JNI_FALSE;
    g_init_seed.store(static_cast<uint32_t>(value));
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_org_unclearness_cross3dgs_NativeBridge_setTrainingPreviewPanels(
        JNIEnv*, jclass, jboolean enabled) {
    // Rejected while a run is in flight so the recorded mode is the one shown.
    if (g_running.load()) return JNI_FALSE;
    g_training_preview_panels.store(enabled == JNI_TRUE);
    return JNI_TRUE;
}

// {image width, image height, then x, y, width, height per panel}; empty until
// a run with panels has its image size.
JNIEXPORT jintArray JNICALL
Java_org_unclearness_cross3dgs_NativeBridge_trainingPreviewLayout(JNIEnv* env,
                                                                  jclass) {
    cross3dgs::TrainingPreviewLayout layout;
    {
        std::lock_guard<std::mutex> lock(g_training_preview_mutex);
        layout = g_training_preview_layout;
    }
    if (layout.width == 0) return env->NewIntArray(0);
    std::vector<jint> values = {static_cast<jint>(layout.width),
                                static_cast<jint>(layout.height)};
    for (const auto& panel : layout.panels) {
        values.insert(values.end(), {static_cast<jint>(panel.x),
                                     static_cast<jint>(panel.y),
                                     static_cast<jint>(panel.width),
                                     static_cast<jint>(panel.height)});
    }
    const auto size = static_cast<jsize>(values.size());
    jintArray result = env->NewIntArray(size);
    if (result != nullptr) {
        env->SetIntArrayRegion(result, 0, size, values.data());
    }
    return result;
}

JNIEXPORT jobjectArray JNICALL
Java_org_unclearness_cross3dgs_NativeBridge_trainingPreviewLabels(JNIEnv* env,
                                                                  jclass) {
    const auto labels = cross3dgs::TrainingPreviewLabels();
    jobjectArray result = env->NewObjectArray(
            static_cast<jsize>(labels.size()),
            env->FindClass("java/lang/String"), nullptr);
    if (result == nullptr) return nullptr;
    for (size_t index = 0; index < labels.size(); index++) {
        env->SetObjectArrayElement(result, static_cast<jsize>(index),
                                   env->NewStringUTF(labels[index].c_str()));
    }
    return result;
}

JNIEXPORT jboolean JNICALL
Java_org_unclearness_cross3dgs_NativeBridge_setTileBinCapacity(
        JNIEnv*, jclass, jint value) {
    // Rejected while a run is in flight: the trainer, the held-out evaluation
    // renderer and the viewer must all size their pair buffers from the same
    // value. The upper bound matches the core's own validation.
    if (value <= 0 || value > 8192 || g_running.load()) return JNI_FALSE;
    g_tile_bin_capacity.store(static_cast<uint32_t>(value));
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_org_unclearness_cross3dgs_NativeBridge_runScheduleProbe(
        JNIEnv* env, jclass, jstring probe_mode, jstring output_dir) {
    if (g_running.exchange(true)) return JNI_FALSE;
    if (g_thread.joinable()) g_thread.join();
    const auto mode = FromJString(env, probe_mode);
    const auto output = FromJString(env, output_dir);
    g_thread = std::thread([mode, output]() {
        RunResult result;
        try {
            const auto status = RunScheduleProbe(mode, output);
            result.exit_code = 0;
            result.message = "schedule probe " + mode + ": " + status;
        } catch (const std::exception& error) {
            result.exit_code = 2;
            result.message = "schedule probe " + mode +
                             " raised: " + error.what();
            NotifyLog(result.message);
            NotifyLog("schedule_probe result mode=" + mode +
                      " status=exception");
        }
        g_running = false;
        NotifyFinished(result.exit_code, result.message, "");
    });
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_org_unclearness_cross3dgs_NativeBridge_isRunning(
        JNIEnv*, jclass) {
    return g_running.load() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_org_unclearness_cross3dgs_NativeBridge_setSurface(
        JNIEnv* env, jclass, jobject surface) {
    ANativeWindow* window =
            surface != nullptr
                    ? ANativeWindow_fromSurface(env, surface)
                    : nullptr;
    cross3dgs_android::SurfaceRenderer::instance().setSurface(window);
}

JNIEXPORT jboolean JNICALL
Java_org_unclearness_cross3dgs_NativeBridge_selectResult(
        JNIEnv*, jclass, jint view_index, jboolean target) {
    if (view_index < 0 || view_index >= 8) return JNI_FALSE;
    std::filesystem::path path;
    {
        std::lock_guard<std::mutex> lock(g_results_mutex);
        if (!g_results.ready) return JNI_FALSE;
        path = target ? g_results.targets[static_cast<size_t>(view_index)]
                      : g_results.renders[static_cast<size_t>(view_index)];
    }
    try {
        cross3dgs_android::SurfaceRenderer::instance().show(
                cross3dgs::LoadImageRgbaF32(path));
        return JNI_TRUE;
    } catch (const std::exception& error) {
        NotifyLog(std::string("Failed to load result: ") + error.what());
        return JNI_FALSE;
    }
}

// Shows an RGBA8 frame (the SfM live preview) through the surface blit.
JNIEXPORT jboolean JNICALL
Java_org_unclearness_cross3dgs_NativeBridge_showRgba(JNIEnv* env, jclass,
                                                     jbyteArray rgba,
                                                     jint width, jint height) {
    if (rgba == nullptr || width <= 0 || height <= 0) return JNI_FALSE;
    const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (static_cast<size_t>(env->GetArrayLength(rgba)) != count * 4) {
        return JNI_FALSE;
    }
    std::vector<jbyte> bytes(count * 4);
    env->GetByteArrayRegion(rgba, 0, static_cast<jsize>(bytes.size()), bytes.data());
    FloatImage image;
    image.width = static_cast<uint32_t>(width);
    image.height = static_cast<uint32_t>(height);
    image.channels = 4;
    image.pixels.resize(bytes.size());
    for (size_t i = 0; i < bytes.size(); i++) {
        image.pixels[i] = static_cast<uint8_t>(bytes[i]) * (1.0f / 255.0f);
    }
    cross3dgs_android::SurfaceRenderer::instance().show(image);
    return JNI_TRUE;
}

// {min, max} pitch of the prepared viewer in radians (contains 0).
JNIEXPORT jfloatArray JNICALL
Java_org_unclearness_cross3dgs_NativeBridge_orbitPitchRange(JNIEnv* env,
                                                            jclass) {
    jfloat range[2];
    {
        std::lock_guard<std::mutex> lock(g_viewer_mutex);
        range[0] = g_viewer.orbit.min_pitch;
        range[1] = g_viewer.orbit.max_pitch;
    }
    jfloatArray result = env->NewFloatArray(2);
    if (result != nullptr) env->SetFloatArrayRegion(result, 0, 2, range);
    return result;
}

JNIEXPORT jboolean JNICALL
Java_org_unclearness_cross3dgs_NativeBridge_renderOrbit(
        JNIEnv*, jclass, jfloat yaw, jfloat pitch, jfloat zoom,
        jfloat pan_x, jfloat pan_y, jint width, jint height) {
    if (width <= 0 || height <= 0 || g_running.load()) return JNI_FALSE;
    try {
        std::lock_guard<std::mutex> lock(g_viewer_mutex);
        if (g_viewer.gaussians.size() == 0) return JNI_FALSE;
        const Camera camera = cross3dgs::OrbitCamera(
                g_viewer.orbit, yaw, pitch, zoom, pan_x, pan_y,
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height));
        if (g_viewer.renderer == nullptr ||
            g_viewer.render_width != camera.width ||
            g_viewer.render_height != camera.height) {
            g_viewer.renderer = std::make_unique<GaussianForwardProgram>(
                    cross3dgs::GaussianForwardConfig{
                            .width = camera.width,
                            .height = camera.height,
                            .gaussian_capacity = g_viewer.gaussians.size(),
                            .tile_size = 16,
                            .tile_bin_capacity = g_tile_bin_capacity.load(),
                            .sh_degree = g_viewer.sh_degree});
            g_viewer.renderer->uploadState(g_viewer.gaussians);
            g_viewer.render_width = camera.width;
            g_viewer.render_height = camera.height;
            NotifyLog("Viewer extent " + std::to_string(width) + "x" +
                      std::to_string(height) + " -> " +
                      std::to_string(camera.width) + "x" +
                      std::to_string(camera.height) +
                      " (aspect preserved)");
        }
        g_viewer.renderer->setCamera(camera);
        g_viewer.renderer->execute();
        cross3dgs_android::SurfaceRenderer::instance().show(
                g_viewer.renderer->downloadColor());
        return JNI_TRUE;
    } catch (const std::exception& error) {
        NotifyLog(std::string("Orbit render failed: ") + error.what());
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL
Java_org_unclearness_cross3dgs_NativeBridge_prepareViewer(
        JNIEnv* env, jclass, jstring scene_dir, jstring image_dir,
        jstring sparse_dir, jstring checkpoint_path,
        jstring reference_image_name, jstring frame_priors_path,
        jstring gravity_up_ply_path, jint sh_degree) {
    if (g_running.load()) return JNI_FALSE;
    try {
        const auto scene = cross3dgs::LoadColmapScene(
                FromJString(env, scene_dir), FromJString(env, image_dir),
                FromJString(env, sparse_dir));
        if (scene.cameras.empty()) return JNI_FALSE;
        auto state = cross3dgs::LoadGaussianCheckpoint(
                FromJString(env, checkpoint_path));
        if (state.size() == 0) return JNI_FALSE;
        const std::string requested_reference =
                FromJString(env, reference_image_name);
        const Camera* reference = &scene.cameras[0];
        if (!requested_reference.empty()) {
            const auto selected = std::find_if(
                    scene.cameras.begin(), scene.cameras.end(),
                    [&requested_reference](const Camera& camera) {
                        return camera.image_path.filename().string() ==
                               requested_reference;
                    });
            if (selected == scene.cameras.end()) {
                throw std::invalid_argument(
                        "viewer reference image is not in the scene: " +
                        requested_reference);
            }
            reference = &*selected;
            NotifyLog("Viewer reference image: " + requested_reference);
        }
        const auto orbit = cross3dgs::MakeOrbitReference(
                scene.cameras, *reference,
                FromJString(env, frame_priors_path), scene.dataset_transform,
                scene.points);
        NotifyLog(orbit.summary);
        const std::string gravity_up_ply =
                FromJString(env, gravity_up_ply_path);
        if (orbit.gravity_aligned && !gravity_up_ply.empty()) {
            cross3dgs::SaveGaussianPly(
                    gravity_up_ply,
                    cross3dgs::RotateGaussianState(
                            state, cross3dgs::RotationToDownPositiveY(
                                           orbit.yaw_axis)));
            NotifyLog("Saved " + gravity_up_ply);
        }
        PublishViewerState(std::move(state), orbit,
                           static_cast<uint32_t>(std::max(sh_degree, 0)));
        return JNI_TRUE;
    } catch (const std::exception& error) {
        NotifyLog(std::string("Viewer preparation failed: ") + error.what());
        return JNI_FALSE;
    }
}

JNIEXPORT void JNICALL
Java_org_unclearness_cross3dgs_NativeBridge_setListener(
        JNIEnv* env, jclass, jobject listener) {
    std::lock_guard<std::mutex> lock(g_listener_mutex);
    if (g_listener != nullptr) {
        env->DeleteGlobalRef(g_listener);
        g_listener = nullptr;
    }
    if (listener != nullptr) {
        g_listener = env->NewGlobalRef(listener);
    }
}

}  // extern "C"
