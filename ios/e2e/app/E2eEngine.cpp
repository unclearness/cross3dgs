// See E2eEngine.h. The training body follows RunTraining() in
// android/app/src/main/cpp/jni_bridge.cpp; the Android-only debug-property
// diagnostics and the schedule probes are not carried over.
#include "E2eEngine.h"

#include <spdlog/spdlog.h>

#include <algorithm>
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
#include <thread>

#include "cross3dgs/dataset.h"
#include "cross3dgs/gaussian_forward.h"
#include "cross3dgs/gaussian_state.h"
#include "cross3dgs/gpu/gpu_program.h"
#include "cross3dgs/orbit_view.h"

// android/app/src/main/cpp/device_caps.cpp (platform-neutral Vulkan probe;
// the namespace name is historical).
namespace cross3dgs_android {
std::string QueryDeviceCapsJson();
}

namespace cross3dgs_ios {
namespace {

using Clock = std::chrono::steady_clock;
using cross3dgs::Camera;
using cross3dgs::FloatImage;
using cross3dgs::GaussianForwardProgram;
using cross3dgs::GaussianStateCpu;

std::mutex g_callbacks_mutex;
EngineCallbacks g_callbacks;
std::atomic<bool> g_cancel{false};
std::atomic<bool> g_running{false};
// Same default as the Android bridge; the capture-to-view UI raises it to
// 2048 before every run (see E2eViewController kTileBinCapacity).
std::atomic<uint32_t> g_tile_bin_capacity{512};
constexpr uint32_t kTrainingViewSeed = 42;
std::thread g_thread;

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

void NotifyLog(const std::string& line) {
    spdlog::info("{}", line);
    std::function<void(const std::string&)> callback;
    {
        std::lock_guard<std::mutex> lock(g_callbacks_mutex);
        callback = g_callbacks.log;
    }
    if (callback) callback(line);
}

void NotifyProgress(int iteration, int total, float loss, double seconds) {
    std::function<void(int, int, float, double)> callback;
    {
        std::lock_guard<std::mutex> lock(g_callbacks_mutex);
        callback = g_callbacks.progress;
    }
    if (callback) callback(iteration, total, loss, seconds);
}

void NotifyFinished(int exit_code, const std::string& message,
                    const std::string& ply_path) {
    std::function<void(int, const std::string&, const std::string&)> callback;
    {
        std::lock_guard<std::mutex> lock(g_callbacks_mutex);
        callback = g_callbacks.finished;
    }
    if (callback) callback(exit_code, message, ply_path);
}

void Show(const FloatImage& image) {
    if (image.width == 0 || image.height == 0 || image.channels < 3) return;
    std::function<void(const RgbaImage&)> callback;
    {
        std::lock_guard<std::mutex> lock(g_callbacks_mutex);
        callback = g_callbacks.frame;
    }
    if (!callback) return;
    RgbaImage rgba;
    rgba.width = image.width;
    rgba.height = image.height;
    const size_t pixels = static_cast<size_t>(image.width) * image.height;
    rgba.pixels.resize(pixels * 4);
    const auto to_byte = [](float value) {
        return static_cast<uint8_t>(
                std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
    };
    for (size_t pixel = 0; pixel < pixels; pixel++) {
        const float* source = &image.pixels[pixel * image.channels];
        rgba.pixels[pixel * 4 + 0] = to_byte(source[0]);
        rgba.pixels[pixel * 4 + 1] = to_byte(source[1]);
        rgba.pixels[pixel * 4 + 2] = to_byte(source[2]);
        rgba.pixels[pixel * 4 + 3] = 255;
    }
    callback(rgba);
}

std::string ViewFilename(size_t view) {
    std::ostringstream name;
    name << "view_" << std::setfill('0') << std::setw(3) << view << ".png";
    return name.str();
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

RunResult RunTraining(const TrainingRequest& request) {
    const auto run_start = Clock::now();
    const std::filesystem::path scene_dir = request.scene_dir;
    const std::filesystem::path output_dir = request.output_dir;
    const uint32_t iterations = request.iterations;
    const uint32_t progress_every = std::max(request.progress_every, 1u);
    const float ssim_weight = request.ssim_weight;
    const uint32_t sh_degree = request.sh_degree;
    const bool enable_mcmc = request.density_strategy == "mcmc";

    NotifyLog("Loading COLMAP metadata and available training views");
    const auto scene = cross3dgs::LoadColmapScene(
            scene_dir, request.image_dir, request.sparse_dir);
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
    NotifyLog("Training views: " + std::to_string(cameras.size()) +
              ", validation views: " +
              std::to_string(validation_cameras.size()));

    auto state = cross3dgs::InitializeFromSparsePoints(scene);
    NotifyLog("Initializing Gaussians from " + std::to_string(state.size()) +
              " sparse points");
    const size_t initial_gaussian_count = state.size();
    const size_t resolved_gaussian_capacity =
            request.gaussian_capacity == 0 ? initial_gaussian_count
                                           : request.gaussian_capacity;
    if (resolved_gaussian_capacity < initial_gaussian_count) {
        throw std::invalid_argument(
                "Gaussian capacity " +
                std::to_string(resolved_gaussian_capacity) +
                " is below the sparse point count " +
                std::to_string(initial_gaussian_count));
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
              ", density: " + request.density_strategy +
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
    uint32_t completed = 0;
    float final_loss = 0.0f;
    cross3dgs::GaussianForwardStats training_stats;
    cross3dgs::GaussianScheduleIR training_schedule;
    double setup_pipeline_upload_seconds = 0.0;
    double optimization_loop_wall_seconds = 0.0;
    double progress_loss_readback_seconds = 0.0;
    double preview_readback_show_host_seconds = 0.0;
    double final_state_readback_seconds = 0.0;
    const auto training_start = Clock::now();
    {
        NotifyLog("Building Vulkan training pipelines");
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
                 .randomize_training_views = request.view_order == "random",
                 .training_view_seed = kTrainingViewSeed,
                 .sh_degree = sh_degree,
                 .means_lr = 0.00016f * scene_scale,
                 .means_lr_final = 0.0000016f * scene_scale,
                 .l1_weight = 1.0f - ssim_weight,
                 .ssim_weight = ssim_weight,
                 .full_image_ssim = request.full_image_ssim,
                 .enable_mcmc = enable_mcmc,
                 .mcmc_clone_offset = request.mcmc_clone_offset,
                 .mcmc_gradient_sampling = request.mcmc_gradient_sampling,
                 .scale_regularizer = enable_mcmc ? 0.01f : 0.0f,
                 .opacity_regularizer = enable_mcmc ? 0.01f : 0.0f,
                 .fp32_adam_moments = request.adam_moments == "fp32",
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
        trainer.uploadTrainingSet(cameras, [&](size_t view) {
            if (view % 16 == 0 || view + 1 == cameras.size()) {
                NotifyLog("Uploading targets " + std::to_string(view + 1) +
                          "/" + std::to_string(cameras.size()));
            }
            return cross3dgs::LoadImageRgbaF32(cameras[view].image_path);
        });
        setup_pipeline_upload_seconds = std::chrono::duration<double>(
                Clock::now() - pipeline_setup_start).count();
        {
            std::function<void(const cross3dgs::TrainingPreviewLayout&)> callback;
            {
                std::lock_guard<std::mutex> lock(g_callbacks_mutex);
                callback = g_callbacks.training_preview_layout;
            }
            if (callback) {
                callback(cross3dgs::MakeTrainingPreviewLayout(
                        cameras[0].width, cameras[0].height));
            }
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
                        "iteration " + std::to_string(iteration + 1) + ": " +
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
            const bool report = completed == 1 || completed == iterations ||
                                completed % progress_every == 0;
            if (report) {
                const auto loss_readback_start = Clock::now();
                final_loss = trainer.downloadMeanL1();
                progress_loss_readback_seconds +=
                        std::chrono::duration<double>(
                                Clock::now() - loss_readback_start).count();
                const auto preview_start = Clock::now();
                Show(cross3dgs::ComposeTrainingPreview(
                        trainer.downloadTrainingPreview()));
                preview_readback_show_host_seconds +=
                        std::chrono::duration<double>(
                                Clock::now() - preview_start).count();
                const double seconds = std::chrono::duration<double>(
                        Clock::now() - training_start).count();
                loss_curve.push_back({completed, final_loss, seconds});
                NotifyProgress(static_cast<int>(completed),
                               static_cast<int>(iterations), final_loss,
                               seconds);
            }
        }
        optimization_loop_wall_seconds = std::chrono::duration<double>(
                Clock::now() - optimization_loop_start).count();
        training_stats = trainer.stats();
        training_schedule = trainer.schedule();
        const auto final_state_readback_start = Clock::now();
        state = trainer.downloadState();
        final_state_readback_seconds = std::chrono::duration<double>(
                Clock::now() - final_state_readback_start).count();
    }
    if (completed == 0) {
        throw std::runtime_error("optimization stopped before the first step");
    }
    const double training_seconds =
            std::chrono::duration<double>(Clock::now() - training_start)
                    .count();
    const size_t non_finite =
            CountNonFinite(state.means) + CountNonFinite(state.scales_opacity) +
            CountNonFinite(state.rotations) + CountNonFinite(state.sh_dc) +
            CountNonFinite(state.sh_rest);
    if (non_finite != 0) {
        throw std::runtime_error(
                "optimization produced " + std::to_string(non_finite) +
                " non-finite Gaussian parameters");
    }

    const auto artifact_start = Clock::now();
    const auto ply_path = output_dir / "splat.ply";
    cross3dgs::SaveGaussianPly(ply_path, state);
    NotifyLog("Saved " + ply_path.string());
    cross3dgs::SaveGaussianCheckpoint(output_dir / "checkpoint.d3gs", state);
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

    // Held-out views when present, training views otherwise; every render
    // and target is saved for host-side rescoring.
    const bool has_validation = !validation_cameras.empty();
    const auto& evaluation_cameras =
            has_validation ? validation_cameras : cameras;
    NotifyLog(std::string("Evaluating ") +
              std::to_string(evaluation_cameras.size()) +
              (has_validation ? " validation" : " training") + " views");
    const auto evaluation_start = Clock::now();
    double evaluation_artifact_io_seconds = 0.0;
    struct ViewMetric {
        std::string name;
        double l1 = 0.0;
        double mse = 0.0;
    };
    std::vector<ViewMetric> view_metrics;
    double absolute_error = 0.0;
    double squared_error = 0.0;
    size_t value_count = 0;
    double psnr_per_view_sum = 0.0;
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
            evaluator.setCamera(evaluation_cameras[view]);
            evaluator.execute();
            const auto rendered = evaluator.downloadColor();
            const auto target = cross3dgs::LoadImageRgbaF32(
                    evaluation_cameras[view].image_path);
            const auto filename = ViewFilename(view);
            const auto evaluation_artifact_start = Clock::now();
            cross3dgs::SaveImagePng(output_dir / "render" / filename,
                                    rendered);
            cross3dgs::SaveImagePng(output_dir / "target" / filename, target);
            evaluation_artifact_io_seconds += std::chrono::duration<double>(
                    Clock::now() - evaluation_artifact_start).count();
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
            view_metrics.push_back({evaluation_cameras[view].image_name,
                                    view_absolute / view_values,
                                    view_squared / view_values});
            psnr_per_view_sum +=
                    -10.0 * std::log10(view_squared / view_values);
            absolute_error += view_absolute;
            squared_error += view_squared;
            value_count += pixels * 3;
        }
    }
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
    const size_t warmup = std::min<size_t>(100, step_milliseconds.size() / 10);
    std::vector<double> steady(step_milliseconds.begin() + warmup,
                               step_milliseconds.end());
    std::sort(steady.begin(), steady.end());
    const auto percentile = [&](double fraction) {
        const double position = fraction * double(steady.size() - 1);
        const size_t lower = static_cast<size_t>(position);
        const size_t upper = std::min(lower + 1, steady.size() - 1);
        const double weight = position - double(lower);
        return steady[lower] * (1.0 - weight) + steady[upper] * weight;
    };
    const double run_wall_seconds =
            std::chrono::duration<double>(Clock::now() - run_start).count();
    const char* bool_text[] = {"false", "true"};

    std::ofstream benchmark(output_dir / "bench.json");
    if (!benchmark) throw std::runtime_error("failed to create bench.json");
    benchmark << "{\n"
              << "  \"schema_version\": 3,\n"
              << "  \"backend\": \"Cross3DGS Vulkan iOS\",\n"
              << "  \"scene\": \"" << scene_dir.filename().string() << "\",\n"
              << "  \"image_dir\": \"" << request.image_dir << "\",\n"
              << "  \"sparse_dir\": \"" << request.sparse_dir << "\",\n"
              << "  \"sampled_train_image_count\": " << cameras.size() << ",\n"
              << "  \"view_limit\": 0,\n"
              << "  \"validation_image_count\": " << validation_cameras.size()
              << ",\n"
              << "  \"evaluation_split\": \""
              << (has_validation ? "validation" : "train") << "\",\n"
              << "  \"init_source\": \"sparse_points\",\n"
              << "  \"scene_scale\": " << scene_scale << ",\n"
              << "  \"sh_degree\": " << sh_degree << ",\n"
              << "  \"l1_weight\": " << (1.0f - ssim_weight) << ",\n"
              << "  \"ssim_weight\": " << ssim_weight << ",\n"
              << "  \"full_image_ssim\": " << bool_text[request.full_image_ssim]
              << ",\n"
              << "  \"density_strategy\": \"" << request.density_strategy
              << "\",\n"
              << "  \"view_order\": \"" << request.view_order << "\",\n"
              << "  \"adam_moments\": \"" << request.adam_moments << "\",\n"
              << "  \"mcmc_clone_offset\": " << request.mcmc_clone_offset
              << ",\n"
              << "  \"mcmc_gradient_sampling\": "
              << request.mcmc_gradient_sampling << ",\n"
              << "  \"training_view_seed\": " << kTrainingViewSeed << ",\n"
              << "  \"alpha_cap\": 0.999,\n"
              << "  \"requested_gaussian_capacity\": "
              << resolved_gaussian_capacity << ",\n"
              << "  \"gaussian_capacity\": " << gaussian_capacity << ",\n"
              << "  \"initial_gaussian_count\": " << initial_gaussian_count
              << ",\n"
              << "  \"gaussian_count\": " << state.size() << ",\n"
              << "  \"requested_iterations\": " << iterations << ",\n"
              << "  \"completed_iterations\": " << completed << ",\n"
              << "  \"backward_mode\": \"exact_scatter\",\n"
              << "  \"requested_backward_candidate\": \"auto\",\n"
              << "  \"stochastic_samples\": 0,\n"
              << "  \"tile_bin_capacity\": " << g_tile_bin_capacity.load()
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
              << "  \"input_setup_seconds\": " << input_setup_seconds << ",\n"
              << "  \"setup_pipeline_upload_seconds\": "
              << setup_pipeline_upload_seconds << ",\n"
              << "  \"optimization_loop_wall_seconds\": "
              << optimization_loop_wall_seconds << ",\n"
              << "  \"step_host_sum_seconds\": " << step_host_sum_seconds
              << ",\n"
              << "  \"progress_loss_readback_seconds\": "
              << progress_loss_readback_seconds << ",\n"
              << "  \"preview_readback_show_host_seconds\": "
              << preview_readback_show_host_seconds << ",\n"
              << "  \"training_preview_panels\": true,\n"
              << "  \"preview_readback_show_host_semantics\": "
                 "\"downloadTrainingPreview_compose_rgba8_conversion_plus_"
                 "ui_dispatch_without_present_completion_guarantee\",\n"
              << "  \"final_state_readback_seconds\": "
              << final_state_readback_seconds << ",\n"
              << "  \"artifact_io_seconds\": " << artifact_io_seconds << ",\n"
              << "  \"evaluation_seconds\": " << evaluation_seconds << ",\n"
              << "  \"evaluation_artifact_io_seconds\": "
              << evaluation_artifact_io_seconds << ",\n"
              << "  \"evaluation_compute_readback_present_seconds\": "
              << evaluation_seconds - evaluation_artifact_io_seconds << ",\n"
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
              << "  \"evaluation_mean_view_psnr\": " << mean_view_psnr << ",\n"
              << "  \"overflow_count\": " << training_stats.overflow_count
              << ",\n"
              << "  \"max_tile_count\": " << training_stats.max_tile_count
              << ",\n"
              << "  \"tile_reference_count\": "
              << training_stats.tile_reference_count << ",\n"
              << "  \"pair_capacity\": " << training_stats.pair_capacity
              << ",\n"
              << "  \"state_non_finite\": " << non_finite << ",\n"
              << "  \"current_memory_bytes\": "
              << training_stats.current_memory_bytes << ",\n"
              << "  \"peak_memory_bytes\": "
              << training_stats.peak_memory_bytes << ",\n"
              << "  \"schedule_ir\": {\n"
              << "    \"vendor_id\": " << training_schedule.vendor_id << ",\n"
              << "    \"device_id\": " << training_schedule.device_id << ",\n"
              << "    \"subgroup_size\": " << training_schedule.subgroup_size
              << ",\n"
              << "    \"max_shared_memory\": "
              << training_schedule.max_shared_memory << ",\n"
              << "    \"max_storage_buffer_range\": "
              << training_schedule.max_storage_buffer_range << ",\n"
              << "    \"tile_size\": " << training_schedule.tile_size << ",\n"
              << "    \"command_batch_size\": "
              << training_schedule.command_batch_size << ",\n"
              << "    \"backward_work_assignment\": \""
              << training_schedule.backward_work_assignment << "\",\n"
              << "    \"backward_global_merge\": \""
              << training_schedule.backward_global_merge << "\",\n"
              << "    \"schedule_candidate_id\": \""
              << training_schedule.schedule_candidate_id << "\",\n"
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
              << "    \"subgroup_reordered_sh\": "
              << bool_text[training_schedule.subgroup_reordered_sh] << ",\n"
              << "    \"native_float_atomic_add\": "
              << bool_text[training_schedule.native_float_atomic_add] << ",\n"
              << "    \"byte_targets\": "
              << bool_text[training_schedule.byte_targets] << ",\n"
              << "    \"fenced_execution\": "
              << bool_text[training_schedule.fenced_execution] << "\n"
              << "  },\n"
              << "  \"stage_timings_ms\": {\n";
    for (size_t index = 0; index < training_stats.timings.size(); index++) {
        const auto& timing = training_stats.timings[index];
        benchmark << "    \"" << timing.label << "\": "
                  << timing.gpu_microseconds / 1000.0
                  << (index + 1 == training_stats.timings.size() ? "\n"
                                                                 : ",\n");
    }
    benchmark << "  },\n"
              << "  \"evaluation_views\": [\n";
    for (size_t index = 0; index < view_metrics.size(); index++) {
        const auto& metric = view_metrics[index];
        benchmark << "    {\"view\": " << index << ", \"image\": \""
                  << metric.name << "\", \"l1\": " << metric.l1
                  << ", \"mse\": " << metric.mse << "}"
                  << (index + 1 == view_metrics.size() ? "\n" : ",\n");
    }
    benchmark << "  ],\n"
              << "  \"loss_curve\": [\n";
    for (size_t index = 0; index < loss_curve.size(); index++) {
        const auto& point = loss_curve[index];
        benchmark << "    {\"iteration\": " << point.iteration
                  << ", \"l1\": " << point.loss
                  << ", \"elapsed_seconds\": " << point.elapsed_seconds << "}"
                  << (index + 1 == loss_curve.size() ? "\n" : ",\n");
    }
    benchmark << "  ]\n}\n";
    if (!benchmark) throw std::runtime_error("failed to write bench.json");

    std::ostringstream message;
    message << (g_cancel.load() ? "Stopped" : "Finished") << " " << completed
            << "/" << iterations << " in " << std::fixed
            << std::setprecision(2) << training_seconds << " s; L1 "
            << std::setprecision(6) << mean_l1 << ", PSNR "
            << std::setprecision(3) << psnr << " dB";
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

Engine& Engine::instance() {
    static Engine engine;
    return engine;
}

void Engine::setCallbacks(EngineCallbacks callbacks) {
    std::lock_guard<std::mutex> lock(g_callbacks_mutex);
    g_callbacks = std::move(callbacks);
}

bool Engine::setTileBinCapacity(uint32_t value) {
    // The trainer, evaluator and viewer must size their pair buffers from the
    // same value, so it cannot change mid-run. Upper bound matches the core.
    if (value == 0 || value > 8192 || g_running.load()) return false;
    g_tile_bin_capacity.store(value);
    return true;
}

bool Engine::startTraining(const TrainingRequest& request) {
    if (request.iterations == 0 || request.sh_degree > 3 ||
        !(request.ssim_weight >= 0.0f && request.ssim_weight <= 1.0f) ||
        (request.density_strategy != "fixed" &&
         request.density_strategy != "mcmc") ||
        (request.view_order != "cyclic" && request.view_order != "random") ||
        (request.adam_moments != "bf16" && request.adam_moments != "fp32") ||
        request.image_dir.empty() || request.sparse_dir.empty()) {
        return false;
    }
    if (g_running.exchange(true)) return false;
    if (g_thread.joinable()) g_thread.join();
    g_cancel = false;
    g_thread = std::thread([request]() {
        RunResult result;
        try {
            result = RunTraining(request);
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
    return true;
}

void Engine::requestStop() { g_cancel = true; }

bool Engine::isRunning() const { return g_running.load(); }

bool Engine::prepareViewer(const std::string& scene_dir,
                           const std::string& image_dir,
                           const std::string& sparse_dir,
                           const std::string& checkpoint_path,
                           const std::string& reference_image_name,
                           const std::string& frame_priors_path,
                           const std::string& gravity_up_ply_path,
                           uint32_t sh_degree) {
    if (g_running.load()) return false;
    try {
        const auto scene =
                cross3dgs::LoadColmapScene(scene_dir, image_dir, sparse_dir);
        if (scene.cameras.empty()) return false;
        auto state = cross3dgs::LoadGaussianCheckpoint(checkpoint_path);
        if (state.size() == 0) return false;
        const Camera* reference = &scene.cameras[0];
        if (!reference_image_name.empty()) {
            const auto selected = std::find_if(
                    scene.cameras.begin(), scene.cameras.end(),
                    [&](const Camera& camera) {
                        return camera.image_path.filename().string() ==
                               reference_image_name;
                    });
            if (selected == scene.cameras.end()) {
                throw std::invalid_argument(
                        "viewer reference image is not in the scene: " +
                        reference_image_name);
            }
            reference = &*selected;
            NotifyLog("Viewer reference image: " + reference_image_name);
        }
        const auto orbit = cross3dgs::MakeOrbitReference(
                scene.cameras, *reference, frame_priors_path,
                scene.dataset_transform, scene.points);
        NotifyLog(orbit.summary);
        if (orbit.gravity_aligned && !gravity_up_ply_path.empty()) {
            cross3dgs::SaveGaussianPly(
                    gravity_up_ply_path,
                    cross3dgs::RotateGaussianState(
                            state, cross3dgs::RotationToDownPositiveY(
                                           orbit.yaw_axis)));
            NotifyLog("Saved " + gravity_up_ply_path);
        }
        PublishViewerState(std::move(state), orbit, sh_degree);
        return true;
    } catch (const std::exception& error) {
        NotifyLog(std::string("Viewer preparation failed: ") + error.what());
        return false;
    }
}

void Engine::orbitPitchRange(float* min_pitch, float* max_pitch) const {
    std::lock_guard<std::mutex> lock(g_viewer_mutex);
    *min_pitch = g_viewer.orbit.min_pitch;
    *max_pitch = g_viewer.orbit.max_pitch;
}

bool Engine::renderOrbit(float yaw, float pitch, float zoom, float pan_x,
                         float pan_y, uint32_t width, uint32_t height) {
    if (width == 0 || height == 0 || g_running.load()) return false;
    try {
        std::lock_guard<std::mutex> lock(g_viewer_mutex);
        if (g_viewer.gaussians.size() == 0) return false;
        const Camera camera = cross3dgs::OrbitCamera(
                g_viewer.orbit, yaw, pitch, zoom, pan_x, pan_y, width, height);
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
                      std::to_string(camera.height) + " (aspect preserved)");
        }
        g_viewer.renderer->setCamera(camera);
        g_viewer.renderer->execute();
        Show(g_viewer.renderer->downloadColor());
        return true;
    } catch (const std::exception& error) {
        NotifyLog(std::string("Orbit render failed: ") + error.what());
        return false;
    }
}

std::string Engine::deviceCapsJson() {
    return cross3dgs_android::QueryDeviceCapsJson();
}

}  // namespace cross3dgs_ios
