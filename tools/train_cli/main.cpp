#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>

#include "cross3dgs/dataset.h"
#include "cross3dgs/gaussian_forward.h"
#include "cross3dgs/gaussian_state.h"
#include "cross3dgs/schedule_ledger.h"
#if CROSS_3DGS_VIEWER
#include "vk_viewer.h"
#endif

namespace {

#if CROSS_3DGS_VIEWER
cross3dgs_viewer::ViewerImage SideBySide(
        const cross3dgs::FloatImage& target,
        const cross3dgs::FloatImage& rendered) {
    if (target.width != rendered.width || target.height != rendered.height ||
        target.channels < 3 || rendered.channels < 3) {
        throw std::invalid_argument("viewer images do not match");
    }
    constexpr uint32_t divider = 4;
    cross3dgs_viewer::ViewerImage image(target.width * 2 + divider,
                                        target.height, 3, 0.0f);
    const auto copy = [&](const cross3dgs::FloatImage& source,
                          uint32_t offset_x) {
        for (uint32_t y = 0; y < source.height; y++) {
            for (uint32_t x = 0; x < source.width; x++) {
                const size_t source_index =
                        (static_cast<size_t>(y) * source.width + x) *
                        source.channels;
                for (uint32_t channel = 0; channel < 3; channel++) {
                    image.at(x + offset_x, y, channel) =
                            source.pixels[source_index + channel];
                }
            }
        }
    };
    copy(target, 0);
    copy(rendered, target.width + divider);
    for (uint32_t y = 0; y < image.height; y++) {
        for (uint32_t x = target.width; x < target.width + divider; x++) {
            image.at(x, y, 0) = 1.0f;
            image.at(x, y, 1) = 1.0f;
            image.at(x, y, 2) = 1.0f;
        }
    }
    return image;
}
#endif  // CROSS_3DGS_VIEWER

}  // namespace

// The iOS bench app cannot spawn a CLI process, so it compiles this file with
// CROSS_3DGS_TRAIN_MAIN_AS_LIBRARY and calls the renamed entry point with a
// synthesized argv. Desktop builds are unchanged (the macro is never defined
// there).
#if defined(CROSS_3DGS_TRAIN_MAIN_AS_LIBRARY)
int cross3dgs_train_main(int argc, char** argv) try {
#else
int main(int argc, char** argv) try {
#endif
    const auto process_start = std::chrono::steady_clock::now();
    // e.g. SPDLOG_LEVEL=debug prints the selected schedule and per-checkpoint
    // progress; the machine-parsed result lines stay on plain stdout.
    spdlog::cfg::load_env_levels();
    std::filesystem::path dataset_root = "data/360_v2";
    std::string scene_name = "bicycle";
    std::string image_dir = "images_4";
    std::string sparse_dir = "sparse/0";
    std::filesystem::path render_initial;
    std::filesystem::path benchmark_json;
    std::filesystem::path step_times_csv;
    // Raw tracked-allocator samples (trainer GaussianForwardProgram scope,
    // bytes). Off by default: the measured desktop path is unchanged unless
    // the flag is passed.
    std::filesystem::path memory_samples_csv;
    std::filesystem::path evaluation_image_dir;
    std::filesystem::path output_ply;
    std::filesystem::path output_checkpoint;
    std::filesystem::path input_checkpoint;
    std::filesystem::path quality_curve_dir;
    std::filesystem::path candidate_ledger_json;
    uint64_t max_storage_buffer_range = 0;
    std::string caps_evidence_key;
    size_t camera_index = 0;
    bool backward_smoke = false;
    uint32_t train_iterations = 0;
    uint32_t max_views = 0;
    uint32_t sh_degree = 0;
    uint32_t warmup_iterations = 20;
    uint32_t quality_eval_every = 0;
    uint32_t stochastic_backward_samples = 0;
    uint32_t init_seed = 42;
    std::string view_order = "cyclic";
    std::string adam_moments = "bf16";
    bool pass_profiling = false;
    // Decode train targets one view at a time during uploadTrainingSet
    // instead of holding every view as host floats (~2.4 GB for bicycle
    // images_4). The engine keeps its own packed copy in both target modes,
    // so the eager cache only serves the viewer and a validation-less
    // evaluation; both combinations are rejected below. Off by default: the
    // measured desktop path is unchanged unless the flag is passed.
    bool stream_train_targets = false;
    bool atomic_free_backward = false;
    bool force_fenced_execution = false;
    bool force_pixel_subgroup_backward = false;
    bool force_per_splat_backward = false;
    bool force_tiled_backward = false;
    bool force_cas_scatter_backward = false;
    bool force_streamed_target = false;
    bool force_portable_radix_sort = false;
    bool force_split_projection_optimizer = false;
    bool force_split_sh_rest_storage = false;
    bool force_direct_pair_dispatch = false;
    bool viewer = false;
    uint32_t viewer_every = 100;
    uint32_t viewer_width = 1280;
    size_t gaussian_capacity = 0;
    size_t requested_gaussian_capacity = 0;
    float ssim_weight = 0.0f;
    bool full_image_ssim = false;
    bool mip3d_filter = false;
    bool full_depth_sort = false;
    float mcmc_noise_lr = 500000.0f;
    uint32_t mcmc_refine_start = 500;
    uint32_t mcmc_refine_stop = 25000;
    uint32_t mcmc_refine_every = 100;
    float mcmc_grow_factor = 1.05f;
    float mcmc_clone_offset = 0.0f;
    float mcmc_opacity_decay = 0.0f;
    uint32_t mcmc_opacity_decay_stop = 0;
    float mcmc_gradient_sampling = 0.0f;
    float mcmc_gradient_reference = 0.0002f;
    float mcmc_scale_regularizer = 0.01f;
    float mcmc_opacity_regularizer = 0.01f;
    std::string density_strategy = "fixed";
    float igs_edge_threshold = 0.1f;
    float igs_grad_threshold = 0.0002f;
    uint32_t igs_edge_warmup_steps = 2;
    float igs_prune_fraction = 0.0f;
    bool igs_growth_control = true;
    bool igs_multistep_update = false;
    for (int i = 1; i < argc; i++) {
        const std::string argument = argv[i];
        const auto value = [&]() -> std::string {
            if (++i >= argc) {
                throw std::invalid_argument("missing value for " + argument);
            }
            return argv[i];
        };
        if (argument == "--dataset-root") dataset_root = value();
        else if (argument == "--scene") scene_name = value();
        else if (argument == "--image-dir") image_dir = value();
        else if (argument == "--sparse-dir") sparse_dir = value();
        else if (argument == "--render-initial") render_initial = value();
        else if (argument == "--benchmark-json") benchmark_json = value();
        else if (argument == "--step-times-csv") step_times_csv = value();
        else if (argument == "--memory-samples-csv")
            memory_samples_csv = value();
        else if (argument == "--evaluation-image-dir")
            evaluation_image_dir = value();
        else if (argument == "--output-ply") output_ply = value();
        else if (argument == "--output-checkpoint") output_checkpoint = value();
        else if (argument == "--input-checkpoint") input_checkpoint = value();
        else if (argument == "--quality-curve-dir") quality_curve_dir = value();
        else if (argument == "--candidate-ledger-json")
            candidate_ledger_json = value();
        else if (argument == "--max-storage-buffer-range")
            max_storage_buffer_range = std::stoull(value(), nullptr, 0);
        else if (argument == "--caps-evidence-key")
            caps_evidence_key = value();
        else if (argument == "--camera-index")
            camera_index = static_cast<size_t>(std::stoull(value()));
        else if (argument == "--backward-smoke") backward_smoke = true;
        else if (argument == "--stream-train-targets")
            stream_train_targets = true;
        else if (argument == "--train-iterations")
            train_iterations = static_cast<uint32_t>(std::stoul(value()));
        else if (argument == "--max-views")
            max_views = static_cast<uint32_t>(std::stoul(value()));
        else if (argument == "--sh-degree")
            sh_degree = static_cast<uint32_t>(std::stoul(value()));
        else if (argument == "--warmup")
            warmup_iterations = static_cast<uint32_t>(std::stoul(value()));
        else if (argument == "--quality-eval-every")
            quality_eval_every = static_cast<uint32_t>(std::stoul(value()));
        else if (argument == "--stochastic-backward-samples")
            stochastic_backward_samples =
                    static_cast<uint32_t>(std::stoul(value()));
        else if (argument == "--init-seed")
            init_seed = static_cast<uint32_t>(std::stoul(value()));
        else if (argument == "--view-order") view_order = value();
        else if (argument == "--adam-moments") adam_moments = value();
        else if (argument == "--atomic-free-backward")
            atomic_free_backward = true;
        else if (argument == "--force-fenced-execution")
            force_fenced_execution = true;
        else if (argument == "--force-pixel-subgroup-backward")
            force_pixel_subgroup_backward = true;
        else if (argument == "--force-per-splat-backward")
            force_per_splat_backward = true;
        else if (argument == "--force-tiled-backward")
            force_tiled_backward = true;
        else if (argument == "--force-cas-scatter-backward")
            force_cas_scatter_backward = true;
        else if (argument == "--force-streamed-target")
            force_streamed_target = true;
        else if (argument == "--force-portable-radix-sort")
            force_portable_radix_sort = true;
        else if (argument == "--force-split-projection-optimizer")
            force_split_projection_optimizer = true;
        else if (argument == "--force-split-sh-rest-storage")
            force_split_sh_rest_storage = true;
        else if (argument == "--force-direct-pair-dispatch")
            force_direct_pair_dispatch = true;
        else if (argument == "--pass-profiling") pass_profiling = true;
        else if (argument == "--viewer") viewer = true;
        else if (argument == "--viewer-every")
            viewer_every = static_cast<uint32_t>(std::stoul(value()));
        else if (argument == "--viewer-width")
            viewer_width = static_cast<uint32_t>(std::stoul(value()));
        else if (argument == "--gaussian-capacity")
            gaussian_capacity = static_cast<size_t>(std::stoull(value()));
        else if (argument == "--ssim-weight") ssim_weight = std::stof(value());
        else if (argument == "--full-image-ssim") full_image_ssim = true;
        else if (argument == "--mip3d-filter") mip3d_filter = true;
        else if (argument == "--full-depth-sort") full_depth_sort = true;
        else if (argument == "--mcmc-noise-lr") mcmc_noise_lr = std::stof(value());
        else if (argument == "--mcmc-refine-start")
            mcmc_refine_start = static_cast<uint32_t>(std::stoul(value()));
        else if (argument == "--mcmc-refine-stop")
            mcmc_refine_stop = static_cast<uint32_t>(std::stoul(value()));
        else if (argument == "--mcmc-refine-every")
            mcmc_refine_every = static_cast<uint32_t>(std::stoul(value()));
        else if (argument == "--mcmc-grow-factor")
            mcmc_grow_factor = std::stof(value());
        else if (argument == "--mcmc-clone-offset")
            mcmc_clone_offset = std::stof(value());
        else if (argument == "--mcmc-opacity-decay")
            mcmc_opacity_decay = std::stof(value());
        else if (argument == "--mcmc-opacity-decay-stop")
            mcmc_opacity_decay_stop =
                    static_cast<uint32_t>(std::stoul(value()));
        else if (argument == "--mcmc-gradient-sampling")
            mcmc_gradient_sampling = std::stof(value());
        else if (argument == "--mcmc-gradient-reference")
            mcmc_gradient_reference = std::stof(value());
        else if (argument == "--mcmc-scale-regularizer")
            mcmc_scale_regularizer = std::stof(value());
        else if (argument == "--mcmc-opacity-regularizer")
            mcmc_opacity_regularizer = std::stof(value());
        else if (argument == "--density-strategy") density_strategy = value();
        else if (argument == "--igs-edge-threshold")
            igs_edge_threshold = std::stof(value());
        else if (argument == "--igs-grad-threshold")
            igs_grad_threshold = std::stof(value());
        else if (argument == "--igs-edge-warmup-steps")
            igs_edge_warmup_steps =
                    static_cast<uint32_t>(std::stoul(value()));
        else if (argument == "--igs-prune-fraction")
            igs_prune_fraction = std::stof(value());
        else if (argument == "--igs-growth-control")
            igs_growth_control = std::stoul(value()) != 0;
        else if (argument == "--igs-multistep-update")
            igs_multistep_update = std::stoul(value()) != 0;
        else if (argument == "--help") {
            std::cout << "cross_3dgs_train --dataset-root PATH --scene NAME "
                         "--image-dir NAME --sparse-dir PATH "
                         "[--render-initial PNG --camera-index N] "
                         "[--backward-smoke] "
                         "[--train-iterations N --max-views N "
                         "--benchmark-json PATH "
                         "--step-times-csv PATH "
                         "--evaluation-image-dir PATH "
                         "--output-ply PATH --sh-degree 0..3 --init-seed N "
                         "--view-order cyclic|fixed-shuffle|random|epoch-shuffle "
                         "--adam-moments bf16|fp32 "
                         "--input-checkpoint PATH "
                         "--ssim-weight 0..1 --full-image-ssim "
                         "--mip3d-filter --full-depth-sort "
                         "--density-strategy fixed|mcmc|igs "
                         "--mcmc-noise-lr F --mcmc-refine-start N "
                         "--mcmc-refine-stop N --mcmc-refine-every N "
                         "--mcmc-grow-factor F --mcmc-scale-regularizer F "
                         "--mcmc-opacity-regularizer F "
                         "--mcmc-clone-offset F --mcmc-opacity-decay F "
                         "--mcmc-opacity-decay-stop N "
                         "--mcmc-gradient-sampling F "
                         "--mcmc-gradient-reference F "
                         "--gaussian-capacity N --warmup N "
                         "--memory-samples-csv PATH "
                         "--pass-profiling --viewer "
                         "--viewer-every N --viewer-width N "
                         "--force-pixel-subgroup-backward "
                         "--force-per-splat-backward "
                         "--force-tiled-backward "
                         "--force-cas-scatter-backward "
                         "--force-streamed-target "
                         "--force-portable-radix-sort "
                         "--force-split-projection-optimizer "
                         "--force-split-sh-rest-storage "
                         "--force-direct-pair-dispatch "
                         "--stochastic-backward-samples 0|1|2|4|8]\n";
            return 0;
        } else {
            throw std::invalid_argument("unknown argument " + argument);
        }
    }
    const auto scene = cross3dgs::LoadColmapScene(
            dataset_root / scene_name, image_dir, sparse_dir);
    std::cout << "scene=" << scene_name << " images=" << scene.cameras.size()
              << " train=" << scene.trainCameraCount()
              << " validation=" << scene.validationCameraCount()
              << " points=" << scene.points.size() << '\n';
    if (!render_initial.empty()) {
        if (camera_index >= scene.cameras.size()) {
            throw std::out_of_range("camera index exceeds scene cameras");
        }
        const auto state =
                cross3dgs::InitializeFromSparsePoints(scene);
        const auto& camera = scene.cameras[camera_index];
        cross3dgs::GaussianForwardProgram forward(
                {.width = camera.width,
                 .height = camera.height,
                 .gaussian_capacity = state.size(),
                 .tile_size = 16,
                 .tile_bin_capacity = 2048,
                 .enable_backward = backward_smoke});
        forward.uploadState(state);
        if (backward_smoke) {
            forward.uploadTarget(
                    cross3dgs::LoadImageRgbaF32(camera.image_path));
        }
        forward.setCamera(camera);
        forward.execute();
        cross3dgs::SaveImagePng(render_initial,
                                forward.downloadColor());
        const auto stats = forward.stats();
        double total_ms = 0.0;
        for (const auto& timing : stats.timings) {
            const double milliseconds = timing.gpu_microseconds / 1000.0;
            total_ms += milliseconds;
            std::cout << "pass=" << timing.label
                      << " gpu_ms=" << milliseconds << '\n';
        }
        std::cout << "render=" << render_initial.string()
                  << " overflow=" << stats.overflow_count
                  << " max_tile_count=" << stats.max_tile_count
                  << " tile_references=" << stats.tile_reference_count
                  << " gpu_ms=" << total_ms << '\n';
        if (backward_smoke) {
            const auto gradients = forward.downloadGradients();
            size_t non_finite = 0;
            for (const auto& values :
                 {&gradients.means, &gradients.scales_opacity,
                  &gradients.rotations, &gradients.sh_dc}) {
                for (const auto& value : *values) {
                    for (int component = 0; component < 4; component++) {
                        if (!std::isfinite(value[component])) non_finite++;
                    }
                }
            }
            std::cout << "mean_l1=" << forward.downloadMeanL1()
                      << " gradient_non_finite=" << non_finite << '\n';
        }
    }
    if (train_iterations > 0) {
        if (sh_degree > 3 || ssim_weight < 0.0f || ssim_weight > 1.0f ||
            (viewer && (viewer_every == 0 || viewer_width == 0)) ||
            mcmc_noise_lr < 0.0f || mcmc_refine_every == 0 ||
            mcmc_refine_start >= mcmc_refine_stop ||
            mcmc_grow_factor < 1.0f ||
            mcmc_clone_offset < 0.0f ||
            mcmc_gradient_sampling < 0.0f ||
            mcmc_gradient_reference <= 0.0f ||
            mcmc_opacity_decay < 0.0f || mcmc_opacity_decay >= 1.0f ||
            mcmc_scale_regularizer < 0.0f ||
            mcmc_opacity_regularizer < 0.0f ||
            (view_order != "cyclic" && view_order != "fixed-shuffle" &&
             view_order != "random" && view_order != "epoch-shuffle") ||
            (adam_moments != "bf16" && adam_moments != "fp32") ||
            (density_strategy != "fixed" && density_strategy != "mcmc" &&
             density_strategy != "igs") ||
            (mip3d_filter && density_strategy == "igs")) {
            throw std::invalid_argument("invalid production training options");
        }
        std::vector<cross3dgs::Camera> cameras;
        std::vector<cross3dgs::FloatImage> targets;
        std::vector<cross3dgs::Camera> validation_cameras;
        std::vector<cross3dgs::FloatImage> validation_targets;
        for (const auto& camera : scene.cameras) {
            if (!camera.validation &&
                (max_views == 0 || cameras.size() < max_views)) {
                cameras.push_back(camera);
                if (!stream_train_targets) {
                    targets.push_back(cross3dgs::LoadImageRgbaF32(
                            camera.image_path));
                }
            } else if (camera.validation) {
                validation_cameras.push_back(camera);
                if (!stream_train_targets) {
                    validation_targets.push_back(
                            cross3dgs::LoadImageRgbaF32(
                                    camera.image_path));
                }
            }
        }
        if (cameras.empty()) {
            throw std::runtime_error("no training cameras selected");
        }
        if (stream_train_targets && viewer) {
            throw std::invalid_argument(
                    "--stream-train-targets is incompatible with --viewer "
                    "(the viewer reads the host target cache)");
        }
        if (stream_train_targets && validation_cameras.empty()) {
            throw std::invalid_argument(
                    "--stream-train-targets requires validation views "
                    "(evaluation would need the host target cache)");
        }
        if (view_order == "fixed-shuffle") {
            std::vector<size_t> order(cameras.size());
            std::iota(order.begin(), order.end(), size_t{0});
            std::mt19937 generator(init_seed);
            std::shuffle(order.begin(), order.end(), generator);
            auto original_cameras = std::move(cameras);
            auto original_targets = std::move(targets);
            cameras.reserve(order.size());
            targets.reserve(original_targets.size());
            for (const size_t index : order) {
                cameras.push_back(std::move(original_cameras[index]));
                if (!stream_train_targets) {
                    targets.push_back(std::move(original_targets[index]));
                }
            }
        }
        auto state = input_checkpoint.empty()
                ? cross3dgs::InitializeFromSparsePoints(
                          scene, 0.1f, 0.5f, init_seed)
                : cross3dgs::LoadGaussianCheckpoint(input_checkpoint);
        const size_t initial_gaussian_count = state.size();
        if (gaussian_capacity == 0) gaussian_capacity = state.size();
        if (gaussian_capacity < state.size())
            throw std::invalid_argument("Gaussian capacity is below initial count");
        // The trainer allocates whole blocks of 32 Gaussians; record both.
        requested_gaussian_capacity = gaussian_capacity;
        gaussian_capacity = cross3dgs::AlignGaussianCapacity(gaussian_capacity);
        const bool enable_mcmc = density_strategy == "mcmc";
        const bool enable_igs = density_strategy == "igs";
        const float scene_scale =
                cross3dgs::CameraExtentSceneScale(scene.cameras);
        std::cout << "scene_scale=" << scene_scale << '\n';
        const double input_setup_seconds =
                std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - process_start)
                        .count();
        const auto pipeline_setup_start = std::chrono::steady_clock::now();
        // Held in an optional only so --stream-train-targets can release the
        // full-capacity training program before the evaluator allocates its
        // own GPU state (keeping both alive doubles the footprint, which is
        // exactly the finalization spike that killed a 6 GB device). The
        // default path never resets it, so lifetime is unchanged there.
        std::optional<cross3dgs::GaussianForwardProgram> trainer_storage;
        trainer_storage.emplace(cross3dgs::GaussianForwardConfig{
                 .width = cameras[0].width,
                 .height = cameras[0].height,
                 .gaussian_capacity = gaussian_capacity,
                 .tile_size = 16,
                 .tile_bin_capacity = 2048,
                 .enable_backward = true,
                 .enable_optimizer = true,
                 .resident_view_capacity =
                         static_cast<uint32_t>(cameras.size()),
                 .randomize_training_views = view_order == "random",
                 .epoch_shuffle_training_views = view_order == "epoch-shuffle",
                 .training_view_seed = init_seed,
                 .sh_degree = sh_degree,
                 // IGS+ raises the positional learning rate (0.000128 ->
                 // 0.0000128) for more granular primitive placement; other
                 // strategies keep the 3DGS default (0.00016 -> 0.0000016).
                 .means_lr =
                         (enable_igs ? 0.000128f : 0.00016f) * scene_scale,
                 .means_lr_final =
                         (enable_igs ? 0.0000128f : 0.0000016f) * scene_scale,
                 .max_steps = 30000,
                 // IGS+ two-stage scale schedule: high-momentum expansion
                 // (0.020) decaying exponentially to 0.002 (ratio 0.1).
                 .scales_lr = enable_igs ? 0.02f : 0.005f,
                 .scales_lr_final = enable_igs ? 0.002f : 0.0f,
                 .l1_weight = 1.0f - ssim_weight,
                 .ssim_weight = ssim_weight,
                 .full_image_ssim = full_image_ssim,
                 .enable_mcmc = enable_mcmc,
                 .mcmc_refine_start = mcmc_refine_start,
                 .mcmc_refine_stop = mcmc_refine_stop,
                 .mcmc_refine_every = mcmc_refine_every,
                 .mcmc_grow_factor = mcmc_grow_factor,
                 .mcmc_noise_lr = mcmc_noise_lr,
                 .mcmc_clone_offset = mcmc_clone_offset,
                 .mcmc_opacity_decay = mcmc_opacity_decay,
                 .mcmc_gradient_sampling = mcmc_gradient_sampling,
                 .mcmc_gradient_reference = mcmc_gradient_reference,
                 .mcmc_opacity_decay_stop = mcmc_opacity_decay_stop,
                 .enable_mip3d_filter = mip3d_filter,
                 .full_depth_sort = full_depth_sort,
                 .scale_regularizer = enable_mcmc
                         ? mcmc_scale_regularizer : 0.0f,
                 .opacity_regularizer = enable_mcmc
                         ? mcmc_opacity_regularizer : 0.0f,
                 .fp32_adam_moments = adam_moments == "fp32",
                 .force_split_sh_rest_storage =
                         force_split_sh_rest_storage,
                 .enable_igs = enable_igs,
                 .igs_grad_threshold = igs_grad_threshold,
                 .igs_edge_threshold = igs_edge_threshold,
                 .igs_edge_warmup_steps = igs_edge_warmup_steps,
                 .igs_growth_control = igs_growth_control,
                 .igs_prune_fraction = igs_prune_fraction,
                 .igs_multistep_update = igs_multistep_update,
                 .stochastic_backward_samples =
                         stochastic_backward_samples,
                 .atomic_free_exact_backward = atomic_free_backward,
                 .force_pixel_subgroup_backward =
                         force_pixel_subgroup_backward,
                 .force_per_splat_backward = force_per_splat_backward,
                 .force_tiled_backward = force_tiled_backward,
                 .force_cas_scatter_backward =
                         force_cas_scatter_backward,
                 .force_streamed_target = force_streamed_target,
                 .force_portable_radix_sort = force_portable_radix_sort,
                 .force_split_projection_optimizer =
                         force_split_projection_optimizer,
                 .force_direct_pair_dispatch =
                         force_direct_pair_dispatch,
                 .enable_pass_profiling = pass_profiling,
                 .fenced_execution = force_fenced_execution,
                 .byte_targets = true});
        auto& trainer = *trainer_storage;
        if (!candidate_ledger_json.empty()) {
            cross3dgs::ScheduleDeviceEvidence device_evidence;
            device_evidence.caps = trainer.deviceCaps();
            if (max_storage_buffer_range != 0 &&
                max_storage_buffer_range !=
                        device_evidence.caps.max_storage_buffer_range) {
                throw std::runtime_error(
                        "supplied maxStorageBufferRange disagrees with live "
                        "Vulkan DeviceCaps");
            }
            device_evidence.max_storage_buffer_range =
                    device_evidence.caps.max_storage_buffer_range;
            device_evidence.evidence_key = caps_evidence_key.empty()
                    ? "live DeviceCaps query of this run"
                    : caps_evidence_key;
            const cross3dgs::ScheduleLedgerWorkload
                    ledger_workload{.gaussian_capacity = gaussian_capacity,
                                    .width = cameras[0].width,
                                    .height = cameras[0].height,
                                    .resident_view_capacity =
                                            static_cast<uint32_t>(
                                                    cameras.size()),
                                    .sh_degree = sh_degree,
                                    .byte_targets = true,
                                    .enable_optimizer = true,
                                    .enable_igs = enable_igs,
                                    .enable_gradient_accumulator =
                                            enable_mcmc &&
                                            mcmc_gradient_sampling > 0.0f};
            const auto ledger =
                    cross3dgs::EnumerateScheduleCandidates(
                            device_evidence, ledger_workload);
            const auto schedule_ir = trainer.schedule();
            const std::string configured_candidate =
                    cross3dgs::ScheduleCandidateId(
                            cross3dgs::FieldsFromScheduleIR(schedule_ir));
            if (!candidate_ledger_json.parent_path().empty()) {
                std::filesystem::create_directories(
                        candidate_ledger_json.parent_path());
            }
            std::ofstream ledger_output(candidate_ledger_json,
                                        std::ios::binary);
            ledger_output << cross3dgs::ScheduleLedgerJson(
                    ledger, device_evidence, ledger_workload, &schedule_ir);
            std::cout << "candidate_ledger=" << candidate_ledger_json
                      << " policy_selected="
                      << ledger.policy_selected_candidate_id
                      << " configured="
                      << configured_candidate
                      << '\n';
            if (ledger.policy_selected_candidate_id.empty()) {
                throw std::runtime_error(
                        "candidate ledger rejected every schedule");
            }
            if (configured_candidate !=
                ledger.policy_selected_candidate_id) {
                const bool explicit_schedule_override =
                        force_pixel_subgroup_backward ||
                        force_per_splat_backward || force_tiled_backward ||
                        atomic_free_backward || force_fenced_execution ||
                        force_cas_scatter_backward || force_streamed_target ||
                        force_portable_radix_sort ||
                        force_split_projection_optimizer ||
                        force_split_sh_rest_storage ||
                        force_direct_pair_dispatch;
                const auto configured_record=std::find_if(
                        ledger.candidates.begin(),ledger.candidates.end(),
                        [&](const auto& candidate) {
                            return candidate.candidate_id==configured_candidate;
                        });
                if(!explicit_schedule_override ||
                   configured_record==ledger.candidates.end() ||
                   !configured_record->accepted) {
                    throw std::runtime_error(
                            "configured schedule is not an accepted policy "
                            "candidate: " + configured_candidate + " != " +
                            ledger.policy_selected_candidate_id);
                }
            }
        }
        trainer.uploadState(state);
        if (stream_train_targets) {
            trainer.uploadTrainingSet(
                    cameras, [&cameras](size_t index) {
                        return cross3dgs::LoadImageRgbaF32(
                                cameras[index].image_path);
                    });
        } else {
            trainer.uploadTrainingSet(cameras, targets);
        }
        const double setup_pipeline_upload_seconds =
                std::chrono::duration<double>(
                        std::chrono::steady_clock::now() -
                        pipeline_setup_start).count();
        // Kept outside the viewer guard so the benchmark JSON carries the
        // same fields whether or not the viewer was built in.
        bool viewer_open = false;
        double viewer_seconds = 0.0;
        uint32_t viewer_frames = 0;
#if CROSS_3DGS_VIEWER
        std::unique_ptr<cross3dgs_viewer::VkViewer> live_viewer;
        if (viewer) {
            const uint32_t source_width = cameras[0].width * 2 + 4;
            const uint32_t viewer_height = std::max(
                    1u, static_cast<uint32_t>(
                                static_cast<uint64_t>(viewer_width) *
                                cameras[0].height / source_width));
            live_viewer = std::make_unique<cross3dgs_viewer::VkViewer>(
                    viewer_width, viewer_height,
                    "IGS+ training: target (left) | render (right)");
            viewer_open = live_viewer->valid();
            if (!viewer_open) {
                std::cerr << "viewer unavailable; continuing headless\n";
            }
        }
#else
        if (viewer) {
            std::cerr << "viewer unavailable; continuing headless\n";
        }
#endif
        std::ofstream memory_samples;
        if (!memory_samples_csv.empty()) {
            if (!memory_samples_csv.parent_path().empty()) {
                std::filesystem::create_directories(
                        memory_samples_csv.parent_path());
            }
            memory_samples.open(memory_samples_csv);
            memory_samples << "completed_iterations,elapsed_wall_seconds,"
                              "tracked_allocator_current_bytes\n";
            memory_samples << 0 << ',' << 0.0 << ','
                           << trainer.stats().current_memory_bytes << '\n';
            memory_samples.flush();
        }
        float initial_loss = 0.0f;
        float final_loss = 0.0f;
        std::vector<double> step_ms;
        std::vector<double> step_elapsed_seconds;
        struct QualityCheckpoint { uint32_t iteration; double train_seconds; size_t count; };
        std::vector<QualityCheckpoint> quality_checkpoints;
        step_ms.reserve(train_iterations);
        step_elapsed_seconds.reserve(train_iterations);
        std::vector<std::pair<std::string, double>> profile_sums;
        uint32_t profile_steps = 0;
        double quality_checkpoint_seconds = 0.0;
        const auto start = std::chrono::steady_clock::now();
        auto last_progress_log = start;
        uint32_t completed = 0;
        while (completed < train_iterations) {
            uint32_t batch = 1;
            if (!pass_profiling && completed > 0 &&
                train_iterations - completed >= 8) {
                const uint32_t until_quality =
                        quality_eval_every == 0
                                ? train_iterations - completed
                                : quality_eval_every -
                                          completed % quality_eval_every;
                const uint32_t until_viewer =
                        !viewer_open
                                ? train_iterations - completed
                                : viewer_every - completed % viewer_every;
                if (until_quality >= 8 && until_viewer >= 8) batch = 8;
            }
            const auto step_start=std::chrono::steady_clock::now();
            if (batch == 8) trainer.executeBatch(8);
            else trainer.execute();
            const auto step_end=std::chrono::steady_clock::now();
            const double batch_ms=std::chrono::duration<double,std::milli>(
                    step_end-step_start).count();
            const double batch_start_elapsed =
                    std::chrono::duration<double>(step_start-start).count();
            for(uint32_t index=0;index<batch;index++) {
                step_ms.push_back(batch_ms/double(batch));
                step_elapsed_seconds.push_back(
                        batch_start_elapsed + batch_ms / 1000.0 *
                                double(index + 1) / double(batch));
            }
            if (pass_profiling) {
                const auto& timings = trainer.stats().timings;
                if (profile_sums.empty()) {
                    for (const auto& timing : timings)
                        profile_sums.emplace_back(timing.label, 0.0);
                }
                for (size_t index = 0; index < timings.size(); index++)
                    profile_sums[index].second +=
                            timings[index].gpu_microseconds / 1000.0;
                profile_steps++;
            }
            if (completed == 0) initial_loss = trainer.downloadMeanL1();
            completed += batch;
            if (memory_samples.is_open()) {
                memory_samples << completed << ','
                               << std::chrono::duration<double>(step_end -
                                                                start).count()
                               << ',' << trainer.stats().current_memory_bytes
                               << '\n';
                memory_samples.flush();
            }
            if (completed == train_iterations)
                final_loss = trainer.downloadMeanL1();
            const auto now = std::chrono::steady_clock::now();
            if (now - last_progress_log >= std::chrono::seconds(10)) {
                last_progress_log = now;
                double timed = 0.0;
                for (double value : step_ms) timed += value / 1000.0;
                spdlog::info("[3dgs] iteration {}/{} timed={:.1f}s "
                             "last_step={:.2f}ms",
                             completed, train_iterations, timed,
                             step_ms.back());
            }
#if CROSS_3DGS_VIEWER
            if (viewer_open &&
                (completed % viewer_every == 0 ||
                 completed == train_iterations)) {
                const auto viewer_start = std::chrono::steady_clock::now();
                const size_t view = (completed - 1) % cameras.size();
                std::ostringstream title;
                title << "IGS+ iteration " << completed << " view " << view
                      << ": target (left) | render (right)";
                live_viewer->setTitle(title.str());
                viewer_open = live_viewer->update(
                        SideBySide(targets[view], trainer.downloadColor()));
                viewer_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - viewer_start).count();
                viewer_frames++;
            }
#endif  // CROSS_3DGS_VIEWER
            if (quality_eval_every > 0 &&
                completed % quality_eval_every == 0 &&
                !quality_curve_dir.empty()) {
                const auto quality_start = std::chrono::steady_clock::now();
                const auto checkpoint = trainer.downloadState();
                cross3dgs::GaussianForwardProgram curve_evaluator(
                        {.width = validation_cameras[0].width,
                         .height = validation_cameras[0].height,
                         .gaussian_capacity = checkpoint.size(),
                         .tile_size = 16,
                         .tile_bin_capacity = 2048,
                         .sh_degree = sh_degree,
                         .enable_mip3d_filter = mip3d_filter});
                curve_evaluator.uploadState(checkpoint);
                for (size_t view = 0; view < validation_cameras.size(); view++) {
                    curve_evaluator.setCamera(validation_cameras[view]);
                    curve_evaluator.execute();
                    std::ostringstream filename;
                    filename << "iteration_" << std::setfill('0')
                             << std::setw(5) << completed << "_view_"
                             << std::setw(3) << view << ".png";
                    cross3dgs::SaveImagePng(
                            quality_curve_dir / filename.str(),
                            curve_evaluator.downloadColor());
                }
                double elapsed_train = 0.0;
                for (double value : step_ms) elapsed_train += value / 1000.0;
                quality_checkpoints.push_back(
                        {completed, elapsed_train, checkpoint.size()});
                quality_checkpoint_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - quality_start)
                                                        .count();
            }
        }
        const double optimization_loop_wall_seconds =
                std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - start).count();
        double seconds = 0.0;
        for (double value : step_ms) seconds += value / 1000.0;
        const size_t warmup=std::min<size_t>(warmup_iterations,step_ms.size()-1);
        std::vector<double> steady(step_ms.begin()+warmup,step_ms.end());
        std::sort(steady.begin(),steady.end());
        const auto percentile=[&](double fraction) {
            const double position=fraction*double(steady.size()-1);
            const size_t lower=static_cast<size_t>(position);
            const size_t upper=std::min(lower+1,steady.size()-1);
            const double weight=position-double(lower);
            return steady[lower]*(1.0-weight)+steady[upper]*weight;
        };
        if (profile_steps > 0) {
            double mean_total = 0.0;
            for (const auto& [label, sum] : profile_sums) {
                const double mean = sum / profile_steps;
                mean_total += mean;
                std::cout << "mean_pass=" << label << " gpu_ms=" << mean
                          << '\n';
            }
            std::cout << "mean_pass_total_gpu_ms=" << mean_total
                      << " profile_steps=" << profile_steps << '\n';
        }
        const auto stats = trainer.stats();
        const auto schedule = trainer.schedule();
        double gpu_ms = 0.0;
        for (const auto& timing : stats.timings) {
            const double milliseconds = timing.gpu_microseconds / 1000.0;
            gpu_ms += milliseconds;
            std::cout << "train_pass=" << timing.label
                      << " gpu_ms=" << milliseconds << '\n';
        }
        const auto final_state_readback_start =
                std::chrono::steady_clock::now();
        state = trainer.downloadState();
        const double final_state_readback_seconds =
                std::chrono::duration<double>(
                        std::chrono::steady_clock::now() -
                        final_state_readback_start).count();
        if (stream_train_targets) {
            // Release the training program before the evaluator allocates
            // its own full-capacity GPU state; `trainer` is not referenced
            // past this point (stats/schedule/state were captured above).
            trainer_storage.reset();
        }
        const auto evaluation_start = std::chrono::steady_clock::now();
        double evaluation_artifact_io_seconds = 0.0;
        const auto& evaluation_cameras = validation_cameras.empty()
                ? cameras : validation_cameras;
        const auto& evaluation_targets = validation_targets.empty()
                ? targets : validation_targets;
        cross3dgs::GaussianForwardProgram evaluator(
                {.width = evaluation_cameras[0].width,
                 .height = evaluation_cameras[0].height,
                 .gaussian_capacity = state.size(),
                 .tile_size = 16,
                 .tile_bin_capacity = 2048,
                 .sh_degree = sh_degree,
                 .enable_mip3d_filter = mip3d_filter});
        evaluator.uploadState(state);
        double evaluation_l1 = 0.0;
        double evaluation_squared_error = 0.0;
        cross3dgs::FloatImage streamed_evaluation_target;
        for (size_t view = 0; view < evaluation_cameras.size(); view++) {
            evaluator.setCamera(evaluation_cameras[view]);
            evaluator.execute();
            const auto rendered = evaluator.downloadColor();
            if (stream_train_targets) {
                streamed_evaluation_target = cross3dgs::LoadImageRgbaF32(
                        evaluation_cameras[view].image_path);
            }
            const auto& evaluation_target = stream_train_targets
                    ? streamed_evaluation_target : evaluation_targets[view];
            if (!evaluation_image_dir.empty()) {
                std::ostringstream filename;
                filename << "view_" << std::setfill('0') << std::setw(3)
                         << view << ".png";
                auto evaluation_artifact_start =
                        std::chrono::steady_clock::now();
                cross3dgs::SaveImagePng(
                        evaluation_image_dir / "target" / filename.str(),
                                evaluation_target);
                cross3dgs::SaveImagePng(
                        evaluation_image_dir / "render" / filename.str(),
                        rendered);
                evaluation_artifact_io_seconds +=
                        std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                evaluation_artifact_start).count();
                auto error = rendered;
                for (size_t pixel = 0;
                     pixel < static_cast<size_t>(rendered.width) *
                                     rendered.height;
                     pixel++) {
                    for (size_t channel = 0; channel < 3; channel++) {
                        error.pixels[pixel * error.channels + channel] =
                                4.0f * std::abs(
                                    rendered.pixels[pixel * rendered.channels +
                                                    channel] -
                                    evaluation_target.pixels[
                                            pixel * evaluation_target.channels +
                                            channel]);
                    }
                }
                evaluation_artifact_start =
                        std::chrono::steady_clock::now();
                cross3dgs::SaveImagePng(
                        evaluation_image_dir / "error_x4" / filename.str(),
                        error);
                evaluation_artifact_io_seconds +=
                        std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                evaluation_artifact_start).count();
            }
            double view_l1 = 0.0;
            const size_t pixel_count =
                    static_cast<size_t>(rendered.width) * rendered.height;
            for (size_t pixel = 0; pixel < pixel_count; pixel++) {
                for (size_t channel = 0; channel < 3; channel++) {
                    view_l1 += std::abs(
                            rendered.pixels[pixel * 4 + channel] -
                            evaluation_target.pixels[pixel * 4 + channel]);
                    const double error=
                            rendered.pixels[pixel * 4 + channel] -
                            evaluation_target.pixels[pixel * 4 + channel];
                    evaluation_squared_error+=error*error;
                }
            }
            evaluation_l1 +=
                    view_l1 / static_cast<double>(pixel_count * 3);
        }
        evaluation_l1 /= static_cast<double>(evaluation_cameras.size());
        const double evaluation_mse=evaluation_squared_error/
                static_cast<double>(evaluation_cameras.size())/
                static_cast<double>(cameras[0].width*cameras[0].height*3);
        const double evaluation_psnr=-10.0*std::log10(evaluation_mse);
        const double evaluation_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - evaluation_start).count();
        size_t non_finite = 0;
        for (const auto& values :
             {&state.means, &state.scales_opacity, &state.rotations,
              &state.sh_dc, &state.sh_rest}) {
            for (const auto& value : *values) {
                for (int component = 0; component < 4; component++) {
                    if (!std::isfinite(value[component])) non_finite++;
                }
            }
        }
        std::cout << "train_iterations=" << train_iterations
                  << " views=" << cameras.size()
                  << " initial_l1=" << initial_loss
                  << " final_l1=" << final_loss
                  << " evaluation_mean_l1=" << evaluation_l1
                  << " evaluation_psnr=" << evaluation_psnr
                  << " median_step_ms=" << percentile(0.5)
                  << " p10_step_ms=" << percentile(0.1)
                  << " p90_step_ms=" << percentile(0.9)
                  << " p99_step_ms=" << percentile(0.99)
                  << " last_gpu_ms=" << gpu_ms
                  << " wall_seconds=" << seconds
                  << " overflow=" << stats.overflow_count
                  << " max_tile_count=" << stats.max_tile_count
                  << " tile_references=" << stats.tile_reference_count
                  << " state_non_finite=" << non_finite << '\n';
        if (viewer) {
            std::cout << "viewer_frames=" << viewer_frames
                      << " viewer_seconds=" << viewer_seconds
                      << " excluded_from_train_timing=1\n";
        }
        const auto artifact_start = std::chrono::steady_clock::now();
        if (step_times_csv.empty() && !benchmark_json.empty()) {
            step_times_csv = benchmark_json.parent_path() /
                             "step_times_ms.csv";
        }
        if (!step_times_csv.empty()) {
            if (!step_times_csv.parent_path().empty()) {
                std::filesystem::create_directories(
                        step_times_csv.parent_path());
            }
            std::ofstream steps(step_times_csv);
            if (!steps) {
                throw std::runtime_error("failed to create step timing CSV");
            }
            steps << "iteration,elapsed_seconds,host_step_ms,step_ms\n";
            for (size_t index = 0; index < step_ms.size(); index++) {
                steps << index + 1 << ',' << step_elapsed_seconds[index]
                      << ','
                      << step_ms[index] << ',' << step_ms[index] << '\n';
            }
        }
        if (!output_ply.empty()) {
            cross3dgs::SaveGaussianPly(output_ply, state);
            std::cout << "output_ply=" << output_ply.string() << '\n';
        }
        if (!output_checkpoint.empty()) {
            cross3dgs::SaveGaussianCheckpoint(output_checkpoint, state);
            const auto reloaded =
                    cross3dgs::LoadGaussianCheckpoint(output_checkpoint);
            const auto equal=[](const auto& left,const auto& right) {
                return left.size()==right.size() &&
                    std::memcmp(left.data(),right.data(),
                                left.size()*sizeof(left[0]))==0;
            };
            if (!equal(reloaded.means,state.means) ||
                !equal(reloaded.scales_opacity,state.scales_opacity) ||
                !equal(reloaded.rotations,state.rotations) ||
                !equal(reloaded.sh_dc,state.sh_dc) ||
                !equal(reloaded.sh_rest,state.sh_rest)) {
                throw std::runtime_error("checkpoint reload mismatch");
            }
            std::cout << "output_checkpoint=" << output_checkpoint.string()
                      << " reload=exact\n";
        }
        const double artifact_io_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - artifact_start).count();
        const double run_wall_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - process_start).count();
        if (!benchmark_json.empty()) {
            if (!benchmark_json.parent_path().empty()) {
                std::filesystem::create_directories(
                        benchmark_json.parent_path());
            }
            std::ofstream output(benchmark_json);
            if (!output) {
                throw std::runtime_error(
                        "failed to create benchmark JSON");
            }
            output << "{\n"
                   << "  \"schema_version\": 1,\n"
                   << "  \"backend\": \"Cross3DGS Vulkan\",\n"
                   << "  \"scene\": \"" << scene_name << "\",\n"
                   << "  \"image_dir\": \""
                   << std::filesystem::path(image_dir).generic_string()
                   << "\",\n"
                   << "  \"sparse_dir\": \""
                   << std::filesystem::path(sparse_dir).generic_string()
                   << "\",\n"
                   << "  \"sampled_train_image_count\": "
                   << cameras.size() << ",\n"
                   << "  \"validation_image_count\": "
                   << evaluation_cameras.size() << ",\n"
                   << "  \"sh_degree\": " << sh_degree << ",\n"
                   << "  \"l1_weight\": " << (1.0f-ssim_weight) << ",\n"
                   << "  \"ssim_weight\": " << ssim_weight << ",\n"
                   << "  \"full_image_ssim\": "
                   << (full_image_ssim ? "true" : "false") << ",\n"
                   << "  \"mip3d_filter\": "
                   << (mip3d_filter ? "true" : "false") << ",\n"
                   << "  \"full_depth_sort\": "
                   << (full_depth_sort ? "true" : "false") << ",\n"
                   << "  \"density_strategy\": \"" << density_strategy
                   << "\",\n"
                   << "  \"mcmc_noise_lr\": " << mcmc_noise_lr << ",\n"
                   << "  \"mcmc_refine_start\": " << mcmc_refine_start << ",\n"
                   << "  \"mcmc_refine_stop\": " << mcmc_refine_stop << ",\n"
                   << "  \"mcmc_refine_every\": " << mcmc_refine_every << ",\n"
                   << "  \"mcmc_grow_factor\": " << mcmc_grow_factor << ",\n"
                   << "  \"mcmc_clone_offset\": " << mcmc_clone_offset << ",\n"
                   << "  \"mcmc_opacity_decay\": " << mcmc_opacity_decay
                   << ",\n"
                   << "  \"mcmc_opacity_decay_stop\": "
                   << mcmc_opacity_decay_stop << ",\n"
                   << "  \"mcmc_gradient_sampling\": "
                   << mcmc_gradient_sampling << ",\n"
                   << "  \"mcmc_gradient_reference\": "
                   << mcmc_gradient_reference << ",\n"
                   << "  \"mcmc_scale_regularizer\": "
                   << mcmc_scale_regularizer << ",\n"
                   << "  \"mcmc_opacity_regularizer\": "
                   << mcmc_opacity_regularizer << ",\n"
                   << "  \"requested_gaussian_capacity\": "
                   << requested_gaussian_capacity << ",\n"
                   << "  \"gaussian_capacity\": " << gaussian_capacity << ",\n"
                   << "  \"initial_gaussian_count\": " << initial_gaussian_count
                   << ",\n"
                   << "  \"final_gaussian_count\": " << state.size()
                   << ",\n"
                   << "  \"iterations\": " << train_iterations << ",\n"
                   << "  \"warmup_iterations\": " << warmup << ",\n"
                   << "  \"stochastic_backward_samples\": "
                   << stochastic_backward_samples << ",\n"
                   << "  \"init_seed\": " << init_seed << ",\n"
                   << "  \"view_order\": \"" << view_order << "\",\n"
                   << "  \"adam_moments\": \"" << adam_moments << "\",\n"
                   << "  \"alpha_cap\": 0.999,\n"
                   << "  \"first_iteration_ms\": " << step_ms.front() << ",\n"
                   << "  \"median_step_ms\": " << percentile(0.5)
                   << ",\n"
                   << "  \"p10_step_ms\": " << percentile(0.1) << ",\n"
                   << "  \"p90_step_ms\": " << percentile(0.9) << ",\n"
                   << "  \"p99_step_ms\": " << percentile(0.99) << ",\n"
                   << "  \"total_train_seconds\": " << seconds << ",\n"
                   << "  \"total_train_seconds_semantics\": "
                      "\"sum_of_host_execute_intervals\",\n"
                   << "  \"input_setup_seconds\": " << input_setup_seconds
                   << ",\n"
                   << "  \"setup_pipeline_upload_seconds\": "
                   << setup_pipeline_upload_seconds << ",\n"
                   << "  \"optimization_loop_wall_seconds\": "
                   << optimization_loop_wall_seconds << ",\n"
                   << "  \"step_host_sum_seconds\": " << seconds << ",\n"
                   << "  \"quality_checkpoint_seconds\": "
                   << quality_checkpoint_seconds << ",\n"
                   << "  \"final_state_readback_seconds\": "
                   << final_state_readback_seconds << ",\n"
                   << "  \"evaluation_seconds\": " << evaluation_seconds
                   << ",\n"
                   << "  \"evaluation_artifact_io_seconds\": "
                   << evaluation_artifact_io_seconds << ",\n"
                   << "  \"evaluation_compute_readback_seconds\": "
                   << evaluation_seconds - evaluation_artifact_io_seconds
                   << ",\n"
                   << "  \"artifact_io_seconds\": " << artifact_io_seconds
                   << ",\n"
                   << "  \"run_wall_seconds\": " << run_wall_seconds
                   << ",\n"
                   << "  \"run_wall_scope\": "
                      "\"process_start_through_artifacts_before_bench_write\",\n"
                   << "  \"step_times_csv\": \""
                   << step_times_csv.generic_string() << "\",\n"
                   << "  \"memory_samples_csv\": \""
                   << memory_samples_csv.generic_string() << "\",\n"
                   << "  \"memory_samples_scope\": "
                      "\"trainer_tracked_allocator_bytes\",\n"
                   << "  \"step_times_elapsed_semantics\": "
                      "\"execute_completion_wall_interpolated_within_batch\",\n"
                   << "  \"step_times_host_step_semantics\": "
                      "\"execute_batch_host_wall_divided_by_batch_size\",\n"
                   << "  \"viewer_enabled\": "
                   << (viewer ? "true" : "false") << ",\n"
                   << "  \"viewer_frames\": " << viewer_frames << ",\n"
                   << "  \"viewer_seconds\": " << viewer_seconds << ",\n"
                   << "  \"viewer_timing_excluded\": true,\n"
                   << "  \"evaluation_mean_l1\": " << evaluation_l1
                   << ",\n"
                   << "  \"evaluation_psnr\": " << evaluation_psnr
                   << ",\n"
                   << "  \"evaluation_split\": \"validation\",\n"
                   << "  \"overflow_count\": " << stats.overflow_count
                   << ",\n"
                   << "  \"overflow_update_count\": "
                   << stats.overflow_update_count << ",\n"
                   << "  \"peak_pair_count\": " << stats.peak_pair_count
                   << ",\n"
                   << "  \"tile_reference_count\": "
                   << stats.tile_reference_count << ",\n"
                   << "  \"max_tile_count\": " << stats.max_tile_count
                   << ",\n"
                   << "  \"pair_capacity\": " << stats.pair_capacity
                   << ",\n"
                   << "  \"atomic_free_pair_scratch_bytes\": "
                   << stats.atomic_free_pair_scratch_bytes << ",\n"
                   << "  \"atomic_free_pair_scratch_used_bytes\": "
                   << stats.atomic_free_pair_scratch_used_bytes << ",\n"
                   << "  \"state_non_finite\": " << non_finite << ",\n"
                   << "  \"last_gpu_ms\": " << gpu_ms << ",\n"
                   << "  \"gpu_timestamp_scope\": "
                      "\"last_instrumented_execute_only_not_total_training\",\n"
                   << "  \"gpu_total_training_seconds\": null,\n"
                   << "  \"gpu_total_training_seconds_note\": "
                      "\"not_reported_without_reliable_whole-run_timestamps\",\n"
                   << "  \"current_memory_bytes\": "
                   << stats.current_memory_bytes << ",\n"
                   << "  \"peak_memory_bytes\": "
                   << stats.peak_memory_bytes << ",\n"
                   << "  \"schedule_ir\": {\n"
                   << "    \"vendor_id\": " << schedule.vendor_id << ",\n"
                   << "    \"device_id\": " << schedule.device_id << ",\n"
                   << "    \"subgroup_size\": " << schedule.subgroup_size << ",\n"
                   << "    \"max_shared_memory\": "
                   << schedule.max_shared_memory << ",\n"
                   << "    \"max_storage_buffer_bindings\": "
                   << schedule.max_storage_buffer_bindings << ",\n"
                   << "    \"max_storage_buffer_range\": "
                   << schedule.max_storage_buffer_range << ",\n"
                   << "    \"tile_size\": " << schedule.tile_size << ",\n"
                   << "    \"subtile_size\": " << schedule.subtile_size << ",\n"
                   << "    \"raster_workgroup_size\": "
                   << schedule.raster_workgroup_size << ",\n"
                   << "    \"backward_workgroup_size\": "
                   << schedule.backward_workgroup_size << ",\n"
                   << "    \"projection_optimizer_workgroup_size\": "
                   << schedule.projection_optimizer_workgroup_size << ",\n"
                   << "    \"command_batch_size\": "
                   << schedule.command_batch_size << ",\n"
                   << "    \"backward_work_assignment\": \""
                   << schedule.backward_work_assignment << "\",\n"
                   << "    \"backward_global_merge\": \""
                   << schedule.backward_global_merge << "\",\n"
                   << "    \"backward_tile_shape\": \""
                   << schedule.backward_tile_shape << "\",\n"
                   << "    \"backward_selection_policy\": \""
                   << schedule.backward_selection_policy << "\",\n"
                   << "    \"schedule_candidate_id\": \""
                   << schedule.schedule_candidate_id << "\",\n"
                   << "    \"policy_selected_candidate_id\": \""
                   << schedule.policy_selected_candidate_id << "\",\n"
                   << "    \"schedule_catalog_version\": \""
                   << schedule.schedule_catalog_version << "\",\n"
                   << "    \"schedule_catalog_sha256\": \""
                   << schedule.schedule_catalog_sha256 << "\",\n"
                   << "    \"schedule_policy_id\": \""
                   << schedule.schedule_policy_id << "\",\n"
                   << "    \"schedule_policy_sha256\": \""
                   << schedule.schedule_policy_sha256 << "\",\n"
                   << "    \"schedule_selection_source\": \""
                   << schedule.schedule_selection_source << "\",\n"
                   << "    \"schedule_candidate_accepted\": "
                   << (schedule.schedule_candidate_accepted ? "true" : "false")
                   << ",\n"
                   << "    \"tiled_backward\": "
                   << (schedule.tiled_backward ? "true" : "false") << ",\n"
                   << "    \"per_splat_backward\": "
                   << (schedule.per_splat_backward ? "true" : "false") << ",\n"
                   << "    \"projection_optimizer_fusion\": "
                   << (schedule.projection_optimizer_fusion ? "true" : "false") << ",\n"
                   << "    \"subgroup_reordered_sh\": "
                   << (schedule.subgroup_reordered_sh ? "true" : "false") << ",\n"
                   << "    \"split_sh_rest_storage\": "
                   << (schedule.split_sh_rest_storage ? "true" : "false")
                   << ",\n"
                   << "    \"mcmc_indirect_dispatch\": "
                   << (schedule.mcmc_indirect_dispatch ? "true" : "false")
                   << ",\n    \"native_float_atomic_add\": "
                   << (schedule.native_float_atomic_add ? "true" : "false")
                   << ",\n    \"f16_storage\": "
                   << (schedule.f16_storage ? "true" : "false")
                   << ",\n    \"byte_targets\": "
                   << (schedule.byte_targets ? "true" : "false")
                   << ",\n    \"fenced_execution\": "
                   << (schedule.fenced_execution ? "true" : "false")
                   << ",\n    \"atomic_free_exact_backward\": "
                   << (schedule.atomic_free_exact_backward ? "true" : "false")
                   << ",\n    \"streamed_target\": "
                   << (schedule.streamed_target ? "true" : "false")
                   << ",\n    \"portable_radix_sort\": "
                   << (schedule.portable_radix_sort ? "true" : "false")
                   << ",\n    \"indirect_pair_dispatch\": "
                   << (schedule.indirect_pair_dispatch ? "true" : "false")
                   << "\n  },\n"
                   << "  \"stage_timings_ms\": {\n";
            for (size_t index=0;index<stats.timings.size();index++) {
                const auto& timing=stats.timings[index];
                output << "    \"" << timing.label << "\": "
                       << timing.gpu_microseconds/1000.0;
                output << (index+1==stats.timings.size() ? "\n" : ",\n");
            }
            output << "  },\n  \"quality_curve_checkpoints\": [\n";
            for (size_t index=0;index<quality_checkpoints.size();index++) {
                const auto& checkpoint=quality_checkpoints[index];
                output << "    {\"iteration\": " << checkpoint.iteration
                       << ", \"train_seconds\": " << checkpoint.train_seconds
                       << ", \"gaussian_count\": " << checkpoint.count << "}"
                       << (index+1==quality_checkpoints.size()?"\n":",\n");
            }
            output << "  ]\n}\n";
        }
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "cross_3dgs_train: " << error.what() << '\n';
    return 1;
}
