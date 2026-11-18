#include "program_impl.h"
#include "shaders.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <spdlog/spdlog.h>

// Set by libs/cross3dgs/CMakeLists.txt; on when built some other way.
#ifndef CROSS3DGS_SH_REORDER_NARROW_SUBGROUPS
#define CROSS3DGS_SH_REORDER_NARROW_SUBGROUPS 1
#endif

namespace cross3dgs {
using namespace detail;

namespace {
gpu::TensorDesc BufferDesc(
        gpu::ScalarType scalar, uint32_t lanes, size_t count,
        gpu::StorageClass storage = gpu::StorageClass::StorageBuffer) {
    gpu::TensorDesc desc;
    desc.storage = storage;
    desc.scalar_type = scalar;
    desc.lanes = lanes;
    desc.logical_shape = {1, {static_cast<uint32_t>(count), 1, 1, 1}};
    desc.capacity_bytes = count * desc.elementBytes();
    return desc;
}

GaussianForwardConfig WithAlignedCapacity(GaussianForwardConfig config) {
    config.gaussian_capacity = AlignGaussianCapacity(config.gaussian_capacity);
    return config;
}

}  // namespace

GaussianForwardProgram::Impl::Impl(const GaussianForwardConfig& input_config,
                                   gpu::VkContextPtr context)
    : config(WithAlignedCapacity(input_config)), program(std::move(context)) {
    const auto options = selectSchedule(input_config);
    createBuffers(options);
    // Qualcomm requires the transfer clear before the forward compute chain.
    if (config.enable_backward && options.adreno_camera_workaround) {
        addGradientClear(options.fuse_projection_optimizer);
    }
    addForwardPasses(options.adreno_camera_workaround);
    if (config.enable_backward) {
        addBackwardPasses(options);
        if (config.enable_optimizer) {
            addOptimizerPasses(options.fuse_projection_optimizer);
            if (config.enable_mcmc) addMcmcPasses();
            if (config.enable_igs) addIgsPasses();
            program.addPass({"increment_optimizer_step",
                             IncrementStepShader(options.adreno_camera_workaround),
                             {optimizer_meta},
                             {{optimizer_meta,
                               gpu::PipelineStage::Compute,
                               gpu::AccessMode::ReadWrite,
                               {}}},
                             gpu::DispatchDesc::Direct(1)});
        }
    }
    buildAndInitialize();
}

detail::GaussianBuildOptions GaussianForwardProgram::Impl::selectSchedule(
        const GaussianForwardConfig& input_config) {
    if (config.width == 0 || config.height == 0 || config.gaussian_capacity == 0 ||
        config.tile_size != 16 || config.sh_degree > 3 || config.l1_weight < 0.0f ||
        config.ssim_weight < 0.0f ||
        std::abs(config.l1_weight + config.ssim_weight - 1.0f) > 1e-6f ||
        config.tile_bin_capacity == 0 || config.tile_bin_capacity > 8192 ||
        (config.enable_optimizer && !config.enable_backward) ||
        (config.enable_mcmc && !config.enable_optimizer) ||
        (config.enable_mcmc &&
         (config.mcmc_refine_every == 0 || config.mcmc_grow_factor < 1.0f ||
          config.mcmc_refine_start >= config.mcmc_refine_stop)) ||
        (config.enable_mip3d_filter &&
         (config.enable_igs || config.mip3d_every == 0)) ||
        (config.enable_igs && !config.enable_optimizer) ||
        (config.enable_igs && config.enable_mcmc) ||
        (config.enable_igs && (config.igs_refine_every == 0 ||
                               config.igs_refine_start >= config.igs_refine_stop)) ||
        (config.stochastic_backward_samples != 0 &&
         (!config.enable_backward || config.stochastic_backward_samples > 8 ||
          (config.stochastic_backward_samples &
           (config.stochastic_backward_samples - 1)) != 0)) ||
        (config.atomic_free_exact_backward &&
         (!config.enable_backward || config.stochastic_backward_samples != 0)) ||
        ((static_cast<uint32_t>(config.force_pixel_subgroup_backward) +
          static_cast<uint32_t>(config.force_per_splat_backward) +
          static_cast<uint32_t>(config.force_tiled_backward)) > 1u) ||
        (config.atomic_free_exact_backward &&
         (config.force_pixel_subgroup_backward || config.force_tiled_backward)) ||
        ((config.force_pixel_subgroup_backward || config.force_per_splat_backward ||
          config.force_tiled_backward) &&
         (!config.enable_backward || config.stochastic_backward_samples != 0)) ||
        (config.force_cas_scatter_backward &&
         (!config.enable_backward || config.atomic_free_exact_backward ||
          config.stochastic_backward_samples != 0)) ||
        (config.force_streamed_target &&
         (!config.enable_backward || config.resident_view_capacity == 0)) ||
        (config.resident_view_capacity > 0 && !config.enable_optimizer)) {
        throw std::invalid_argument("invalid GaussianForwardConfig");
    }
    grid_width = (config.width + 15) / 16;
    grid_height = (config.height + 15) / 16;
    tile_count = grid_width * grid_height;
    uint32_t tile_bits = 0;
    while ((uint64_t{1} << tile_bits) < tile_count) tile_bits++;
    if (tile_bits == 0 || tile_bits >= 32) {
        throw std::invalid_argument("unsupported tile grid");
    }
    depth_bits = config.full_depth_sort ? 0u : 32u - tile_bits;
    pair_key_bits = config.full_depth_sort ? tile_bits : 32u;
    const uint64_t sh_rest_full_bytes =
            uint64_t{config.gaussian_capacity} * 12u * sizeof(glm::vec4);
    const uint64_t sh_rest_shard_bytes = sh_rest_full_bytes / 2u;
    const uint64_t storage_range = program.deviceCaps().max_storage_buffer_range;
    split_sh_rest_storage = config.force_split_sh_rest_storage ||
                            (storage_range != 0 && sh_rest_full_bytes > storage_range);
    if (split_sh_rest_storage &&
        (storage_range == 0 || sh_rest_shard_bytes > storage_range)) {
        throw std::runtime_error("split SH-rest shard exceeds maxStorageBufferRange");
    }
    if (split_sh_rest_storage && config.enable_igs) {
        throw std::runtime_error("split SH-rest storage is not yet supported with IGS");
    }
    // The reordered SH-rest layout interleaves fixed blocks of 32 Gaussians so
    // each packed vec4 load/store is contiguous across neighboring threads.
    // It does not depend on the subgroup width; without it, 64-wide subgroups
    // read SH and Adam moments with a 192-byte stride in the fused
    // projection-backward + Adam pass.
    // Subgroups narrower than 32 (Mali, 16 lanes) use it too unless the CMake
    // option CROSS3DGS_SH_REORDER_NARROW_SUBGROUPS is OFF. The blocks always
    // fit: the capacity is rounded up to a multiple of 32
    // (AlignGaussianCapacity).
    reorder_sh = !split_sh_rest_storage &&
                 (CROSS3DGS_SH_REORDER_NARROW_SUBGROUPS ||
                  program.deviceCaps().subgroup_size >= 32);
    program.setProfilingEnabled(config.enable_pass_profiling);
    program.setConservativeBarriers(config.conservative_barriers);
    ScheduleDeviceEvidence schedule_device;
    schedule_device.caps = program.deviceCaps();
    schedule_device.max_storage_buffer_range =
            program.deviceCaps().max_storage_buffer_range;
    schedule_device.evidence_key = "live Vulkan DeviceCaps query";
    const ScheduleLedgerWorkload schedule_workload{
            .gaussian_capacity = config.gaussian_capacity,
            .width = config.width,
            .height = config.height,
            .resident_view_capacity = config.resident_view_capacity,
            .sh_degree = config.sh_degree,
            .byte_targets = config.byte_targets,
            .enable_optimizer = config.enable_optimizer,
            .enable_igs = config.enable_igs,
            .enable_gradient_accumulator =
                    config.enable_mcmc && config.mcmc_gradient_sampling > 0.0f};
    if (config.enable_backward) {
        schedule_ledger =
                EnumerateScheduleCandidates(schedule_device, schedule_workload);
        if (schedule_ledger.policy_selected_candidate_id.empty()) {
            throw std::runtime_error(
                    "runtime candidate ledger rejected every schedule");
        }
        const auto policy_record = std::find_if(
                schedule_ledger.candidates.begin(), schedule_ledger.candidates.end(),
                [&](const auto& candidate) {
                    return candidate.candidate_id ==
                           schedule_ledger.policy_selected_candidate_id;
                });
        if (policy_record == schedule_ledger.candidates.end()) {
            throw std::runtime_error(
                    "runtime candidate ledger policy selection is missing");
        }
        ScheduleCandidateFields selected = policy_record->fields;
        if (input_config.atomic_free_exact_backward) {
            selected.work_assignment = input_config.force_per_splat_backward
                                               ? "per_splat"
                                               : "pair_partial";
            selected.global_merge = "gaussian_major_fixed_reduce";
            selected.tile_shape = "none";
        } else if (input_config.force_pixel_subgroup_backward ||
                   input_config.force_per_splat_backward ||
                   input_config.force_tiled_backward ||
                   input_config.force_cas_scatter_backward) {
            selected.work_assignment =
                    input_config.force_tiled_backward      ? "tiled_8x8"
                    : input_config.force_per_splat_backward ? "per_splat"
                                                            : "pixel_stream_subgroup";
            selected.tile_shape =
                    input_config.force_tiled_backward ? "8x8" : "none";
            selected.global_merge =
                    input_config.force_cas_scatter_backward ||
                                    !program.deviceCaps().float32_atomic_add
                            ? "u32_cas"
                            : "native_atomic";
        }
        if (input_config.fenced_execution) {
            selected.submission = "per_pass_fenced";
        }
        if (input_config.force_streamed_target) selected.targets = "streamed";
        if (input_config.force_portable_radix_sort) {
            selected.radix = "subgroup_independent";
        }
        if (input_config.force_split_projection_optimizer) {
            selected.projection_adam = "split";
        }
        // A split SH-rest layout needs two parameter/gradient/moment bindings
        // and is implemented only by the split projection+Adam passes. The
        // ledger rejects fused candidates whenever the capacity forces the
        // split (HC-BINDING-SHREST-SPLIT-01), so this only acts on
        // force_split_sh_rest_storage, which the ledger workload does not
        // see; schedule_ir must not claim a fused pass that never ran.
        if (split_sh_rest_storage) selected.projection_adam = "split";
        if (input_config.force_direct_pair_dispatch) {
            selected.pair_dispatch = "direct";
        }
        if (input_config.unsafe_force_batched_submission) {
            selected.submission = "batched_8";
        }
        if (input_config.unsafe_force_subgroup_radix) {
            selected.radix = "subgroup_dependent";
        }
        selected_candidate_id = ScheduleCandidateId(selected);
        const auto selected_record = std::find_if(
                schedule_ledger.candidates.begin(), schedule_ledger.candidates.end(),
                [&](const auto& candidate) {
                    return candidate.candidate_id == selected_candidate_id;
                });
        if (selected_record == schedule_ledger.candidates.end()) {
            throw std::runtime_error(
                    "requested runtime schedule is outside candidate schema: " +
                    selected_candidate_id);
        }
        selected_candidate_accepted = selected_record->accepted;
        const bool unsafe_probe = input_config.unsafe_allow_rejected_schedule ||
                                  input_config.unsafe_force_batched_submission ||
                                  input_config.unsafe_force_subgroup_radix;
        if (!selected_candidate_accepted && !unsafe_probe) {
            throw std::runtime_error(
                    "requested runtime schedule is rejected by candidate "
                    "ledger: " +
                    selected_candidate_id);
        }

        // From this point onward the executable configuration is derived from
        // the selected ledger tuple. The input flags only constrain which
        // accepted tuple is selected above; they are not a second selector.
        config.atomic_free_exact_backward =
                selected.global_merge == "gaussian_major_fixed_reduce";
        config.force_pixel_subgroup_backward =
                selected.work_assignment == "pixel_stream_subgroup";
        config.force_per_splat_backward = selected.work_assignment == "per_splat";
        config.force_tiled_backward = selected.work_assignment == "tiled_8x8";
        config.force_cas_scatter_backward = selected.global_merge == "u32_cas";
        config.fenced_execution = selected.submission == "per_pass_fenced";
        config.force_streamed_target = selected.targets == "streamed";
        config.force_portable_radix_sort = selected.radix == "subgroup_independent";
        config.force_split_projection_optimizer = selected.projection_adam == "split";
        config.force_direct_pair_dispatch = selected.pair_dispatch == "direct";
    }
    // The ledger constraint catalog encodes the Adreno submission,
    // target, and radix restrictions. Do not repeat those vendor rules
    // here: the selected tuple is the sole source of executable flags.
    use_fenced_execution =
            config.fenced_execution && !config.unsafe_force_batched_submission;
    const bool adreno_camera_workaround = program.deviceCaps().vendor_id == 0x5143;
    const bool adreno_ssim_channel_split = program.deviceCaps().vendor_id == 0x5143;
    stream_resident_target = config.force_streamed_target && config.enable_backward &&
                             config.resident_view_capacity > 0;
    use_host_step_fill = (adreno_camera_workaround || stream_resident_target) &&
                         config.enable_optimizer && config.resident_view_capacity > 0;
    // Edge-Aware Score needs the resident multi-view target (edge weight in
    // its alpha channel) sampled from the projection forward. The streamed
    // single-view Adreno target path is unsupported, so IGS+ falls back to
    // the pure-gradient gate there.
    igs_edge_enabled = config.enable_igs && !stream_resident_target &&
                       config.enable_backward && config.resident_view_capacity > 0;
    gradient_accumulator =
            config.enable_backward &&
            (config.enable_igs ||
             (config.enable_mcmc && config.mcmc_gradient_sampling > 0.0f));
    // Multi-step Update needs the split backward+Adam passes (it gates the
    // Adam step and accumulates gradients between updates), so it forces
    // the non-fused path.
    const bool fuse_projection_optimizer =
            config.enable_optimizer && !split_sh_rest_storage &&
            !config.igs_multistep_update && !config.force_split_projection_optimizer &&
            program.deviceCaps().max_storage_buffer_bindings >=
                    (gradient_accumulator ? 26u : 25u);
    // Same limits as the ledger rule HC-SUBGROUP-TILED-01.
    const bool tiled_capable =
            program.deviceCaps().subgroup_size >= kTiledBackwardMinSubgroupSize &&
            program.deviceCaps().max_shared_memory >= kTiledBackwardSharedBytes;
    if (config.force_tiled_backward && !tiled_capable) {
        throw std::runtime_error(
                "forced tiled_8x8 backward is not supported by device caps");
    }
    const bool explicit_scatter_assignment = config.force_pixel_subgroup_backward ||
                                             config.force_per_splat_backward ||
                                             config.force_tiled_backward;
    const bool use_tiled_backward =
            config.enable_backward && config.stochastic_backward_samples == 0 &&
            !config.atomic_free_exact_backward && tiled_capable &&
            (config.force_tiled_backward ||
             (!config.force_cas_scatter_backward && !explicit_scatter_assignment));
    const bool use_per_splat_backward =
            config.enable_backward && config.stochastic_backward_samples == 0 &&
            !config.atomic_free_exact_backward && config.force_per_splat_backward;
    const bool use_per_splat_pair_backward =
            config.enable_backward && config.stochastic_backward_samples == 0 &&
            config.atomic_free_exact_backward && config.force_per_splat_backward;
    const bool use_native_float_atomic_add = program.deviceCaps().float32_atomic_add &&
                                             !config.force_cas_scatter_backward;
    schedule_ir.vendor_id = program.deviceCaps().vendor_id;
    schedule_ir.device_id = program.deviceCaps().device_id;
    schedule_ir.subgroup_size = program.deviceCaps().subgroup_size;
    schedule_ir.max_shared_memory = program.deviceCaps().max_shared_memory;
    schedule_ir.max_storage_buffer_bindings =
            program.deviceCaps().max_storage_buffer_bindings;
    schedule_ir.max_storage_buffer_range =
            program.deviceCaps().max_storage_buffer_range;
    schedule_ir.tile_size = config.tile_size;
    schedule_ir.command_batch_size = config.resident_view_capacity > 0 ? 8u : 1u;
    schedule_ir.tiled_backward = use_tiled_backward;
    schedule_ir.per_splat_backward =
            use_per_splat_backward || use_per_splat_pair_backward;
    if (use_per_splat_pair_backward) {
        schedule_ir.backward_work_assignment = "per_splat";
        schedule_ir.backward_global_merge = "gaussian_major_fixed_reduce";
    } else if (config.atomic_free_exact_backward) {
        schedule_ir.backward_work_assignment = "pair_partial";
        schedule_ir.backward_global_merge = "gaussian_major_fixed_reduce";
    } else if (use_tiled_backward) {
        schedule_ir.backward_work_assignment = "tiled_8x8";
        schedule_ir.backward_global_merge =
                use_native_float_atomic_add ? "native_atomic" : "u32_cas";
        schedule_ir.backward_tile_shape = "8x8";
    } else if (use_per_splat_backward) {
        schedule_ir.backward_work_assignment = "per_splat";
        schedule_ir.backward_global_merge =
                use_native_float_atomic_add ? "native_atomic" : "u32_cas";
    } else if (config.enable_backward && config.stochastic_backward_samples == 0) {
        schedule_ir.backward_work_assignment = "pixel_stream_subgroup";
        schedule_ir.backward_global_merge =
                use_native_float_atomic_add ? "native_atomic" : "u32_cas";
    } else if (config.stochastic_backward_samples != 0) {
        schedule_ir.backward_work_assignment = "gaussian_sobol_sampled";
        schedule_ir.backward_global_merge = "none";
    }
    schedule_ir.backward_workgroup_size = use_tiled_backward ? 256u : 128u;
    schedule_ir.projection_optimizer_fusion = fuse_projection_optimizer;
    schedule_ir.subgroup_reordered_sh = reorder_sh;
    schedule_ir.split_sh_rest_storage = split_sh_rest_storage;
    schedule_ir.mcmc_indirect_dispatch = config.enable_mcmc &&
                                         program.deviceCaps().dispatch_indirect;
    schedule_ir.native_float_atomic_add = use_native_float_atomic_add;
    schedule_ir.f16_storage = program.deviceCaps().f16_storage;
    schedule_ir.byte_targets = config.byte_targets;
    schedule_ir.fenced_execution = use_fenced_execution;
    schedule_ir.atomic_free_exact_backward = config.atomic_free_exact_backward;
    schedule_ir.streamed_target = stream_resident_target;
    schedule_ir.portable_radix_sort =
            config.force_portable_radix_sort && !config.unsafe_force_subgroup_radix;
    if (config.enable_backward) {
        schedule_ir.schedule_candidate_id = selected_candidate_id;
        schedule_ir.policy_selected_candidate_id =
                schedule_ledger.policy_selected_candidate_id;
        schedule_ir.schedule_catalog_version = schedule_ledger.catalog_version;
        schedule_ir.schedule_catalog_sha256 = schedule_ledger.catalog_sha256;
        schedule_ir.schedule_policy_id = schedule_ledger.policy_id;
        schedule_ir.schedule_policy_sha256 = schedule_ledger.policy_sha256;
        schedule_ir.schedule_candidate_accepted = selected_candidate_accepted;
        if (split_sh_rest_storage) {
            schedule_ir.schedule_selection_source =
                    "runtime_candidate_ledger+storage_range_lowering";
        }
    } else {
        schedule_ir.schedule_selection_source = "not_applicable_forward_only";
    }
    spdlog::debug(
            "[3dgs] schedule vendor={:#06x} device={:#x} subgroup={} "
            "shared={} backward={}/{} tiled_backward={} per_splat={} gather={} "
            "fusion={} "
            "reordered_sh={} batch={} fenced={} streamed_target={} "
            "byte_targets={} float_atomic={}",
            schedule_ir.vendor_id, schedule_ir.device_id, schedule_ir.subgroup_size,
            schedule_ir.max_shared_memory, schedule_ir.backward_work_assignment,
            schedule_ir.backward_global_merge, schedule_ir.tiled_backward,
            schedule_ir.per_splat_backward, schedule_ir.atomic_free_exact_backward,
            schedule_ir.projection_optimizer_fusion, schedule_ir.subgroup_reordered_sh,
            schedule_ir.command_batch_size, schedule_ir.fenced_execution,
            schedule_ir.streamed_target, schedule_ir.byte_targets,
            schedule_ir.native_float_atomic_add);
    return {adreno_camera_workaround,
            adreno_ssim_channel_split,
            fuse_projection_optimizer,
            use_tiled_backward,
            use_per_splat_backward,
            use_per_splat_pair_backward,
            use_native_float_atomic_add,
            schedule_device,
            schedule_workload};
}

void GaussianForwardProgram::Impl::createBuffers(
        const detail::GaussianBuildOptions& options) {
    const auto& adreno_camera_workaround = options.adreno_camera_workaround;
    const auto& schedule_device = options.schedule_device;
    const auto& schedule_workload = options.schedule_workload;
    const size_t requested_pairs =
            static_cast<size_t>(tile_count) * config.tile_bin_capacity;
    pair_capacity = (requested_pairs + 2047u) / 2048u * 2048u;
    auto camera_desc = BufferDesc(gpu::ScalarType::F32, 4, sizeof(CameraUniform) / 16);
    camera_desc.host_visible = adreno_camera_workaround;
    camera = program.createBuffer(camera_desc);
    if (config.resident_view_capacity > 0) {
        camera_atlas = program.createBuffer(BufferDesc(
                gpu::ScalarType::F32, 4,
                (sizeof(CameraUniform) / 16) * config.resident_view_capacity));
    }
    means = program.createBuffer(
            BufferDesc(gpu::ScalarType::F32, 4, config.gaussian_capacity));
    scales_opacity = program.createBuffer(
            BufferDesc(gpu::ScalarType::F32, 4, config.gaussian_capacity));
    rotations = program.createBuffer(
            BufferDesc(gpu::ScalarType::F32, 4, config.gaussian_capacity));
    sh_dc = program.createBuffer(
            BufferDesc(gpu::ScalarType::F32, 4, config.gaussian_capacity));
    sh_rest = program.createBuffer(
            BufferDesc(gpu::ScalarType::F32, 4,
                       config.gaussian_capacity * (split_sh_rest_storage ? 6u : 12u)));
    if (split_sh_rest_storage) {
        sh_rest_b = program.createBuffer(
                BufferDesc(gpu::ScalarType::F32, 4, config.gaussian_capacity * 6u));
    }
    projected = program.createBuffer(
            BufferDesc(gpu::ScalarType::F32, 4, config.gaussian_capacity));
    conic_opacity = program.createBuffer(
            BufferDesc(gpu::ScalarType::F32, 4, config.gaussian_capacity));
    rgb = program.createBuffer(
            BufferDesc(gpu::ScalarType::F32, 4, config.gaussian_capacity));
    tile_rect = program.createBuffer(
            BufferDesc(gpu::ScalarType::U32, 4, config.gaussian_capacity));
    gaussian_tile_counts = program.createBuffer(
            BufferDesc(gpu::ScalarType::U32, 1, config.gaussian_capacity));
    gaussian_tile_offsets = program.createBuffer(
            BufferDesc(gpu::ScalarType::U32, 1, config.gaussian_capacity));
    tile_counts = program.createBuffer(BufferDesc(gpu::ScalarType::U32, 1, tile_count));
    tile_offsets =
            program.createBuffer(BufferDesc(gpu::ScalarType::U32, 1, tile_count));
    radix_keys_a =
            program.createBuffer(BufferDesc(gpu::ScalarType::U32, 1, pair_capacity));
    radix_values_a =
            program.createBuffer(BufferDesc(gpu::ScalarType::U32, 1, pair_capacity));
    radix_keys_b =
            program.createBuffer(BufferDesc(gpu::ScalarType::U32, 1, pair_capacity));
    radix_values_b =
            program.createBuffer(BufferDesc(gpu::ScalarType::U32, 1, pair_capacity));
    radix_meta = program.createBuffer(BufferDesc(gpu::ScalarType::U32, 4, 4));
    if (config.full_depth_sort) {
        splat_sort_capacity = ((config.gaussian_capacity + 2047) / 2048) * 2048;
        depth_keys_a = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 1, splat_sort_capacity));
        depth_vals_a = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 1, splat_sort_capacity));
        depth_keys_b = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 1, splat_sort_capacity));
        depth_vals_b = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 1, splat_sort_capacity));
        depth_meta = program.createBuffer(BufferDesc(gpu::ScalarType::U32, 4, 4));
        rank_counts = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 1, splat_sort_capacity));
        rank_offsets = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 1, splat_sort_capacity));
    }
    // Indirect dispatch also runs under per-pass fenced submission: the
    // arguments a fenced pass writes are read by a later fenced pass.
    indirect_pairs = program.deviceCaps().dispatch_indirect &&
                     !config.force_direct_pair_dispatch;
    schedule_ir.indirect_pair_dispatch = indirect_pairs;
    if (config.enable_backward) {
        candidate_ledger_json = ScheduleLedgerJson(schedule_ledger, schedule_device,
                                                   schedule_workload, &schedule_ir);
    }
    if (indirect_pairs) {
        pair_dispatch_args = program.createBuffer(BufferDesc(
                gpu::ScalarType::U32, 1, 3, gpu::StorageClass::IndirectBuffer));
    }
    tile_ids = radix_values_a;
    tile_masks =
            program.createBuffer(BufferDesc(gpu::ScalarType::U32, 1, pair_capacity));
    overflow = program.createBuffer(BufferDesc(gpu::ScalarType::U32, 1, 3));
    if (config.enable_mcmc) {
        mcmc_dispatch_args = program.createBuffer(BufferDesc(
                gpu::ScalarType::U32, 1, 11 * 3, gpu::StorageClass::IndirectBuffer));
        mcmc_probabilities = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 1, config.gaussian_capacity));
        mcmc_probability_offsets = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 1, config.gaussian_capacity));
        mcmc_index_map = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 1, config.gaussian_capacity));
        mcmc_clone_counts = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 1, config.gaussian_capacity));
        mcmc_relocation_counts = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 1, config.gaussian_capacity));
    }
    if (gradient_accumulator) {
        // Persistent per-Gaussian accumulator: xy = (sum |dL/d(NDC pos)|,
        // contribution count) for the gradient gate; zw = (sum
        // edge_weight*opacity, view count) for the Edge-Aware Score.
        // MCMC gradient sampling reuses the xy half.
        igs_score = program.createBuffer(
                BufferDesc(gpu::ScalarType::F32, 4, config.gaussian_capacity));
    }
    if (config.enable_igs) {
        // Split-candidate mask and its exclusive-scan offsets, which
        // assign each selected Gaussian a unique appended child slot.
        igs_selected = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 1, config.gaussian_capacity));
        igs_offsets = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 1, config.gaussian_capacity));
        // Recovery-Aware Pruning: 256-bin opacity histogram + the derived
        // prune threshold (opacity below which a Gaussian is culled).
        igs_histogram = program.createBuffer(BufferDesc(gpu::ScalarType::U32, 1, 256));
        igs_prune_threshold =
                program.createBuffer(BufferDesc(gpu::ScalarType::F32, 1, 1));
    }
    auto color_desc = BufferDesc(gpu::ScalarType::F32, 4,
                                 static_cast<size_t>(config.width) * config.height);
    color_desc.host_visible = adreno_camera_workaround;
    color = program.createBuffer(color_desc);
    contributors = program.createBuffer(
            BufferDesc(gpu::ScalarType::U32, 1,
                       static_cast<size_t>(config.width) * config.height));
    if (config.enable_backward) {
        const size_t pixels = static_cast<size_t>(config.width) * config.height;
        target_pixels = pixels;
        const size_t target_views =
                stream_resident_target
                        ? 1
                        : std::max<uint32_t>(config.resident_view_capacity, 1);
        auto target_desc =
                config.byte_targets
                        ? BufferDesc(gpu::ScalarType::U32, 1, pixels * target_views)
                        : BufferDesc(gpu::ScalarType::F32, 4, pixels * target_views);
        // Adreno uses unified memory. Direct host writes between the
        // per-pass fenced submissions avoid a driver fault caused by
        // repeatedly copying into the same device-local target SSBO.
        target_desc.host_visible = stream_resident_target;
        target = program.createBuffer(target_desc);
        loss_value = program.createBuffer(BufferDesc(gpu::ScalarType::F32, 1, pixels));
        if (config.ssim_weight > 0.0f) {
            pixel_grad =
                    program.createBuffer(BufferDesc(gpu::ScalarType::F32, 4, pixels));
            ssim_map = program.createBuffer(
                    BufferDesc(gpu::ScalarType::F32, 4, pixels * 3));
        }
        projected_grad = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 4, config.gaussian_capacity));
        conic_grad = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 4, config.gaussian_capacity));
        rgb_grad = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 4, config.gaussian_capacity));
        if (config.atomic_free_exact_backward) {
            pair_grad = program.createBuffer(
                    BufferDesc(gpu::ScalarType::F32, 4, pair_capacity * 3));
        }
        means_grad = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 4, config.gaussian_capacity));
        scale_grad = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 4, config.gaussian_capacity));
        rotation_grad = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 4, config.gaussian_capacity));
        sh_grad = program.createBuffer(
                BufferDesc(gpu::ScalarType::U32, 4, config.gaussian_capacity));
        sh_rest_grad = program.createBuffer(BufferDesc(
                gpu::ScalarType::U32, 4,
                config.gaussian_capacity * (split_sh_rest_storage ? 6u : 12u)));
        if (split_sh_rest_storage) {
            sh_rest_grad_b = program.createBuffer(
                    BufferDesc(gpu::ScalarType::U32, 4, config.gaussian_capacity * 6u));
        }
        const auto meta_desc = BufferDesc(gpu::ScalarType::U32, 4, 1);
        optimizer_meta = program.createBuffer(meta_desc);
        if (config.enable_optimizer) {
            const auto moment_type = config.fp32_adam_moments ? gpu::ScalarType::F32
                                                              : gpu::ScalarType::U32;
            const uint32_t moment_components = config.fp32_adam_moments ? 4u : 2u;
            means_m = program.createBuffer(BufferDesc(moment_type, moment_components,
                                                      config.gaussian_capacity));
            means_v = program.createBuffer(BufferDesc(moment_type, moment_components,
                                                      config.gaussian_capacity));
            scale_m = program.createBuffer(BufferDesc(moment_type, moment_components,
                                                      config.gaussian_capacity));
            scale_v = program.createBuffer(BufferDesc(moment_type, moment_components,
                                                      config.gaussian_capacity));
            rotation_m = program.createBuffer(BufferDesc(moment_type, moment_components,
                                                         config.gaussian_capacity));
            rotation_v = program.createBuffer(BufferDesc(moment_type, moment_components,
                                                         config.gaussian_capacity));
            sh_m = program.createBuffer(BufferDesc(moment_type, moment_components,
                                                   config.gaussian_capacity));
            sh_v = program.createBuffer(BufferDesc(moment_type, moment_components,
                                                   config.gaussian_capacity));
            const size_t sh_moment_count =
                    config.gaussian_capacity * (split_sh_rest_storage ? 6u : 12u);
            sh_rest_m = program.createBuffer(
                    BufferDesc(moment_type, moment_components, sh_moment_count));
            sh_rest_v = program.createBuffer(
                    BufferDesc(moment_type, moment_components, sh_moment_count));
            if (split_sh_rest_storage) {
                sh_rest_m_b = program.createBuffer(BufferDesc(
                        moment_type, moment_components, config.gaussian_capacity * 6u));
                sh_rest_v_b = program.createBuffer(BufferDesc(
                        moment_type, moment_components, config.gaussian_capacity * 6u));
            }
        }
    }
}

void GaussianForwardProgram::Impl::buildAndInitialize() {
    program.build();
    if (config.enable_backward) {
        const glm::uvec4 meta(0, 0, 1, 0);
        program.upload(optimizer_meta, &meta, sizeof(meta));
    }
    if (config.enable_optimizer) {
        std::vector<glm::uvec2> zeros(config.gaussian_capacity, glm::uvec2(0));
        for (const auto& buffer :
             {means_m, means_v, scale_m, scale_v, rotation_m, rotation_v, sh_m, sh_v}) {
            program.upload(buffer, zeros.data(), zeros.size() * sizeof(glm::uvec2));
        }
        std::vector<glm::uvec2> sh_zeros(
                config.gaussian_capacity * (split_sh_rest_storage ? 6u : 12u),
                glm::uvec2(0));
        for (const auto& buffer : {sh_rest_m, sh_rest_v})
            program.upload(buffer, sh_zeros.data(),
                           sh_zeros.size() * sizeof(glm::uvec2));
        if (split_sh_rest_storage) {
            for (const auto& buffer : {sh_rest_m_b, sh_rest_v_b})
                program.upload(buffer, sh_zeros.data(),
                               sh_zeros.size() * sizeof(glm::uvec2));
        }
    }
    if (gradient_accumulator) {
        std::vector<glm::vec4> score_zeros(config.gaussian_capacity, glm::vec4(0.0f));
        program.upload(igs_score, score_zeros.data(),
                       score_zeros.size() * sizeof(glm::vec4));
    }
    if (config.igs_multistep_update && config.enable_backward) {
        // MU accumulates into the grad buffers, so they must start zeroed.
        std::vector<glm::vec4> grad_zeros(config.gaussian_capacity, glm::vec4(0.0f));
        for (const auto& buffer : {means_grad, scale_grad, rotation_grad, sh_grad}) {
            program.upload(buffer, grad_zeros.data(),
                           grad_zeros.size() * sizeof(glm::vec4));
        }
        std::vector<glm::vec4> sh_grad_zeros(config.gaussian_capacity * 12,
                                             glm::vec4(0.0f));
        program.upload(sh_rest_grad, sh_grad_zeros.data(),
                       sh_grad_zeros.size() * sizeof(glm::vec4));
    }
}

}  // namespace cross3dgs
