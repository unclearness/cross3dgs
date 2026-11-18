#include "program_impl.h"
#include "shaders.h"

#include "cross3dgs/gpu/gpu_primitives.h"

namespace cross3dgs {
using namespace detail;

void GaussianForwardProgram::Impl::addMcmcPasses() {
    const uint32_t groups =
            static_cast<uint32_t>((config.gaussian_capacity + 255) / 256);
    const uint32_t groups_64 =
            static_cast<uint32_t>((config.gaussian_capacity + 63) / 64);
    if (schedule_ir.mcmc_indirect_dispatch) {
        program.addPass(
                {"mcmc_prepare_dispatch",
                 McmcDispatchShader(config.gaussian_capacity, config.mcmc_refine_start,
                                    config.mcmc_refine_stop, config.mcmc_refine_every),
                 {optimizer_meta, mcmc_dispatch_args},
                 {{optimizer_meta,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Read,
                   {}},
                  {mcmc_dispatch_args,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Write,
                   {}}},
                 gpu::DispatchDesc::Direct(1)});
    }
    uint32_t mcmc_command = 0;
    const auto mcmc_dispatch = [&](uint32_t direct_groups) {
        if (schedule_ir.mcmc_indirect_dispatch) {
            return gpu::DispatchDesc::Indirect(mcmc_dispatch_args,
                                               3 * sizeof(uint32_t) * mcmc_command++);
        }
        return gpu::DispatchDesc::Direct(direct_groups);
    };
    program.addPass(
            {"mcmc_prepare_probabilities",
             McmcPrepareShader(config.gaussian_capacity, config.mcmc_refine_start,
                               config.mcmc_refine_stop, config.mcmc_refine_every,
                               config.mcmc_min_opacity, config.mcmc_gradient_sampling,
                               config.mcmc_gradient_reference),
             [&]() {
                 std::vector<gpu::BufferVariable> buffers{
                         scales_opacity, mcmc_probabilities, mcmc_clone_counts,
                         mcmc_relocation_counts, optimizer_meta};
                 if (config.mcmc_gradient_sampling > 0.0f) buffers.push_back(igs_score);
                 return buffers;
             }(),
             [&]() {
                 std::vector<gpu::ResourceUse> resources{{scales_opacity,
                                                          gpu::PipelineStage::Compute,
                                                          gpu::AccessMode::Read,
                                                          {}},
                                                         {mcmc_probabilities,
                                                          gpu::PipelineStage::Compute,
                                                          gpu::AccessMode::Write,
                                                          {}},
                                                         {mcmc_clone_counts,
                                                          gpu::PipelineStage::Compute,
                                                          gpu::AccessMode::Write,
                                                          {}},
                                                         {mcmc_relocation_counts,
                                                          gpu::PipelineStage::Compute,
                                                          gpu::AccessMode::Write,
                                                          {}},
                                                         {optimizer_meta,
                                                          gpu::PipelineStage::Compute,
                                                          gpu::AccessMode::Read,
                                                          {}}};
                 if (config.mcmc_gradient_sampling > 0.0f)
                     resources.push_back({igs_score,
                                          gpu::PipelineStage::Compute,
                                          gpu::AccessMode::Read,
                                          {}});
                 return resources;
             }(),
             mcmc_dispatch(groups)});
    if (schedule_ir.mcmc_indirect_dispatch) {
        gpu::AddExclusiveScanU32(program, mcmc_probabilities, mcmc_probability_offsets,
                                 config.gaussian_capacity, "mcmc_scan_probabilities",
                                 mcmc_dispatch_args, &mcmc_command);
    } else {
        gpu::AddExclusiveScanU32(program, mcmc_probabilities, mcmc_probability_offsets,
                                 config.gaussian_capacity, "mcmc_scan_probabilities");
    }
    program.addPass(
            {"mcmc_build_index_map",
             McmcMapShader(config.gaussian_capacity, config.mcmc_refine_start,
                           config.mcmc_refine_stop, config.mcmc_refine_every,
                           config.mcmc_grow_factor),
             {mcmc_probabilities, mcmc_probability_offsets, mcmc_index_map,
              mcmc_clone_counts, mcmc_relocation_counts, optimizer_meta},
             {{mcmc_probabilities,
               gpu::PipelineStage::Compute,
               gpu::AccessMode::Read,
               {}},
              {mcmc_probability_offsets,
               gpu::PipelineStage::Compute,
               gpu::AccessMode::Read,
               {}},
              {mcmc_index_map, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
              {mcmc_clone_counts,
               gpu::PipelineStage::Compute,
               gpu::AccessMode::ReadWrite,
               {}},
              {mcmc_relocation_counts,
               gpu::PipelineStage::Compute,
               gpu::AccessMode::ReadWrite,
               {}},
              {optimizer_meta, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}}},
             mcmc_dispatch(groups)});
    program.addPass(
            {"mcmc_relocate_sources",
             McmcRelocateSourceShader(config.gaussian_capacity,
                                      config.mcmc_min_opacity),
             {mcmc_clone_counts, scales_opacity, optimizer_meta},
             {{mcmc_clone_counts,
               gpu::PipelineStage::Compute,
               gpu::AccessMode::Read,
               {}},
              {scales_opacity,
               gpu::PipelineStage::Compute,
               gpu::AccessMode::ReadWrite,
               {}},
              {optimizer_meta, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}}},
             mcmc_dispatch(groups_64)});
    program.addPass({"mcmc_copy_relocated_and_added",
                     McmcCopyShader(config.gaussian_capacity, config.mcmc_refine_start,
                                    config.mcmc_refine_stop, config.mcmc_refine_every,
                                    config.mcmc_grow_factor, reorder_sh,
                                    split_sh_rest_storage, config.mcmc_clone_offset),
                     [&]() {
                         std::vector<gpu::BufferVariable> buffers{
                                 mcmc_index_map, means,   scales_opacity, rotations,
                                 sh_dc,          sh_rest, optimizer_meta};
                         if (split_sh_rest_storage) buffers.push_back(sh_rest_b);
                         return buffers;
                     }(),
                     [&]() {
                         std::vector<gpu::ResourceUse> resources{
                                 {mcmc_index_map,
                                  gpu::PipelineStage::Compute,
                                  gpu::AccessMode::Read,
                                  {}},
                                 {means,
                                  gpu::PipelineStage::Compute,
                                  gpu::AccessMode::ReadWrite,
                                  {}},
                                 {scales_opacity,
                                  gpu::PipelineStage::Compute,
                                  gpu::AccessMode::ReadWrite,
                                  {}},
                                 {rotations,
                                  gpu::PipelineStage::Compute,
                                  gpu::AccessMode::ReadWrite,
                                  {}},
                                 {sh_dc,
                                  gpu::PipelineStage::Compute,
                                  gpu::AccessMode::ReadWrite,
                                  {}},
                                 {sh_rest,
                                  gpu::PipelineStage::Compute,
                                  gpu::AccessMode::ReadWrite,
                                  {}},
                                 {optimizer_meta,
                                  gpu::PipelineStage::Compute,
                                  gpu::AccessMode::Read,
                                  {}}};
                         if (split_sh_rest_storage)
                             resources.push_back({sh_rest_b,
                                                  gpu::PipelineStage::Compute,
                                                  gpu::AccessMode::ReadWrite,
                                                  {}});
                         return resources;
                     }(),
                     mcmc_dispatch(groups_64)});
    program.addPass(
            {"mcmc_reset_optimizer_state",
             McmcResetMomentsShader(config.gaussian_capacity, config.mcmc_refine_start,
                                    config.mcmc_refine_stop, config.mcmc_refine_every,
                                    config.mcmc_grow_factor, reorder_sh,
                                    config.fp32_adam_moments, split_sh_rest_storage),
             [&]() {
                 std::vector<gpu::BufferVariable> buffers{mcmc_relocation_counts,
                                                          means_m,
                                                          means_v,
                                                          scale_m,
                                                          scale_v,
                                                          rotation_m,
                                                          rotation_v,
                                                          sh_m,
                                                          sh_v,
                                                          sh_rest_m,
                                                          sh_rest_v,
                                                          optimizer_meta};
                 if (split_sh_rest_storage) {
                     buffers.push_back(sh_rest_m_b);
                     buffers.push_back(sh_rest_v_b);
                 }
                 return buffers;
             }(),
             [&]() {
                 std::vector<gpu::ResourceUse> resources{{mcmc_relocation_counts,
                                                          gpu::PipelineStage::Compute,
                                                          gpu::AccessMode::Read,
                                                          {}},
                                                         {means_m,
                                                          gpu::PipelineStage::Compute,
                                                          gpu::AccessMode::Write,
                                                          {}},
                                                         {means_v,
                                                          gpu::PipelineStage::Compute,
                                                          gpu::AccessMode::Write,
                                                          {}},
                                                         {scale_m,
                                                          gpu::PipelineStage::Compute,
                                                          gpu::AccessMode::Write,
                                                          {}},
                                                         {scale_v,
                                                          gpu::PipelineStage::Compute,
                                                          gpu::AccessMode::Write,
                                                          {}},
                                                         {rotation_m,
                                                          gpu::PipelineStage::Compute,
                                                          gpu::AccessMode::Write,
                                                          {}},
                                                         {rotation_v,
                                                          gpu::PipelineStage::Compute,
                                                          gpu::AccessMode::Write,
                                                          {}},
                                                         {sh_m,
                                                          gpu::PipelineStage::Compute,
                                                          gpu::AccessMode::Write,
                                                          {}},
                                                         {sh_v,
                                                          gpu::PipelineStage::Compute,
                                                          gpu::AccessMode::Write,
                                                          {}},
                                                         {sh_rest_m,
                                                          gpu::PipelineStage::Compute,
                                                          gpu::AccessMode::Write,
                                                          {}},
                                                         {sh_rest_v,
                                                          gpu::PipelineStage::Compute,
                                                          gpu::AccessMode::Write,
                                                          {}},
                                                         {optimizer_meta,
                                                          gpu::PipelineStage::Compute,
                                                          gpu::AccessMode::Read,
                                                          {}}};
                 if (split_sh_rest_storage) {
                     resources.push_back({sh_rest_m_b,
                                          gpu::PipelineStage::Compute,
                                          gpu::AccessMode::Write,
                                          {}});
                     resources.push_back({sh_rest_v_b,
                                          gpu::PipelineStage::Compute,
                                          gpu::AccessMode::Write,
                                          {}});
                 }
                 return resources;
             }(),
             mcmc_dispatch(groups_64)});
    program.addPass(
            {"mcmc_update_active_count",
             McmcFinalizeShader(config.gaussian_capacity, config.mcmc_refine_start,
                                config.mcmc_refine_stop, config.mcmc_refine_every,
                                config.mcmc_grow_factor),
             {optimizer_meta},
             {{optimizer_meta,
               gpu::PipelineStage::Compute,
               gpu::AccessMode::ReadWrite,
               {}}},
             mcmc_dispatch(1)});
    if (config.mcmc_opacity_decay > 0.0f) {
        program.addPass({"mcmc_opacity_decay",
                         McmcOpacityDecayShader(
                                 config.gaussian_capacity, config.mcmc_refine_start,
                                 config.mcmc_opacity_decay_stop > 0
                                         ? config.mcmc_opacity_decay_stop
                                         : config.mcmc_refine_stop,
                                 config.mcmc_refine_every, config.mcmc_opacity_decay,
                                 config.max_steps),
                         {scales_opacity, optimizer_meta},
                         {{scales_opacity,
                           gpu::PipelineStage::Compute,
                           gpu::AccessMode::ReadWrite,
                           {}},
                          {optimizer_meta,
                           gpu::PipelineStage::Compute,
                           gpu::AccessMode::Read,
                           {}}},
                         gpu::DispatchDesc::Direct(groups)});
    }
    if (config.mcmc_gradient_sampling > 0.0f) {
        // Clear the screen-gradient accumulator so the next
        // refine window samples on fresh statistics.
        program.addPass(
                {"mcmc_reset_gradient_accumulator",
                 IgsResetAccumulatorShader(
                         config.gaussian_capacity, config.mcmc_refine_start,
                         config.mcmc_refine_stop, config.mcmc_refine_every),
                 {igs_score, optimizer_meta},
                 {{igs_score, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
                  {optimizer_meta,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Read,
                   {}}},
                 gpu::DispatchDesc::Direct(groups)});
    }
    program.addPass(
            {"mcmc_inject_noise",
             McmcNoiseShader(config.gaussian_capacity, config.mcmc_noise_lr,
                             config.means_lr, config.means_lr_final, config.max_steps),
             {means, scales_opacity, rotations, optimizer_meta},
             {{means, gpu::PipelineStage::Compute, gpu::AccessMode::ReadWrite, {}},
              {scales_opacity, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {rotations, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {optimizer_meta, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}}},
             gpu::DispatchDesc::Direct(groups)});
}

void GaussianForwardProgram::Impl::addIgsPasses() {
    const uint32_t igs_groups_256 =
            static_cast<uint32_t>((config.gaussian_capacity + 255) / 256);
    const uint32_t igs_groups_64 =
            static_cast<uint32_t>((config.gaussian_capacity + 63) / 64);
    // 1. Mark split candidates from the accumulated gradient.
    program.addPass(
            {"igs_mark_candidates",
             IgsMarkShader(config.gaussian_capacity, config.igs_refine_start,
                           config.igs_refine_stop, config.igs_refine_every,
                           config.igs_grad_threshold,
                           igs_edge_enabled ? config.igs_edge_threshold : -1.0f,
                           igs_edge_enabled ? config.igs_edge_warmup_steps : 0u),
             {igs_score, igs_selected, optimizer_meta},
             {{igs_score, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {igs_selected, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
              {optimizer_meta, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}}},
             gpu::DispatchDesc::Direct(igs_groups_256)});
    // 2. Exclusive scan -> unique appended slot per candidate.
    gpu::AddExclusiveScanU32(program, igs_selected, igs_offsets,
                             config.gaussian_capacity, "igs_scan_candidates");
    // 3. Long-Axis-Split each candidate into two children.
    program.addPass(
            {"igs_long_axis_split",
             IgsSplitShader(config.gaussian_capacity, config.igs_refine_start,
                            config.igs_refine_stop, config.igs_refine_every,
                            config.igs_las_long, config.igs_las_short,
                            config.igs_las_opacity, reorder_sh,
                            config.igs_growth_control),
             {igs_selected, igs_offsets, means, scales_opacity, rotations, sh_dc,
              sh_rest, optimizer_meta},
             {{igs_selected, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {igs_offsets, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {means, gpu::PipelineStage::Compute, gpu::AccessMode::ReadWrite, {}},
              {scales_opacity,
               gpu::PipelineStage::Compute,
               gpu::AccessMode::ReadWrite,
               {}},
              {rotations, gpu::PipelineStage::Compute, gpu::AccessMode::ReadWrite, {}},
              {sh_dc, gpu::PipelineStage::Compute, gpu::AccessMode::ReadWrite, {}},
              {sh_rest, gpu::PipelineStage::Compute, gpu::AccessMode::ReadWrite, {}},
              {optimizer_meta, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}}},
             gpu::DispatchDesc::Direct(igs_groups_64)});
    // 4. Reset Adam moments for split parents and new children.
    program.addPass(
            {"igs_reset_optimizer_state",
             IgsResetMomentsShader(config.gaussian_capacity, config.igs_refine_start,
                                   config.igs_refine_stop, config.igs_refine_every,
                                   reorder_sh, config.igs_growth_control),
             {igs_selected, igs_offsets, means_m, means_v, scale_m, scale_v, rotation_m,
              rotation_v, sh_m, sh_v, sh_rest_m, sh_rest_v, optimizer_meta},
             {{igs_selected, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {igs_offsets, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {means_m, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
              {means_v, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
              {scale_m, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
              {scale_v, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
              {rotation_m, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
              {rotation_v, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
              {sh_m, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
              {sh_v, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
              {sh_rest_m, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
              {sh_rest_v, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
              {optimizer_meta, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}}},
             gpu::DispatchDesc::Direct(igs_groups_64)});
    // 5. Grow the active count by the number of new children.
    program.addPass(
            {"igs_update_active_count",
             IgsFinalizeShader(config.gaussian_capacity, config.igs_refine_start,
                               config.igs_refine_stop, config.igs_refine_every,
                               config.igs_growth_control),
             {igs_selected, igs_offsets, optimizer_meta},
             {{igs_selected, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {igs_offsets, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {optimizer_meta,
               gpu::PipelineStage::Compute,
               gpu::AccessMode::ReadWrite,
               {}}},
             gpu::DispatchDesc::Direct(1)});
    // 6. Clear the accumulator for the next refine window.
    program.addPass(
            {"igs_reset_accumulator",
             IgsResetAccumulatorShader(config.gaussian_capacity,
                                       config.igs_refine_start, config.igs_refine_stop,
                                       config.igs_refine_every),
             {igs_score, optimizer_meta},
             {{igs_score, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
              {optimizer_meta, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}}},
             gpu::DispatchDesc::Direct(igs_groups_256)});
    // Recovery-Aware Pruning (soft): opacity histogram ->
    // percentile threshold -> cull the lowest-opacity fraction.
    if (config.igs_prune_fraction > 0.0f) {
        program.addPass(
                {"igs_rap_hist_zero",
                 IgsRapHistZeroShader(config.igs_prune_step_a, config.igs_prune_step_b),
                 {igs_histogram, optimizer_meta},
                 {{igs_histogram,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Write,
                   {}},
                  {optimizer_meta,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Read,
                   {}}},
                 gpu::DispatchDesc::Direct(1)});
        program.addPass({"igs_rap_hist_build",
                         IgsRapHistBuildShader(config.gaussian_capacity,
                                               config.igs_prune_step_a,
                                               config.igs_prune_step_b),
                         {scales_opacity, igs_histogram, optimizer_meta},
                         {{scales_opacity,
                           gpu::PipelineStage::Compute,
                           gpu::AccessMode::Read,
                           {}},
                          {igs_histogram,
                           gpu::PipelineStage::Compute,
                           gpu::AccessMode::ReadWrite,
                           {}},
                          {optimizer_meta,
                           gpu::PipelineStage::Compute,
                           gpu::AccessMode::Read,
                           {}}},
                         gpu::DispatchDesc::Direct(igs_groups_256)});
        program.addPass(
                {"igs_rap_threshold",
                 IgsRapThresholdShader(config.igs_prune_step_a, config.igs_prune_step_b,
                                       config.igs_prune_fraction),
                 {igs_histogram, igs_prune_threshold, optimizer_meta},
                 {{igs_histogram,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Read,
                   {}},
                  {igs_prune_threshold,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Write,
                   {}},
                  {optimizer_meta,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Read,
                   {}}},
                 gpu::DispatchDesc::Direct(1)});
        program.addPass(
                {"igs_rap_prune",
                 IgsRapPruneShader(config.gaussian_capacity, config.igs_prune_step_a,
                                   config.igs_prune_step_b),
                 {scales_opacity, igs_prune_threshold, optimizer_meta},
                 {{scales_opacity,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::ReadWrite,
                   {}},
                  {igs_prune_threshold,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Read,
                   {}},
                  {optimizer_meta,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Read,
                   {}}},
                 gpu::DispatchDesc::Direct(igs_groups_256)});
    }
}

}  // namespace cross3dgs
