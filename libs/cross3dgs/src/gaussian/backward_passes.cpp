#include "program_impl.h"
#include "shaders.h"

#include <utility>

namespace cross3dgs {
using namespace detail;

void GaussianForwardProgram::Impl::addGradientClear(bool fuse_projection_optimizer) {
    std::vector<gpu::BufferVariable> clear_buffers{projected_grad, conic_grad, rgb_grad,
                                                   loss_value};
    if (!fuse_projection_optimizer) {
        clear_buffers.insert(
                clear_buffers.end(),
                {means_grad, scale_grad, rotation_grad, sh_grad, sh_rest_grad});
        if (split_sh_rest_storage) {
            clear_buffers.push_back(sh_rest_grad_b);
        }
    }
    std::vector<gpu::ResourceUse> clear_resources;
    clear_resources.reserve(clear_buffers.size());
    for (const auto& buffer : clear_buffers) {
        clear_resources.push_back(
                {buffer, gpu::PipelineStage::Transfer, gpu::AccessMode::Write, {}});
    }
    gpu::ComputePassDesc clear_pass;
    clear_pass.label = "clear_gradients";
    clear_pass.buffers = std::move(clear_buffers);
    clear_pass.resources = std::move(clear_resources);
    clear_pass.fill_zero = true;
    program.addPass(clear_pass);
}

void GaussianForwardProgram::Impl::addBackwardPasses(
        const detail::GaussianBuildOptions& options) {
    const bool adreno_camera_workaround = options.adreno_camera_workaround;
    const bool adreno_ssim_channel_split = options.adreno_ssim_channel_split;
    const bool fuse_projection_optimizer = options.fuse_projection_optimizer;
    const bool use_tiled_backward = options.use_tiled_backward;
    const bool use_per_splat_backward = options.use_per_splat_backward;
    const bool use_per_splat_pair_backward = options.use_per_splat_pair_backward;
    const bool use_native_float_atomic_add = options.use_native_float_atomic_add;
    const uint32_t pixel_count = config.width * config.height;
    // Zero the gradient/loss buffers with vkCmdFillBuffer rather than a
    // large trivial memset dispatch. The scatter backward accumulates
    // into projected/conic/rgb grad, so those and the per-pixel loss
    // always need clearing; the split (non-fused) path also clears the
    // parameter gradients that its Adam passes read for inactive
    // Gaussians. Whole-buffer fill matches each buffer's logical size.
    if (!adreno_camera_workaround) addGradientClear(fuse_projection_optimizer);
    if (config.stochastic_backward_samples == 0) {
        uint32_t ssim_dispatch_z = 1u;
        if (adreno_ssim_channel_split) ssim_dispatch_z = 3u;
        if (config.ssim_weight > 0.0f) {
            program.addPass(
                    {"ssim_forward",
                     adreno_ssim_channel_split
                             ? AdrenoSeparableSsimForwardShader(config.byte_targets,
                                                                stream_resident_target,
                                                                target_pixels, true)
                             : LegacySeparableSsimForwardShader(config.byte_targets,
                                                                stream_resident_target,
                                                                target_pixels),
                     {camera, color, target, ssim_map, optimizer_meta},
                     {{camera, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                      {color, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                      {target, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                      {ssim_map,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Write,
                       {}},
                      {optimizer_meta,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}}},
                     gpu::DispatchDesc::Direct((config.width + 15) / 16,
                                               (config.height + 15) / 16,
                                               ssim_dispatch_z)});
            program.addPass(
                    {"ssim_backward",
                     adreno_ssim_channel_split
                             ? AdrenoSeparableSsimBackwardShader(
                                       config.l1_weight, config.ssim_weight,
                                       config.byte_targets, stream_resident_target,
                                       target_pixels, config.full_image_ssim, true)
                             : LegacySeparableSsimBackwardShader(
                                       config.l1_weight, config.ssim_weight,
                                       config.byte_targets, stream_resident_target,
                                       target_pixels, config.full_image_ssim),
                     {camera, color, target, ssim_map, pixel_grad, loss_value,
                      optimizer_meta},
                     {{camera, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                      {color, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                      {target, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                      {ssim_map,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {pixel_grad,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Write,
                       {}},
                      {loss_value,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Write,
                       {}},
                      {optimizer_meta,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}}},
                     gpu::DispatchDesc::Direct((config.width + 15) / 16,
                                               (config.height + 15) / 16,
                                               ssim_dispatch_z)});
        }
        const auto& backward_target = config.ssim_weight > 0.0f ? pixel_grad : target;
        if (config.atomic_free_exact_backward) {
            if (config.ssim_weight == 0.0f) {
                program.addPass(
                        {"l1_loss_exact_gather",
                         StochasticLossShader(config.byte_targets,
                                              stream_resident_target, target_pixels),
                         {camera, color, target, loss_value, optimizer_meta},
                         {{camera,
                           gpu::PipelineStage::Compute,
                           gpu::AccessMode::Read,
                           {}},
                          {color,
                           gpu::PipelineStage::Compute,
                           gpu::AccessMode::Read,
                           {}},
                          {target,
                           gpu::PipelineStage::Compute,
                           gpu::AccessMode::Read,
                           {}},
                          {loss_value,
                           gpu::PipelineStage::Compute,
                           gpu::AccessMode::Write,
                           {}},
                          {optimizer_meta,
                           gpu::PipelineStage::Compute,
                           gpu::AccessMode::Read,
                           {}}},
                         gpu::DispatchDesc::Direct((pixel_count + 255) / 256)});
            }
            const uint32_t tile_workgroups = std::min<uint32_t>(tile_count, 65535);
            const uint32_t gaussian_workgroups = static_cast<uint32_t>(
                    std::min<size_t>((config.gaussian_capacity + 63) / 64, 65535));
            program.addPass(
                    {use_per_splat_pair_backward ? "raster_backward_per_splat_pairs"
                                                 : "raster_backward_exact_tiles",
                     use_per_splat_pair_backward
                             ? PerSplatRasterBackwardShader(
                                       false, config.ssim_weight > 0.0f,
                                       config.byte_targets, stream_resident_target,
                                       target_pixels, true)
                             : ExactTileRasterBackwardShader(
                                       tile_count, tile_workgroups,
                                       program.deviceCaps().subgroup_size,
                                       config.byte_targets, stream_resident_target,
                                       target_pixels, config.ssim_weight > 0.0f),
                     {camera, projected, conic_opacity, rgb, tile_counts, tile_ids,
                      color, backward_target, pair_grad, tile_masks, tile_offsets,
                      contributors, gaussian_tile_offsets, tile_rect},
                     {{camera, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                      {projected,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {conic_opacity,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {rgb, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                      {tile_counts,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {tile_ids,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {color, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                      {backward_target,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {pair_grad,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Write,
                       {}},
                      {tile_masks,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {tile_offsets,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {contributors,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {gaussian_tile_offsets,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {tile_rect,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}}},
                     gpu::DispatchDesc::Direct(use_per_splat_pair_backward
                                                       ? tile_count
                                                       : tile_workgroups),
                     gpu::ShaderOptimization::Throughput});
            program.addPass(
                    {"raster_backward_exact_reduce",
                     ExactPairReduceShader(gaussian_workgroups,
                                           config.unsafe_negative_control_drop_partial),
                     {camera, gaussian_tile_counts, gaussian_tile_offsets, radix_meta,
                      pair_grad, projected_grad, conic_grad, rgb_grad},
                     {{camera, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                      {gaussian_tile_counts,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {gaussian_tile_offsets,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {radix_meta,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {pair_grad,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {projected_grad,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Write,
                       {}},
                      {conic_grad,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Write,
                       {}},
                      {rgb_grad,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Write,
                       {}}},
                     gpu::DispatchDesc::Direct(gaussian_workgroups),
                     gpu::ShaderOptimization::Throughput});
        } else {
            program.addPass(
                    {use_tiled_backward      ? "raster_backward_tiled_8x8"
                     : use_per_splat_backward ? "raster_backward_per_splat"
                                              : "raster_l1_backward",
                     use_tiled_backward
                             ? TiledRasterBackwardShader(
                                       use_native_float_atomic_add,
                                       config.ssim_weight > 0.0f, config.byte_targets,
                                       stream_resident_target, target_pixels)
                     : use_per_splat_backward
                             ? PerSplatRasterBackwardShader(
                                       use_native_float_atomic_add,
                                       config.ssim_weight > 0.0f, config.byte_targets,
                                       stream_resident_target, target_pixels)
                             : RasterBackwardShader(
                                       config.tile_bin_capacity,
                                       use_native_float_atomic_add,
                                       config.ssim_weight > 0.0f, config.byte_targets,
                                       stream_resident_target, target_pixels),
                     {camera, projected, conic_opacity, rgb, tile_counts, tile_ids,
                      color, backward_target, projected_grad, conic_grad, rgb_grad,
                      loss_value, optimizer_meta, tile_masks, tile_offsets,
                      contributors},
                     {{camera, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                      {projected,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {conic_opacity,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {rgb, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                      {tile_counts,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {tile_ids,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {color, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                      {backward_target,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {projected_grad,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::ReadWrite,
                       {}},
                      {conic_grad,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::ReadWrite,
                       {}},
                      {rgb_grad,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::ReadWrite,
                       {}},
                      {loss_value,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Write,
                       {}},
                      {optimizer_meta,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {tile_masks,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {tile_offsets,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}},
                      {contributors,
                       gpu::PipelineStage::Compute,
                       gpu::AccessMode::Read,
                       {}}},
                     gpu::DispatchDesc::Direct(tile_count),
                     gpu::ShaderOptimization::Throughput});
        }
    } else {
        program.addPass(
                {"l1_loss",
                 StochasticLossShader(config.byte_targets, stream_resident_target,
                                      target_pixels),
                 {camera, color, target, loss_value, optimizer_meta},
                 {{camera, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                  {color, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                  {target, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                  {loss_value, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
                  {optimizer_meta,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Read,
                   {}}},
                 gpu::DispatchDesc::Direct((pixel_count + 255) / 256)});
        program.addPass(
                {"raster_l1_backward_sobol_k" +
                         std::to_string(config.stochastic_backward_samples),
                 StochasticRasterBackwardShader(
                         config.tile_bin_capacity, config.stochastic_backward_samples,
                         config.byte_targets, stream_resident_target, target_pixels),
                 {camera, projected, conic_opacity, rgb, tile_counts, tile_ids, color,
                  target, projected_grad, conic_grad, rgb_grad, optimizer_meta,
                  tile_masks, tile_offsets, contributors},
                 {{camera, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                  {projected, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                  {conic_opacity,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Read,
                   {}},
                  {rgb, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                  {tile_counts, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                  {tile_ids, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                  {color, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                  {target, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                  {projected_grad,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Write,
                   {}},
                  {conic_grad, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
                  {rgb_grad, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
                  {optimizer_meta,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Read,
                   {}},
                  {tile_masks, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                  {tile_offsets,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Read,
                   {}},
                  {contributors,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Read,
                   {}}},
                 gpu::DispatchDesc::Direct(static_cast<uint32_t>(
                         (config.gaussian_capacity + 255) / 256))});
    }
    std::vector<gpu::BufferVariable> projection_buffers{
            camera,     means,          scales_opacity, rotations,   sh_dc,
            sh_rest,    projected_grad, conic_grad,     rgb_grad,    means_grad,
            scale_grad, rotation_grad,  sh_grad,        sh_rest_grad};
    const auto parameter_access = fuse_projection_optimizer ? gpu::AccessMode::ReadWrite
                                                            : gpu::AccessMode::Read;
    std::vector<gpu::ResourceUse> projection_resources{
            {camera, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
            {means, gpu::PipelineStage::Compute, parameter_access, {}},
            {scales_opacity, gpu::PipelineStage::Compute, parameter_access, {}},
            {rotations, gpu::PipelineStage::Compute, parameter_access, {}},
            {sh_dc, gpu::PipelineStage::Compute, parameter_access, {}},
            {sh_rest, gpu::PipelineStage::Compute, parameter_access, {}},
            {projected_grad, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
            {conic_grad, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
            {rgb_grad, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
            {means_grad, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
            {scale_grad, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
            {rotation_grad, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
            {sh_grad, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
            {sh_rest_grad, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}}};
    if (split_sh_rest_storage) {
        projection_buffers.push_back(sh_rest_b);
        projection_buffers.push_back(sh_rest_grad_b);
        projection_resources.push_back(
                {sh_rest_b, gpu::PipelineStage::Compute, parameter_access, {}});
        projection_resources.push_back({sh_rest_grad_b,
                                        gpu::PipelineStage::Compute,
                                        gpu::AccessMode::Write,
                                        {}});
    }
    if (fuse_projection_optimizer) {
        for (const auto& buffer : {means_m, means_v, scale_m, scale_v, rotation_m,
                                   rotation_v, sh_m, sh_v, sh_rest_m, sh_rest_v}) {
            projection_buffers.push_back(buffer);
            projection_resources.push_back({buffer,
                                            gpu::PipelineStage::Compute,
                                            gpu::AccessMode::ReadWrite,
                                            {}});
        }
        projection_buffers.push_back(optimizer_meta);
        projection_resources.push_back({optimizer_meta,
                                        gpu::PipelineStage::Compute,
                                        gpu::AccessMode::Read,
                                        {}});
    }
    // IGS+ screen-gradient accumulator is the last binding (14 in the
    // split backward, 25 in the fused backward+Adam variant).
    if (gradient_accumulator) {
        projection_buffers.push_back(igs_score);
        projection_resources.push_back({igs_score,
                                        gpu::PipelineStage::Compute,
                                        gpu::AccessMode::ReadWrite,
                                        {}});
    }
    program.addPass(
            {fuse_projection_optimizer ? "projection_backward_adam_fused"
                                       : "projection_backward",
             ProjectionBackwardShader(
                     config.sh_degree, fuse_projection_optimizer ? &config : nullptr,
                     reorder_sh, gradient_accumulator, config.igs_multistep_update,
                     split_sh_rest_storage, config.enable_mip3d_filter,
                     config.unsafe_negative_control_zero_sh_rest_shard_b),
             std::move(projection_buffers), std::move(projection_resources),
             gpu::DispatchDesc::Direct(
                     static_cast<uint32_t>((config.gaussian_capacity +
                                            (fuse_projection_optimizer ? 31 : 255)) /
                                           (fuse_projection_optimizer ? 32 : 256)))});
}

}  // namespace cross3dgs
