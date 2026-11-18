#include "program_impl.h"
#include "shaders.h"

#include <utility>

#include "cross3dgs/gpu/gpu_primitives.h"
#include "cross3dgs/radix_sort.h"

namespace cross3dgs {
using namespace detail;

void GaussianForwardProgram::Impl::addForwardPasses(bool adreno_camera_workaround) {
    if (config.resident_view_capacity > 0) {
        program.addPass(
                {"select_resident_camera",
                 SelectCameraShader(adreno_camera_workaround,
                                    config.randomize_training_views,
                                    config.epoch_shuffle_training_views,
                                    config.training_view_seed),
                 {camera, camera_atlas, optimizer_meta},
                 {{camera, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
                  {camera_atlas,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Read,
                   {}},
                  {optimizer_meta,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Read,
                   {}}},
                 gpu::DispatchDesc::Direct(1)});
    }
    program.addPass(
            {"clear_bins",
             Replace(ClearShader(), "TILE_COUNT", std::to_string(tile_count) + "u"),
             {tile_counts, tile_offsets, overflow},
             {{tile_counts, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
              {tile_offsets, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
              {overflow, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}}},
             gpu::DispatchDesc::Direct((tile_count + 255) / 256)});
    std::vector<gpu::BufferVariable> projection_forward_buffers{
            camera, means,     scales_opacity,      rotations,
            sh_dc,  sh_rest,   projected,           conic_opacity,
            rgb,    tile_rect, gaussian_tile_counts};
    std::vector<gpu::ResourceUse> projection_forward_resources{
            {camera, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
            {means, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
            {scales_opacity, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
            {rotations, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
            {sh_dc, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
            {sh_rest, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
            {projected, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
            {conic_opacity, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
            {rgb, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
            {tile_rect, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
            {gaussian_tile_counts,
             gpu::PipelineStage::Compute,
             gpu::AccessMode::Write,
             {}}};
    if (split_sh_rest_storage) {
        projection_forward_buffers.push_back(sh_rest_b);
        projection_forward_resources.push_back(
                {sh_rest_b, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}});
    }
    // IGS+ Edge-Aware Score: sample the edge weight (target alpha) at each
    // Gaussian's projected pixel and accumulate it into igs_score.zw.
    if (igs_edge_enabled) {
        projection_forward_buffers.push_back(target);
        projection_forward_buffers.push_back(igs_score);
        projection_forward_resources.push_back(
                {target, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}});
        projection_forward_resources.push_back({igs_score,
                                                gpu::PipelineStage::Compute,
                                                gpu::AccessMode::ReadWrite,
                                                {}});
    }
    if (config.enable_mip3d_filter && config.resident_view_capacity > 0) {
        program.addPass(
                {"mip3d_update_filter",
                 Mip3dFilterShader(config.gaussian_capacity, config.mip3d_every,
                                   static_cast<uint32_t>(0.9f * config.max_steps)),
                 {camera_atlas, means, optimizer_meta},
                 {{camera_atlas,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Read,
                   {}},
                  {means, gpu::PipelineStage::Compute, gpu::AccessMode::ReadWrite, {}},
                  {optimizer_meta,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Read,
                   {}}},
                 gpu::DispatchDesc::Direct(static_cast<uint32_t>(
                         (config.gaussian_capacity + 255) / 256))});
    }
    program.addPass({"projection_sh_count",
                     ProjectionShader(config.gaussian_capacity, config.sh_degree,
                                      config.enable_optimizer, reorder_sh,
                                      adreno_camera_workaround, igs_edge_enabled,
                                      config.byte_targets, split_sh_rest_storage,
                                      config.enable_mip3d_filter),
                     std::move(projection_forward_buffers),
                     std::move(projection_forward_resources),
                     gpu::DispatchDesc::Direct(static_cast<uint32_t>(
                             (config.gaussian_capacity + 255) / 256))});
    if (config.full_depth_sort) {
        program.addPass(
                {"emit_depth_keys",
                 EmitDepthKeysShader(config.gaussian_capacity, splat_sort_capacity),
                 {camera, gaussian_tile_counts, projected, depth_keys_a, depth_vals_a},
                 {{camera, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                  {gaussian_tile_counts,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Read,
                   {}},
                  {projected, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                  {depth_keys_a,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Write,
                   {}},
                  {depth_vals_a,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Write,
                   {}}},
                 gpu::DispatchDesc::Direct(
                         static_cast<uint32_t>((splat_sort_capacity + 255) / 256))});
        program.addPass({"prepare_depth_meta",
                         DepthMetaShader(splat_sort_capacity),
                         {depth_meta},
                         {{depth_meta,
                           gpu::PipelineStage::Compute,
                           gpu::AccessMode::Write,
                           {}}},
                         gpu::DispatchDesc::Direct(1)});
        AddRadixSortU32(program, depth_keys_a, depth_vals_a, depth_keys_b, depth_vals_b,
                        depth_meta, splat_sort_capacity, "sort_splat_depth",
                        config.force_portable_radix_sort,
                        config.unsafe_force_subgroup_radix);
        program.addPass(
                {"gather_counts_by_rank",
                 GatherCountsByRankShader(config.gaussian_capacity,
                                          splat_sort_capacity),
                 {camera, depth_vals_a, gaussian_tile_counts, rank_counts},
                 {{camera, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
                  {depth_vals_a,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Read,
                   {}},
                  {gaussian_tile_counts,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Read,
                   {}},
                  {rank_counts,
                   gpu::PipelineStage::Compute,
                   gpu::AccessMode::Write,
                   {}}},
                 gpu::DispatchDesc::Direct(
                         static_cast<uint32_t>((splat_sort_capacity + 255) / 256))});
        gpu::AddExclusiveScanU32(program, rank_counts, rank_offsets,
                                 splat_sort_capacity, "scan_rank_counts");
        program.addPass({"scatter_offsets_by_id",
                         ScatterOffsetsByIdShader(config.gaussian_capacity,
                                                  splat_sort_capacity),
                         {depth_vals_a, rank_offsets, gaussian_tile_offsets},
                         {{depth_vals_a,
                           gpu::PipelineStage::Compute,
                           gpu::AccessMode::Read,
                           {}},
                          {rank_offsets,
                           gpu::PipelineStage::Compute,
                           gpu::AccessMode::Read,
                           {}},
                          {gaussian_tile_offsets,
                           gpu::PipelineStage::Compute,
                           gpu::AccessMode::Write,
                           {}}},
                         gpu::DispatchDesc::Direct(static_cast<uint32_t>(
                                 (splat_sort_capacity + 255) / 256))});
    } else {
        gpu::AddExclusiveScanU32(program, gaussian_tile_counts, gaussian_tile_offsets,
                                 config.gaussian_capacity, "scan_gaussian_tile_counts");
    }
    std::vector<gpu::BufferVariable> prepare_buffers{
            camera, config.full_depth_sort ? rank_counts : gaussian_tile_counts,
            config.full_depth_sort ? rank_offsets : gaussian_tile_offsets, radix_meta,
            overflow};
    std::vector<gpu::ResourceUse> prepare_resources{
            {camera, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
            {config.full_depth_sort ? rank_counts : gaussian_tile_counts,
             gpu::PipelineStage::Compute,
             gpu::AccessMode::Read,
             {}},
            {config.full_depth_sort ? rank_offsets : gaussian_tile_offsets,
             gpu::PipelineStage::Compute,
             gpu::AccessMode::Read,
             {}},
            {radix_meta, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
            {overflow, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}}};
    if (indirect_pairs) {
        prepare_buffers.push_back(pair_dispatch_args);
        prepare_resources.push_back({pair_dispatch_args,
                                     gpu::PipelineStage::Compute,
                                     gpu::AccessMode::Write,
                                     {}});
    }
    program.addPass(
            {"prepare_radix_pairs",
             PreparePairsShader(config.gaussian_capacity, pair_capacity, indirect_pairs,
                                config.full_depth_sort ? splat_sort_capacity : 0),
             std::move(prepare_buffers), std::move(prepare_resources),
             gpu::DispatchDesc::Direct(1)});
    const auto pair_dispatch = [&]() {
        return indirect_pairs ? gpu::DispatchDesc::Indirect(pair_dispatch_args, 0)
                              : gpu::DispatchDesc::Direct(static_cast<uint32_t>(
                                        (pair_capacity + 255) / 256));
    };
    std::vector<gpu::BufferVariable> emit_buffers{
            camera,       tile_rect,
            projected,    config.full_depth_sort ? rank_offsets : gaussian_tile_offsets,
            radix_keys_a, radix_values_a,
            radix_meta};
    std::vector<gpu::ResourceUse> emit_resources{
            {camera, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
            {tile_rect, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
            {projected, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
            {config.full_depth_sort ? rank_offsets : gaussian_tile_offsets,
             gpu::PipelineStage::Compute,
             gpu::AccessMode::Read,
             {}},
            {radix_keys_a, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
            {radix_values_a, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
            {radix_meta, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}}};
    if (config.full_depth_sort) {
        emit_buffers.push_back(depth_vals_a);
        emit_resources.push_back(
                {depth_vals_a, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}});
    }
    program.addPass({"emit_radix_pairs",
                     EmitPairsShader(config.gaussian_capacity, pair_capacity,
                                     depth_bits, config.full_depth_sort),
                     std::move(emit_buffers), std::move(emit_resources),
                     pair_dispatch()});
    AddRadixSortU32(program, radix_keys_a, radix_values_a, radix_keys_b, radix_values_b,
                    radix_meta, pair_capacity, "sort_tile_depth",
                    config.force_portable_radix_sort,
                    config.unsafe_force_subgroup_radix, pair_key_bits);
    program.addPass(
            {"find_tile_starts",
             RangeStartsShader(depth_bits),
             {radix_keys_a, radix_meta, tile_offsets},
             {{radix_keys_a, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {radix_meta, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {tile_offsets, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}}},
             pair_dispatch()});
    program.addPass(
            {"find_tile_ends",
             RangeEndsShader(depth_bits, config.tile_bin_capacity, grid_width),
             {radix_keys_a, radix_meta, tile_offsets, tile_counts, overflow,
              radix_values_a, projected, conic_opacity, tile_masks},
             {{radix_keys_a, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {radix_meta, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {tile_offsets, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {tile_counts, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
              {overflow, gpu::PipelineStage::Compute, gpu::AccessMode::ReadWrite, {}},
              {radix_values_a, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {projected, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {conic_opacity, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {tile_masks, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}}},
             pair_dispatch()});
    program.addPass(
            {"tile_raster",
             RasterShader(config.tile_bin_capacity),
             {camera, projected, conic_opacity, rgb, tile_counts, tile_ids, color,
              contributors, tile_masks, tile_offsets},
             {{camera, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {projected, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {conic_opacity, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {rgb, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {tile_counts, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {tile_ids, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {color, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
              {contributors, gpu::PipelineStage::Compute, gpu::AccessMode::Write, {}},
              {tile_masks, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}},
              {tile_offsets, gpu::PipelineStage::Compute, gpu::AccessMode::Read, {}}},
             gpu::DispatchDesc::Direct(tile_count)});
}

}  // namespace cross3dgs
