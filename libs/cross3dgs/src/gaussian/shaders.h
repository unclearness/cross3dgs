#pragma once

#include "cross3dgs/gaussian_forward.h"

// Private shader-source builders. Raw GLSL and substitution order are
// deliberately preserved across the source-file split.
namespace cross3dgs::detail {

std::string ProjectionShader(size_t gaussian_capacity, uint32_t sh_degree,
                             bool training, bool reorder_sh, bool flatten_camera_block,
                             bool enable_igs_edge = false, bool byte_targets = false,
                             bool split_sh_rest = false, bool enable_mip3d = false);

std::string RasterShader(uint32_t tile_capacity);

std::string ClearShader();

std::string EmitDepthKeysShader(size_t gaussian_capacity, size_t padded_capacity);

std::string DepthMetaShader(size_t padded_capacity);

std::string GatherCountsByRankShader(size_t gaussian_capacity, size_t padded_capacity);

std::string ScatterOffsetsByIdShader(size_t gaussian_capacity, size_t padded_capacity);

std::string PreparePairsShader(size_t gaussian_capacity, size_t pair_capacity,
                               bool indirect_pairs, size_t rank_padded_capacity = 0);

std::string EmitPairsShader(size_t gaussian_capacity, size_t pair_capacity,
                            uint32_t depth_bits, bool depth_ordered = false);

std::string RangeStartsShader(uint32_t depth_bits);

std::string RangeEndsShader(uint32_t depth_bits, uint32_t tile_capacity,
                            uint32_t grid_width);

std::string SelectCameraShader(bool explicit_copy, bool randomize_views,
                               bool epoch_shuffle_views, uint32_t view_seed);

std::string SsimForwardShader();

std::string SsimBackwardShader(float l1_weight, float ssim_weight);

std::string LegacySeparableSsimForwardShader(bool byte_targets,
                                             bool stream_target = false,
                                             size_t target_pixels = 0);

std::string LegacySeparableSsimBackwardShader(float l1_weight, float ssim_weight,
                                              bool byte_targets,
                                              bool stream_target = false,
                                              size_t target_pixels = 0,
                                              bool full_image_ssim = false);

std::string AdrenoSeparableSsimForwardShader(bool byte_targets,
                                             bool stream_target = false,
                                             size_t target_pixels = 0,
                                             bool adreno_channel_split = false);

std::string AdrenoSeparableSsimBackwardShader(float l1_weight, float ssim_weight,
                                              bool byte_targets,
                                              bool stream_target = false,
                                              size_t target_pixels = 0,
                                              bool full_image_ssim = false,
                                              bool adreno_channel_split = false);

std::string RasterBackwardShader(uint32_t tile_capacity, bool native_float_atomic_add,
                                 bool external_pixel_gradient = false,
                                 bool byte_targets = false, bool stream_target = false,
                                 size_t target_pixels = 0);

std::string PerSplatRasterBackwardShader(bool native_float_atomic_add,
                                         bool external_pixel_gradient,
                                         bool byte_targets, bool stream_target = false,
                                         size_t target_pixels = 0,
                                         bool pair_output = false);

std::string TiledRasterBackwardShader(bool native_float_atomic_add,
                                       bool external_pixel_gradient, bool byte_targets,
                                       bool stream_target = false,
                                       size_t target_pixels = 0);

std::string StochasticLossShader(bool byte_targets, bool stream_target = false,
                                 size_t target_pixels = 0);

std::string StochasticRasterBackwardShader(uint32_t tile_capacity, uint32_t samples,
                                           bool byte_targets,
                                           bool stream_target = false,
                                           size_t target_pixels = 0);

std::string ExactTileRasterBackwardShader(uint32_t tile_count, uint32_t tile_workgroups,
                                          uint32_t subgroup_size, bool byte_targets,
                                          bool stream_target, size_t target_pixels,
                                          bool external_pixel_gradient);

std::string ExactPairReduceShader(uint32_t gaussian_workgroups,
                                  bool drop_first_partial = false);

std::string ProjectionBackwardShader(uint32_t sh_degree,
                                     const GaussianForwardConfig* optimizer,
                                     bool reorder_sh, bool enable_igs = false,
                                     bool multistep_update = false,
                                     bool split_sh_rest = false,
                                     bool enable_mip3d = false,
                                     bool zero_sh_rest_shard_b = false);

std::string AdamShader(uint32_t group, float learning_rate_xyz, float learning_rate_w,
                       float regularizer_xyz, float regularizer_w,
                       float learning_rate_final = 0.0f, uint32_t max_steps = 30000,
                       bool multistep = false, uint32_t mu_start = 0,
                       uint32_t mu_stage2 = 0, uint32_t mu_n1 = 1, uint32_t mu_n2 = 1,
                       bool fp32_moments = false);

std::string IncrementStepShader(bool scalar_meta_store);

std::string McmcPrepareShader(size_t capacity, uint32_t start, uint32_t stop,
                              uint32_t every, float min_opacity,
                              float gradient_sampling = 0.0f,
                              float gradient_reference = 0.0002f);

std::string McmcDispatchShader(size_t capacity, uint32_t start, uint32_t stop,
                               uint32_t every);

std::string McmcMapShader(size_t capacity, uint32_t start, uint32_t stop,
                          uint32_t every, float grow_factor);

std::string McmcRelocateSourceShader(size_t capacity, float min_opacity);

std::string McmcCopyShader(size_t capacity, uint32_t start, uint32_t stop,
                           uint32_t every, float grow_factor, bool reorder_sh,
                           bool split_sh_rest, float clone_offset);

std::string McmcFinalizeShader(size_t capacity, uint32_t start, uint32_t stop,
                               uint32_t every, float grow_factor);

std::string McmcOpacityDecayShader(size_t capacity, uint32_t start, uint32_t stop,
                                   uint32_t every, float decay, uint32_t max_steps);

std::string McmcNoiseShader(size_t capacity, float noise_lr, float means_lr,
                            float means_lr_final, uint32_t max_steps);

std::string McmcResetMomentsShader(size_t capacity, uint32_t start, uint32_t stop,
                                   uint32_t every, float grow_factor, bool reorder_sh,
                                   bool fp32_moments, bool split_sh_rest);

std::string IgsMarkShader(size_t capacity, uint32_t start, uint32_t stop,
                          uint32_t every, float grad_threshold, float edge_threshold,
                          uint32_t warmup_steps);

std::string GrowthControlCapExpr(size_t capacity, bool growth_control,
                                 uint32_t gc_start, uint32_t gc_stop);

std::string IgsSplitShader(size_t capacity, uint32_t start, uint32_t stop,
                           uint32_t every, float alpha, float gamma, float beta,
                           bool reorder_sh, bool growth_control);

std::string IgsResetMomentsShader(size_t capacity, uint32_t start, uint32_t stop,
                                  uint32_t every, bool reorder_sh, bool growth_control);

std::string IgsFinalizeShader(size_t capacity, uint32_t start, uint32_t stop,
                              uint32_t every, bool growth_control);

std::string IgsResetAccumulatorShader(size_t capacity, uint32_t start, uint32_t stop,
                                      uint32_t every);

std::string IgsRapPredicate();

std::string IgsRapHistZeroShader(uint32_t step_a, uint32_t step_b);

std::string IgsRapHistBuildShader(size_t capacity, uint32_t step_a, uint32_t step_b);

std::string IgsRapThresholdShader(uint32_t step_a, uint32_t step_b, float fraction);

std::string IgsRapPruneShader(size_t capacity, uint32_t step_a, uint32_t step_b);

std::string Mip3dFilterShader(size_t capacity, uint32_t every, uint32_t freeze_step);

std::string Replace(std::string source, const std::string& token,
                    const std::string& value);

std::string ApplyTargetStorage(std::string source, bool packed, bool streamed = false,
                               size_t target_pixels = 0);

}  // namespace cross3dgs::detail
