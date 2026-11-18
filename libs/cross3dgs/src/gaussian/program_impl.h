#pragma once

#include "cross3dgs/gaussian_forward.h"
#include "cross3dgs/schedule_ledger.h"

namespace cross3dgs {

namespace detail {
struct alignas(16) CameraUniform {
    glm::mat4 world_to_camera{1.0f};
    glm::vec4 intrinsics{};
    glm::uvec4 dims{};
    glm::uvec4 state{};
};

// Construction-only decisions shared by allocation and pass registration.
// They are discarded after build; the runtime state below is unchanged.
struct GaussianBuildOptions {
    bool adreno_camera_workaround;
    bool adreno_ssim_channel_split;
    bool fuse_projection_optimizer;
    bool use_tiled_backward;
    bool use_per_splat_backward;
    bool use_per_splat_pair_backward;
    bool use_native_float_atomic_add;
    ScheduleDeviceEvidence schedule_device;
    ScheduleLedgerWorkload schedule_workload;
};
}  // namespace detail

// Private PImpl shared only by this library's implementation files.
struct GaussianForwardProgram::Impl {
    explicit Impl(const GaussianForwardConfig& input_config, gpu::VkContextPtr context);

    detail::GaussianBuildOptions selectSchedule(
            const GaussianForwardConfig& input_config);
    void createBuffers(const detail::GaussianBuildOptions& options);
    void addGradientClear(bool fuse_projection_optimizer);
    void addForwardPasses(bool adreno_camera_workaround);
    void addBackwardPasses(const detail::GaussianBuildOptions& options);
    void addOptimizerPasses(bool fuse_projection_optimizer);
    void addMcmcPasses();
    void addIgsPasses();
    void buildAndInitialize();

    GaussianForwardConfig config;
    gpu::CustomGpuProgram program;
    GaussianScheduleIR schedule_ir{};
    ScheduleCandidateLedger schedule_ledger{};
    std::string selected_candidate_id;
    bool selected_candidate_accepted = false;
    std::string candidate_ledger_json;
    uint32_t grid_width = 0;
    uint32_t grid_height = 0;
    uint32_t tile_count = 0;
    uint32_t depth_bits = 0;
    uint32_t pair_key_bits = 32;
    size_t splat_sort_capacity = 0;
    size_t pair_capacity = 0;
    uint32_t gaussian_count = 0;
    bool reorder_sh = false;
    bool split_sh_rest_storage = false;
    bool indirect_pairs = false;
    bool use_fenced_execution = false;
    bool use_host_step_fill = false;
    bool stream_resident_target = false;
    bool igs_edge_enabled = false;
    // The per-Gaussian screen-gradient accumulator serves both IGS+ split
    // selection and MCMC gradient-weighted clone sampling.
    bool gradient_accumulator = false;
    size_t target_pixels = 0;
    size_t resident_target_stride = 0;
    std::vector<std::byte> resident_target_bytes;
    uint32_t host_optimizer_step = 0;
    uint32_t resident_view_count = 0;
    gpu::BufferVariable camera, camera_atlas;
    gpu::BufferVariable means, scales_opacity, rotations, sh_dc, sh_rest;
    gpu::BufferVariable sh_rest_b;
    gpu::BufferVariable projected, conic_opacity, rgb, tile_rect;
    gpu::BufferVariable gaussian_tile_counts, gaussian_tile_offsets;
    gpu::BufferVariable tile_counts, tile_offsets;
    gpu::BufferVariable radix_keys_a, radix_values_a;
    gpu::BufferVariable radix_keys_b, radix_values_b, radix_meta;
    gpu::BufferVariable depth_keys_a, depth_vals_a;
    gpu::BufferVariable depth_keys_b, depth_vals_b, depth_meta;
    gpu::BufferVariable rank_counts, rank_offsets;
    gpu::BufferVariable pair_dispatch_args;
    gpu::BufferVariable tile_ids, tile_masks;
    gpu::BufferVariable overflow, color, contributors;
    gpu::BufferVariable mcmc_dispatch_args;
    gpu::BufferVariable mcmc_probabilities, mcmc_probability_offsets;
    gpu::BufferVariable mcmc_index_map, mcmc_clone_counts;
    gpu::BufferVariable mcmc_relocation_counts;
    gpu::BufferVariable igs_score, igs_selected, igs_offsets;
    gpu::BufferVariable igs_histogram, igs_prune_threshold;
    gpu::BufferVariable target, loss_value, pixel_grad, ssim_map;
    gpu::BufferVariable projected_grad, conic_grad, rgb_grad;
    gpu::BufferVariable pair_grad;
    gpu::BufferVariable means_grad, scale_grad, rotation_grad, sh_grad;
    gpu::BufferVariable sh_rest_grad, sh_rest_grad_b;
    gpu::BufferVariable optimizer_meta;
    gpu::BufferVariable means_m, means_v, scale_m, scale_v;
    gpu::BufferVariable rotation_m, rotation_v, sh_m, sh_v;
    gpu::BufferVariable sh_rest_m, sh_rest_v, sh_rest_m_b, sh_rest_v_b;
};

}  // namespace cross3dgs
