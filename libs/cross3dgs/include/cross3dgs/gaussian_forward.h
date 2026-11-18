#ifndef CROSS3DGS_GAUSSIAN_FORWARD_H
#define CROSS3DGS_GAUSSIAN_FORWARD_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "cross3dgs/dataset.h"
#include "cross3dgs/gaussian_state.h"
#include "cross3dgs/gpu/gpu_program.h"
#include "cross3dgs/training_preview.h"

namespace cross3dgs {

// Gaussian storage is allocated in whole blocks of 32 so the reordered
// SH-rest layout, which interleaves blocks of 32 Gaussians, applies at every
// capacity. GaussianForwardProgram rounds the requested gaussian_capacity up
// to a multiple of this; callers that record the capacity should record
// AlignGaussianCapacity(requested), the capacity that was allocated.
inline constexpr size_t kGaussianCapacityAlignment = 32;
inline constexpr size_t AlignGaussianCapacity(size_t requested) {
    return (requested + kGaussianCapacityAlignment - 1) /
           kGaussianCapacityAlignment * kGaussianCapacityAlignment;
}

struct GaussianForwardConfig {
    uint32_t width = 0;
    uint32_t height = 0;
    size_t gaussian_capacity = 0;
    uint32_t tile_size = 16;
    uint32_t tile_bin_capacity = 2048;
    bool enable_backward = false;
    bool enable_optimizer = false;
    uint32_t resident_view_capacity = 0;
    // Standard 3DGS-style seeded random-with-replacement training-camera
    // sampling. False preserves the historical cyclic resident-view order.
    bool randomize_training_views = false;
    bool epoch_shuffle_training_views = false;
    uint32_t training_view_seed = 42;
    uint32_t sh_degree = 0;
    float means_lr = 0.00016f;
    float means_lr_final = 0.0000016f;
    uint32_t max_steps = 30000;
    float scales_lr = 0.005f;
    // 0 disables the scale-group exponential decay (constant scale LR, the
    // classic behavior). A positive value enables an on-GPU exponential
    // decay of the scale learning rate from scales_lr to scales_lr_final,
    // mirroring the means decay. Used by the IGS+ two-stage scale schedule.
    float scales_lr_final = 0.0f;
    float opacity_lr = 0.05f;
    float rotation_lr = 0.001f;
    float sh_dc_lr = 0.0025f;
    float sh_rest_lr = 0.000125f;
    float l1_weight = 1.0f;
    float ssim_weight = 0.0f;
    // Compute the DSSIM term as a zero-padded full-image mean (fused-ssim /
    // Brush semantics) and drop the composited-pixel gradient gate. False
    // keeps the legacy 5-px-cropped "valid" DSSIM mean with the gate.
    bool full_image_ssim = false;
    bool enable_mcmc = false;
    uint32_t mcmc_refine_start = 500;
    uint32_t mcmc_refine_stop = 25000;
    uint32_t mcmc_refine_every = 100;
    float mcmc_grow_factor = 1.05f;
    float mcmc_min_opacity = 0.005f;
    float mcmc_noise_lr = 500000.0f;
    // Offset each relocated/appended clone's mean by a covariance-shaped
    // normal sample scaled by this factor (0 disables). The Langevin noise
    // gate only moves near-transparent Gaussians, so clones of opaque
    // sources otherwise land exactly on the source and never separate.
    float mcmc_clone_offset = 0.0f;
    // Subtract mcmc_opacity_decay*(1-step/max_steps) from every active
    // Gaussian's opacity at each refine step (Brush-style continuous
    // replacement for the classic opacity reset). 0 disables.
    float mcmc_opacity_decay = 0.0f;
    // Bias MCMC clone-source sampling toward Gaussians whose screen-space
    // gradient is large, instead of sampling on opacity alone. Weight becomes
    //   opacity * (1 + w * min(mean|dL/d(NDC pos)| / reference, 8))
    // so w=0 reproduces the classic 3DGS-MCMC opacity-proportional sampling.
    // Brush and IGS+ both select densification candidates by gradient; plain
    // MCMC clones well-converged Gaussians regardless of whether their region
    // still needs detail. Enabling this turns on the per-Gaussian gradient
    // accumulator that IGS+ also uses.
    float mcmc_gradient_sampling = 0.0f;
    float mcmc_gradient_reference = 0.0002f;
    // Step at which the opacity decay stops; 0 follows mcmc_refine_stop.
    // Sustained late decay makes the optimizer trade opacity for scale, which
    // inflates screen footprints and with them the tile pairs and raster
    // work, so stopping it early is a speed knob.
    uint32_t mcmc_opacity_decay_stop = 0;
    // Mip-Splatting 3D low-pass filter (Brush trains with it always on): a
    // per-splat filter std-dev f = sqrt(0.1)*min_v(depth/focal_px) stored in
    // means.w (otherwise unused; incompatible with enable_igs whose split
    // pass rewrites means.w), refreshed every mip3d_every steps until 90% of
    // max_steps, and folded into projection as scale'^2 = s^2 + f^2 with
    // opacity compensation sqrt(det(s^2)/det(s^2+f^2)). The filter update
    // pass needs the resident camera atlas; without resident views the
    // stored means.w (0 after upload) leaves the filter inert.
    bool enable_mip3d_filter = false;
    uint32_t mip3d_every = 100;
    // Order tile pairs by full-precision f32 view depth (Brush scheme):
    // argsort visible splats by depth once, then stable-sort pairs by the
    // tile id alone, so LSD stability preserves the global depth order.
    // False keeps the legacy packed (tile << bits | quantized-depth) key,
    // whose ~20 depth bits tie-break by splat id.
    bool full_depth_sort = false;
    float scale_regularizer = 0.0f;
    float opacity_regularizer = 0.0f;
    // Use full FP32 Adam first/second moments instead of the compact BF16
    // storage path. Intended for quality attribution and memory permitting.
    bool fp32_adam_moments = false;
    // Store the 12 vec4 SH-rest coefficients as two fixed six-vec4 shards.
    // This is also enabled automatically when the monolithic buffer exceeds
    // maxStorageBufferRange but one shard fits.
    bool force_split_sh_rest_storage = false;
    // ImprovedGS+ (IGS+, arXiv:2603.08661) gradient-driven Long-Axis-Split
    // densification. Mutually exclusive with enable_mcmc; requires the
    // optimizer. A persistent per-Gaussian screen-space gradient accumulator
    // (NDC convention) selects split candidates every igs_refine_every steps
    // within (igs_refine_start, igs_refine_stop); each selected Gaussian is
    // split symmetrically along its longest principal axis.
    bool enable_igs = false;
    uint32_t igs_refine_start = 500;
    uint32_t igs_refine_stop = 15000;
    uint32_t igs_refine_every = 500;
    // Average |dL/d(NDC screen position)| threshold selecting split
    // candidates (classic 3DGS convention; 0.0002 is the 3DGS default,
    // IGS+/ImprovedGS report 0.0003 on a comparable convention).
    float igs_grad_threshold = 0.0002f;
    // Long-Axis-Split factors: longest-axis shrink (alpha), secondary-axis
    // shrink (gamma), opacity retention (beta). Offset magnitude along the
    // long axis equals long-axis std-dev * alpha.
    float igs_las_long = 0.5f;
    float igs_las_short = 0.85f;
    float igs_las_opacity = 0.6f;
    // Edge-Aware Score (Phase C). A Laplacian edge map per training image is
    // carried in the target alpha channel; the projection forward accumulates
    // the view-averaged edge weight * opacity per Gaussian. A candidate must
    // additionally see an average edge weight above this threshold, so splits
    // concentrate on geometric boundaries instead of drifting onto planar
    // surfaces. 0 disables the edge gate (pure gradient selection, Phase B).
    float igs_edge_threshold = 0.1f;
    // The first igs_edge_warmup_steps refine steps select on the edge score
    // alone (the gradient gate is suspended) so broad, low-gradient surfaces
    // are populated before strict refinement begins.
    uint32_t igs_edge_warmup_steps = 2;
    // Growth Control (ImprovedGS). Caps the active count to a sqrt budget
    // N_max*sqrt((step-start)/(stop-start)) across the densification window so
    // the model grows gradually instead of saturating capacity early. N_max is
    // the gaussian_capacity.
    bool igs_growth_control = true;
    // Recovery-Aware Pruning (ImprovedGS): at each listed step, cull the
    // lowest-opacity fraction of Gaussians (soft prune). 0 disables it.
    // Defaults off: without a full SoA compaction the culled slots are not
    // reclaimed under Growth Control, so on bicycle it was quality-neutral to
    // slightly negative. Kept behind a flag for experimentation.
    float igs_prune_fraction = 0.0f;
    uint32_t igs_prune_step_a = 3000;
    uint32_t igs_prune_step_b = 6000;
    // Multi-step Update (ImprovedGS): after densification ends the optimizer
    // accumulates gradients over N views per Adam step (N=1 during
    // densification, igs_mu_n1 until igs_mu_stage2, then igs_mu_n2) to reduce
    // gradient noise in refinement. Forces the split (non-fused) backward+Adam
    // path since the fused path applies Adam every step. Default off: it trades
    // the fused fast-path speed for a refinement-only effect.
    bool igs_multistep_update = false;
    uint32_t igs_mu_stage2_step = 22500;
    uint32_t igs_mu_n1 = 5;
    uint32_t igs_mu_n2 = 20;
    // 0 selects the exact scatter backward (native float atomic or CAS).
    // A positive value selects a Gaussian-owned Sobol gather with this many
    // pixel samples per Gaussian, avoiding all gradient atomics.
    uint32_t stochastic_backward_samples = 0;
    // Enumerate every pixel per tile/Gaussian pair, then reduce pair partials
    // once per Gaussian. This is an atomic-free exact alternative for drivers
    // where the exact scatter CAS loop is unreliable.
    bool atomic_free_exact_backward = false;
    // Experiment-only schedule overrides. These select existing
    // semantics-preserving lowerings without changing the mathematical
    // training contract; every selected value is serialized in schedule_ir.
    bool force_pixel_subgroup_backward = false;
    bool force_per_splat_backward = false;
    bool force_tiled_backward = false;
    bool force_cas_scatter_backward = false;
    bool force_streamed_target = false;
    bool force_portable_radix_sort = false;
    bool force_split_projection_optimizer = false;
    bool force_direct_pair_dispatch = false;
    // Timestamp instrumentation is useful for probes but excluded from
    // production time-to-quality measurements.
    bool enable_pass_profiling = true;
    // Insert a full compute+transfer barrier before every generated pass in
    // the batched command buffer. A conservative schedule; defaults off.
    bool conservative_barriers = false;
    // Run one vkQueueSubmit + fence per pass instead of one batched submit.
    // Required on drivers (Adreno) where a large multi-pass submit hangs even
    // with full barriers. Auto-enabled for Qualcomm; force here for testing.
    bool fenced_execution = false;
    // Store resident targets as packed RGBA8 instead of RGBA32F. For 8-bit
    // image sources this is bitwise exact (unpackUnorm4x8 matches the CPU
    // byte/255.0f conversion) and shrinks target memory 4x; targets that are
    // not 8-bit quantized would be rounded, so this defaults to off.
    bool byte_targets = false;
    // Bounded failure-probe overrides. These re-enable candidates that
    // the vendor evidence normally rejects (batched submission and the
    // subgroup-dependent radix on Qualcomm) so a minimal witness can be
    // captured at most once per device. Never set them for training runs;
    // the resulting schedule may hang or corrupt on the probed device.
    bool unsafe_force_batched_submission = false;
    bool unsafe_force_subgroup_radix = false;
    // Allows an explicitly requested rejected candidate only for a bounded
    // failure-witness probe. Training must leave this false.
    bool unsafe_allow_rejected_schedule = false;
    // Negative control: deliberately drops the first tile-Gaussian
    // partial per Gaussian in the atomic-free fixed-order reduce so the
    // one-step conformance gate can demonstrate a caught violation. It does
    // not change the selected schedule tuple. Probe-only; training and
    // conformance runs must leave this false.
    bool unsafe_negative_control_drop_partial = false;
    // Split-axis negative control: with the 6+6 SH-rest split active, the
    // projection backward drops every shard-B gradient write (the upper SH
    // band keeps its zero-initialized gradient), modeling a lost shard across
    // the split boundary. Only takes effect when split SH-rest storage is
    // selected; it does not change the schedule tuple. Probe-only;
    // training and conformance runs must leave this false.
    bool unsafe_negative_control_zero_sh_rest_shard_b = false;
};

struct GaussianForwardStats {
    // Pairs lost in the most recent update only; reset at every update.
    uint32_t overflow_count = 0;
    // Over the program's whole life: updates that lost pairs, and the
    // largest pair count any update asked for.
    uint32_t overflow_update_count = 0;
    uint32_t peak_pair_count = 0;
    uint32_t max_tile_count = 0;
    uint64_t tile_reference_count = 0;
    // Exact-gather backward stores three vec4 partials per tile/Gaussian
    // pair. The allocation is sized from pair_capacity, whereas the logical
    // bytes used by the current view follow tile_reference_count.
    size_t pair_capacity = 0;
    size_t atomic_free_pair_scratch_bytes = 0;
    size_t atomic_free_pair_scratch_used_bytes = 0;
    size_t current_memory_bytes = 0;
    size_t peak_memory_bytes = 0;
    std::vector<gpu::PassTiming> timings;
    std::vector<gpu::HostPassTiming> fenced_host_timings;
};

// Device-selected lowering parameters. These describe execution only; changing
// them must not change GaussianSplat's differentiable math.
struct GaussianScheduleIR {
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint32_t subgroup_size = 0;
    uint32_t max_shared_memory = 0;
    uint32_t max_storage_buffer_bindings = 0;
    uint64_t max_storage_buffer_range = 0;
    uint32_t tile_size = 16;
    uint32_t subtile_size = 8;
    uint32_t raster_workgroup_size = 256;
    uint32_t backward_workgroup_size = 32;
    uint32_t projection_optimizer_workgroup_size = 32;
    uint32_t command_batch_size = 1;
    // Canonical backward axes. Legacy booleans below remain serialized for
    // compatibility with the v1 artifact readers.
    std::string backward_work_assignment = "disabled";
    std::string backward_global_merge = "none";
    std::string backward_tile_shape = "none";
    std::string backward_selection_policy = "fixed_capability_rule";
    std::string schedule_candidate_id;
    std::string policy_selected_candidate_id;
    std::string schedule_catalog_version;
    std::string schedule_catalog_sha256;
    std::string schedule_policy_id;
    std::string schedule_policy_sha256;
    std::string schedule_selection_source = "runtime_candidate_ledger";
    bool schedule_candidate_accepted = false;
    bool tiled_backward = false;
    bool per_splat_backward = false;
    bool projection_optimizer_fusion = false;
    bool subgroup_reordered_sh = false;
    bool split_sh_rest_storage = false;
    bool mcmc_indirect_dispatch = false;
    bool native_float_atomic_add = false;
    bool f16_storage = false;
    bool byte_targets = false;
    // Per-pass command-buffer construction (Qualcomm) vs batched-8. The
    // production Qualcomm path may submit the recorded buffers together and
    // wait on one fence; this flag does not count host queue submissions.
    bool fenced_execution = false;
    // Gaussian-owned atomic-free exact gather backward vs scatter.
    bool atomic_free_exact_backward = false;
    // One-view streamed target upload (descriptor-range limited devices)
    // vs fully resident multi-view targets.
    bool streamed_target = false;
    bool portable_radix_sort = false;
    bool indirect_pair_dispatch = false;
};

struct GaussianGradientCpu {
    std::vector<glm::vec4> means;
    std::vector<glm::vec4> scales_opacity;
    std::vector<glm::vec4> rotations;
    std::vector<glm::vec4> sh_dc;
    std::vector<glm::vec4> sh_rest;
};

class GaussianForwardProgram {
public:
    explicit GaussianForwardProgram(const GaussianForwardConfig& config,
                                    gpu::VkContextPtr context = nullptr);
    ~GaussianForwardProgram();
    GaussianForwardProgram(const GaussianForwardProgram&) = delete;
    GaussianForwardProgram& operator=(const GaussianForwardProgram&) = delete;

    void uploadState(const GaussianStateCpu& state);
    void uploadTarget(const FloatImage& target);
    void uploadTrainingSet(const std::vector<Camera>& cameras,
                           const std::vector<FloatImage>& targets);
    // Streaming variant: load_target(view) is called once per camera in
    // order, so only one decoded view needs to be resident on the host.
    void uploadTrainingSet(
            const std::vector<Camera>& cameras,
            const std::function<FloatImage(size_t)>& load_target);
    void setCamera(const Camera& camera);
    void execute();
    void executeBatch(uint32_t repeat_count);
    // Diagnostic: returns the label of the first training pass whose GPU work
    // does not finish within `timeout_ns`, or an empty string if all complete.
    std::string firstHangingPass(uint64_t timeout_ns);
    FloatImage downloadColor();
    GaussianGradientCpu downloadGradients();
    GaussianStateCpu downloadState();
    // Diagnostic: raw optimizer_meta uvec4 (step, live count, view count, _).
    glm::uvec4 debugMeta();
    // Diagnostic/test hook for deterministic progressive-SH validation.
    // Production training advances this value on GPU instead.
    void debugSetStep(uint32_t step);
    // Diagnostic: (tile_coverage_lo, tile_coverage_hi, total_pairs, _) from the
    // last forward — projection coverage sum and the scan's pair total.
    glm::uvec4 debugForwardCounts();
    // Diagnostic: (pair count, key order errors, key/value errors,
    // tile-rectangle membership errors) after radix sorting.
    glm::uvec4 debugPairIntegrity();
    float downloadMeanL1();
    // The photo, render and per-pixel loss of the last training step. The photo
    // comes from the GPU's own target, so random view orders need no host copy.
    TrainingPreview downloadTrainingPreview();
    GaussianForwardStats stats();
    GaussianScheduleIR schedule() const;
    std::string candidateLedgerJson() const;
    gpu::DeviceCaps deviceCaps() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace cross3dgs

#endif
