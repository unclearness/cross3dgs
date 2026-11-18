#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "cross3dgs/gaussian_forward.h"

int main() {
    using namespace cross3dgs;
    constexpr uint32_t kWidth = 64;
    constexpr uint32_t kHeight = 60;
    constexpr float kShC0 = 0.28209479177387814f;

    GaussianStateCpu state;
    state.means.emplace_back(0.0f, 0.0f, 3.0f, 0.0f);
    state.scales_opacity.emplace_back(0.2f, 0.2f, 0.2f, 0.8f);
    state.rotations.emplace_back(1.0f, 0.0f, 0.0f, 0.0f);
    state.sh_dc.emplace_back((glm::vec3(1.0f, 0.25f, 0.1f) - 0.5f) / kShC0,
                             0.0f);
    state.sh_rest.assign(12, glm::vec4(0.0f));

    Camera camera;
    camera.width = kWidth;
    camera.height = kHeight;
    camera.fx = 50.0f;
    camera.fy = 50.0f;
    camera.cx = 32.0f;
    camera.cy = 30.0f;
    camera.model = CameraModel::Pinhole;

    GaussianForwardProgram forward(
            {.width = kWidth,
             .height = kHeight,
             .gaussian_capacity = state.size(),
             .tile_size = 16,
             .tile_bin_capacity = 32});
    forward.uploadState(state);
    forward.setCamera(camera);
    forward.execute();

    const auto image = forward.downloadColor();
    const auto stats = forward.stats();
    if (image.pixels.size() !=
        static_cast<size_t>(kWidth) * kHeight * 4 ||
        stats.overflow_count != 0 || stats.timings.empty()) {
        std::cerr << "forward output shape/stats mismatch\n";
        return 1;
    }
    const size_t center =
            (static_cast<size_t>(kHeight / 2) * kWidth + kWidth / 2) * 4;
    if (!std::isfinite(image.pixels[center]) ||
        image.pixels[center] <= image.pixels[center + 1] ||
        image.pixels[center] < 0.5f || image.pixels[center + 3] < 0.5f) {
        std::cerr << "center Gaussian was not rasterized as expected: "
                  << image.pixels[center] << ", "
                  << image.pixels[center + 1] << ", "
                  << image.pixels[center + 2] << ", "
                  << image.pixels[center + 3] << "\n";
        return 1;
    }
    std::cout << "fixed-count Gaussian forward validated\n";

    FloatImage target;
    target.width = kWidth;
    target.height = kHeight;
    target.pixels.assign(static_cast<size_t>(kWidth) * kHeight * 4, 0.0f);
    state.means[0].x = 0.13f;
    state.scales_opacity[0] = glm::vec4(0.16f, 0.23f, 0.19f, 0.72f);
    state.rotations[0] =
            glm::normalize(glm::vec4(0.91f, 0.17f, -0.23f, 0.29f));
    GaussianForwardProgram differentiable(
            {.width = kWidth,
             .height = kHeight,
             .gaussian_capacity = state.size(),
             .tile_size = 16,
             .tile_bin_capacity = 32,
             .enable_backward = true});
    differentiable.uploadState(state);
    differentiable.uploadTarget(target);
    differentiable.setCamera(camera);
    differentiable.execute();
    const float loss = differentiable.downloadMeanL1();
    const auto gradients = differentiable.downloadGradients();
    if (!(loss > 0.0f) || gradients.means.size() != 1 ||
        !std::isfinite(gradients.scales_opacity[0].w) ||
        gradients.scales_opacity[0].w <= 0.0f ||
        gradients.sh_dc[0].x <= 0.0f) {
        std::cerr << "L1 backward did not produce finite positive gradients\n";
        return 1;
    }

    GaussianForwardProgram exact_gather(
            {.width = kWidth,
             .height = kHeight,
             .gaussian_capacity = state.size(),
             .tile_size = 16,
             .tile_bin_capacity = 32,
             .enable_backward = true,
             .atomic_free_exact_backward = true});
    exact_gather.uploadState(state);
    exact_gather.uploadTarget(target);
    exact_gather.setCamera(camera);
    exact_gather.execute();
    const auto exact_gather_gradients = exact_gather.downloadGradients();
    bool found_exact_gather_pass = false;
    for (const auto& timing : exact_gather.stats().timings) {
        found_exact_gather_pass |=
                timing.label == "raster_backward_exact_reduce";
    }
    float exact_gather_max_error = 0.0f;
    const auto compare_vec4 = [&](const glm::vec4& expected,
                                  const glm::vec4& actual) {
        for (int component = 0; component < 4; component++) {
            exact_gather_max_error =
                    std::max(exact_gather_max_error,
                             std::abs(expected[component] -
                                      actual[component]));
        }
    };
    compare_vec4(gradients.means[0], exact_gather_gradients.means[0]);
    compare_vec4(gradients.scales_opacity[0],
                 exact_gather_gradients.scales_opacity[0]);
    compare_vec4(gradients.rotations[0],
                 exact_gather_gradients.rotations[0]);
    compare_vec4(gradients.sh_dc[0], exact_gather_gradients.sh_dc[0]);
    if (!found_exact_gather_pass ||
        !std::isfinite(exact_gather_max_error) ||
        exact_gather_max_error > 1.0e-5f) {
        std::cerr << "Atomic-free exact gather differs from exact scatter: "
                  << exact_gather_max_error << "\n";
        return 1;
    }

    const auto run_scatter_candidate = [&](bool per_splat, bool cas) {
        GaussianForwardProgram candidate(
                {.width = kWidth,
                 .height = kHeight,
                 .gaussian_capacity = state.size(),
                 .tile_size = 16,
                 .tile_bin_capacity = 32,
                 .enable_backward = true,
                 .force_pixel_subgroup_backward = !per_splat,
                 .force_per_splat_backward = per_splat,
                 .force_cas_scatter_backward = cas});
        candidate.uploadState(state);
        candidate.uploadTarget(target);
        candidate.setCamera(camera);
        candidate.execute();
        return std::pair{candidate.downloadGradients(), candidate.schedule()};
    };
    const auto [pixel_gradients, pixel_schedule] =
            run_scatter_candidate(false, false);
    const auto [per_splat_gradients, per_splat_schedule] =
            run_scatter_candidate(true, false);
    const auto [per_splat_cas_gradients, per_splat_cas_schedule] =
            run_scatter_candidate(true, true);
    const auto [per_splat_pair_gradients, per_splat_pair_schedule] = [&] {
        GaussianForwardProgram candidate(
                {.width = kWidth,
                 .height = kHeight,
                 .gaussian_capacity = state.size(),
                 .tile_size = 16,
                 .tile_bin_capacity = 32,
                 .enable_backward = true,
                 .atomic_free_exact_backward = true,
                 .force_per_splat_backward = true});
        candidate.uploadState(state);
        candidate.uploadTarget(target);
        candidate.setCamera(camera);
        candidate.execute();
        return std::pair{candidate.downloadGradients(), candidate.schedule()};
    }();
    const auto candidate_error = [&](const GaussianGradientCpu& actual) {
        float maximum = 0.0f;
        const auto update = [&](const glm::vec4& expected,
                                const glm::vec4& observed) {
            for (int component = 0; component < 4; component++) {
                maximum = std::max(maximum,
                                   std::abs(expected[component] -
                                            observed[component]));
            }
        };
        update(exact_gather_gradients.means[0], actual.means[0]);
        update(exact_gather_gradients.scales_opacity[0],
               actual.scales_opacity[0]);
        update(exact_gather_gradients.rotations[0], actual.rotations[0]);
        update(exact_gather_gradients.sh_dc[0], actual.sh_dc[0]);
        return maximum;
    };
    const float pixel_error=candidate_error(pixel_gradients);
    const float per_splat_error=candidate_error(per_splat_gradients);
    const float per_splat_cas_error=candidate_error(per_splat_cas_gradients);
    const float per_splat_pair_error=candidate_error(per_splat_pair_gradients);
    if(pixel_schedule.backward_work_assignment!="pixel_stream_subgroup" ||
       per_splat_schedule.backward_work_assignment!="per_splat" ||
       per_splat_cas_schedule.backward_work_assignment!="per_splat" ||
       per_splat_cas_schedule.backward_global_merge!="u32_cas" ||
       per_splat_pair_schedule.backward_work_assignment!="per_splat" ||
       per_splat_pair_schedule.backward_global_merge!=
               "gaussian_major_fixed_reduce" ||
       pixel_schedule.schedule_candidate_id.empty() ||
       per_splat_schedule.schedule_candidate_id.empty() ||
       per_splat_cas_schedule.schedule_candidate_id.empty() ||
       per_splat_pair_schedule.schedule_candidate_id.empty() ||
       !pixel_schedule.schedule_candidate_accepted ||
       !per_splat_schedule.schedule_candidate_accepted ||
       !per_splat_cas_schedule.schedule_candidate_accepted ||
       !per_splat_pair_schedule.schedule_candidate_accepted ||
       pixel_schedule.schedule_selection_source !=
               "runtime_candidate_ledger" ||
       !per_splat_schedule.per_splat_backward ||
       !std::isfinite(pixel_error) || !std::isfinite(per_splat_error) ||
       !std::isfinite(per_splat_cas_error) || pixel_error>1.0e-4f ||
       !std::isfinite(per_splat_pair_error) ||
       per_splat_error>1.0e-4f || per_splat_cas_error>1.0e-4f ||
       per_splat_pair_error>1.0e-4f) {
        std::cerr << "scatter candidate mismatch: pixel=" << pixel_error
                  << " per_splat=" << per_splat_error
                  << " per_splat_cas=" << per_splat_cas_error
                  << " per_splat_pair=" << per_splat_pair_error << "\n";
        return 1;
    }

    GaussianForwardProgram stochastic(
            {.width = kWidth,
             .height = kHeight,
             .gaussian_capacity = state.size(),
             .tile_size = 16,
             .tile_bin_capacity = 32,
             .enable_backward = true,
             .stochastic_backward_samples = 4});
    stochastic.uploadState(state);
    stochastic.uploadTarget(target);
    stochastic.setCamera(camera);
    stochastic.execute();
    const auto stochastic_gradients = stochastic.downloadGradients();
    bool found_sobol_pass = false;
    for (const auto& timing : stochastic.stats().timings) {
        if (timing.label == "raster_l1_backward_sobol_k4") {
            found_sobol_pass = true;
        }
    }
    if (!found_sobol_pass || stochastic_gradients.means.size() != 1 ||
        !std::isfinite(stochastic_gradients.means[0].x) ||
        !std::isfinite(stochastic_gradients.scales_opacity[0].w) ||
        !std::isfinite(stochastic_gradients.sh_dc[0].x)) {
        std::cerr << "Sobol gather backward did not produce finite gradients\n";
        return 1;
    }

    GaussianForwardProgram ssim(
            {.width = kWidth,
             .height = kHeight,
             .gaussian_capacity = state.size(),
             .tile_size = 16,
             .tile_bin_capacity = 32,
             .enable_backward = true,
             .l1_weight = 0.8f,
             .ssim_weight = 0.2f});
    ssim.uploadState(state);
    ssim.uploadTarget(target);
    ssim.setCamera(camera);
    ssim.execute();
    const auto ssim_gradients = ssim.downloadGradients();
    bool found_ssim_forward = false, found_ssim_backward = false;
    for (const auto& timing : ssim.stats().timings) {
        found_ssim_forward |= timing.label == "ssim_forward";
        found_ssim_backward |= timing.label == "ssim_backward";
    }
    if (!found_ssim_forward || !found_ssim_backward ||
        !std::isfinite(ssim.downloadMeanL1()) ||
        !std::isfinite(ssim_gradients.sh_dc[0].x)) {
        std::cerr << "L1+SSIM backward did not produce finite gradients\n";
        return 1;
    }
    const auto evaluate_ssim_loss = [&](const GaussianStateCpu& candidate) {
        ssim.uploadState(candidate);
        ssim.setCamera(camera);
        ssim.execute();
        return ssim.downloadMeanL1();
    };
    constexpr float kSsimEpsilon = 1.0e-3f;
    auto plus_ssim = state, minus_ssim = state;
    plus_ssim.sh_dc[0].x += kSsimEpsilon;
    minus_ssim.sh_dc[0].x -= kSsimEpsilon;
    const float numerical_ssim =
            (evaluate_ssim_loss(plus_ssim) - evaluate_ssim_loss(minus_ssim)) /
            (2.0f * kSsimEpsilon);
    const float analytical_ssim = ssim_gradients.sh_dc[0].x;
    if (std::abs(numerical_ssim - analytical_ssim) >
        0.03f * std::max(1.0f, std::abs(numerical_ssim))) {
        std::cerr << "SSIM gradient mismatch: analytical=" << analytical_ssim
                  << " numerical=" << numerical_ssim << "\n";
        return 1;
    }
    auto plus_opacity = state, minus_opacity = state;
    plus_opacity.scales_opacity[0].w += kSsimEpsilon;
    minus_opacity.scales_opacity[0].w -= kSsimEpsilon;
    const float numerical_ssim_opacity =
            (evaluate_ssim_loss(plus_opacity) -
             evaluate_ssim_loss(minus_opacity)) /
            (2.0f * kSsimEpsilon);
    const float analytical_ssim_opacity = ssim_gradients.scales_opacity[0].w;
    if (std::abs(numerical_ssim_opacity - analytical_ssim_opacity) >
        0.03f * std::max(1.0f, std::abs(numerical_ssim_opacity))) {
        std::cerr << "SSIM opacity gradient mismatch: analytical="
                  << analytical_ssim_opacity
                  << " numerical=" << numerical_ssim_opacity << "\n";
        return 1;
    }

    GaussianForwardProgram ssim_full(
            {.width = kWidth,
             .height = kHeight,
             .gaussian_capacity = state.size(),
             .tile_size = 16,
             .tile_bin_capacity = 32,
             .enable_backward = true,
             .l1_weight = 0.8f,
             .ssim_weight = 0.2f,
             .full_image_ssim = true});
    ssim_full.uploadState(state);
    ssim_full.uploadTarget(target);
    ssim_full.setCamera(camera);
    ssim_full.execute();
    const auto ssim_full_gradients = ssim_full.downloadGradients();
    const auto evaluate_full_ssim_loss =
            [&](const GaussianStateCpu& candidate) {
                ssim_full.uploadState(candidate);
                ssim_full.setCamera(camera);
                ssim_full.execute();
                return ssim_full.downloadMeanL1();
            };
    const float numerical_full_ssim =
            (evaluate_full_ssim_loss(plus_ssim) -
             evaluate_full_ssim_loss(minus_ssim)) /
            (2.0f * kSsimEpsilon);
    const float analytical_full_ssim = ssim_full_gradients.sh_dc[0].x;
    if (std::abs(numerical_full_ssim - analytical_full_ssim) >
        0.03f * std::max(1.0f, std::abs(numerical_full_ssim))) {
        std::cerr << "full-image SSIM gradient mismatch: analytical="
                  << analytical_full_ssim
                  << " numerical=" << numerical_full_ssim << "\n";
        return 1;
    }
    const float numerical_full_ssim_opacity =
            (evaluate_full_ssim_loss(plus_opacity) -
             evaluate_full_ssim_loss(minus_opacity)) /
            (2.0f * kSsimEpsilon);
    const float analytical_full_ssim_opacity =
            ssim_full_gradients.scales_opacity[0].w;
    if (std::abs(numerical_full_ssim_opacity -
                 analytical_full_ssim_opacity) >
        0.03f * std::max(1.0f, std::abs(numerical_full_ssim_opacity))) {
        std::cerr << "full-image SSIM opacity gradient mismatch: analytical="
                  << analytical_full_ssim_opacity
                  << " numerical=" << numerical_full_ssim_opacity << "\n";
        return 1;
    }
    std::cout << "full-image SSIM gradients validated\n";

    auto mip_state = state;
    mip_state.means[0].w = 0.4f;
    GaussianForwardProgram mip_off(
            {.width = kWidth,
             .height = kHeight,
             .gaussian_capacity = mip_state.size(),
             .tile_size = 16,
             .tile_bin_capacity = 32});
    mip_off.uploadState(mip_state);
    mip_off.setCamera(camera);
    mip_off.execute();
    const auto mip_off_image = mip_off.downloadColor();
    GaussianForwardProgram mip_on(
            {.width = kWidth,
             .height = kHeight,
             .gaussian_capacity = mip_state.size(),
             .tile_size = 16,
             .tile_bin_capacity = 32,
             .enable_backward = true,
             .enable_mip3d_filter = true});
    mip_on.uploadState(mip_state);
    mip_on.uploadTarget(target);
    mip_on.setCamera(camera);
    mip_on.execute();
    const auto mip_on_image = mip_on.downloadColor();
    const size_t mip_center =
            (static_cast<size_t>(kHeight / 2) * kWidth + kWidth / 2) * 4;
    // The 3D filter must both blur (larger footprint) and dim (opacity
    // compensation) the splat, so the peak alpha must strictly drop; if the
    // filter silently failed to apply, the images would match.
    if (!(mip_on_image.pixels[mip_center + 3] <
          mip_off_image.pixels[mip_center + 3] - 1.0e-3f)) {
        std::cerr << "mip3d filter did not attenuate the rendered splat: "
                  << mip_on_image.pixels[mip_center + 3] << " vs "
                  << mip_off_image.pixels[mip_center + 3] << "\n";
        return 1;
    }
    const auto mip_gradients = mip_on.downloadGradients();
    const auto evaluate_mip_loss = [&](const GaussianStateCpu& candidate) {
        mip_on.uploadState(candidate);
        mip_on.setCamera(camera);
        mip_on.execute();
        return mip_on.downloadMeanL1();
    };
    constexpr float kMipEpsilon = 1.0e-3f;
    const auto mip_check = [&](int component, float analytical) {
        auto plus = mip_state, minus = mip_state;
        if (component < 3) {
            plus.scales_opacity[0][component] += kMipEpsilon;
            minus.scales_opacity[0][component] -= kMipEpsilon;
        } else {
            plus.scales_opacity[0].w += kMipEpsilon;
            minus.scales_opacity[0].w -= kMipEpsilon;
        }
        const float numerical =
                (evaluate_mip_loss(plus) - evaluate_mip_loss(minus)) /
                (2.0f * kMipEpsilon);
        if (std::abs(numerical - analytical) >
            0.03f * std::max(1.0f, std::abs(numerical))) {
            std::cerr << "mip3d gradient mismatch (component " << component
                      << "): analytical=" << analytical
                      << " numerical=" << numerical << "\n";
            return false;
        }
        return true;
    };
    if (!mip_check(0, mip_gradients.scales_opacity[0].x) ||
        !mip_check(3, mip_gradients.scales_opacity[0].w)) {
        return 1;
    }
    std::cout << "mip3d filter forward/backward validated\n";

    GaussianForwardProgram mcmc(
            {.width = kWidth,
             .height = kHeight,
             .gaussian_capacity = 2,
             .tile_size = 16,
             .tile_bin_capacity = 32,
             .enable_backward = true,
             .enable_optimizer = true,
             .resident_view_capacity = 1,
             .enable_mcmc = true,
             .mcmc_refine_start = 0,
             .mcmc_refine_stop = 3,
             .mcmc_refine_every = 1,
             .mcmc_grow_factor = 2.0f});
    mcmc.uploadState(state);
    mcmc.uploadTrainingSet({camera}, {target});
    mcmc.execute();
    mcmc.execute();
    const auto dense_state = mcmc.downloadState();
    if (dense_state.size() != 2 || dense_state.sh_rest.size() != 24) {
        std::cerr << "MCMC active-count growth/state migration failed\n";
        return 1;
    }

    // The preview must pair each render with the photo of the view that was
    // trained, however the targets are stored.
    {
        const auto solid_target = [&](float red, float green, float blue) {
            FloatImage image = target;
            for (size_t i = 0; i < image.pixels.size(); i += 4) {
                image.pixels[i] = red;
                image.pixels[i + 1] = green;
                image.pixels[i + 2] = blue;
            }
            return image;
        };
        const std::vector<FloatImage> views = {solid_target(0.4f, 0.0f, 1.0f),
                                               solid_target(0.2f, 0.6f, 0.8f)};
        for (const bool byte_targets : {false, true}) {
            for (const bool streamed : {false, true}) {
                GaussianForwardProgram trainer(
                        {.width = kWidth,
                         .height = kHeight,
                         .gaussian_capacity = 2,
                         .tile_size = 16,
                         .tile_bin_capacity = 32,
                         .enable_backward = true,
                         .enable_optimizer = true,
                         .resident_view_capacity = 2,
                         .force_streamed_target = streamed,
                         .byte_targets = byte_targets});
                trainer.uploadState(state);
                trainer.uploadTrainingSet({camera, camera}, views);
                for (uint32_t step = 0; step < 2; step++) {
                    trainer.execute();
                    const auto preview = trainer.downloadTrainingPreview();
                    double loss_sum = 0.0;
                    for (const float value : preview.loss) loss_sum += value;
                    const double loss_mean =
                            loss_sum / static_cast<double>(preview.loss.size());
                    float target_error = 0.0f;
                    for (size_t i = 0; i < views[step].pixels.size(); i++) {
                        target_error = std::max(
                                target_error,
                                std::abs(preview.target.pixels[i] -
                                         views[step].pixels[i]));
                    }
                    if (preview.view != step ||
                        target_error > 0.51f / 255.0f ||
                        preview.rendered.pixels !=
                                trainer.downloadColor().pixels ||
                        std::abs(loss_mean - trainer.downloadMeanL1()) >
                                1.0e-6) {
                        std::cerr << "training preview mismatch: byte_targets="
                                  << byte_targets << " streamed=" << streamed
                                  << " step=" << step
                                  << " view=" << preview.view
                                  << " target_error=" << target_error << "\n";
                        return 1;
                    }
                }
            }
        }
    }
    std::cout << "training preview validated\n";

    // Force the range-safe 6+6 SH-rest lowering on a device where the
    // monolithic buffer would also fit. This is the semantic A/B oracle used
    // before enabling the lowering automatically on 128 MiB Adreno devices.
    auto split_state = state;
    for (size_t i = 0; i < split_state.sh_rest.size(); i++) {
        split_state.sh_rest[i] = glm::vec4(
                0.001f * static_cast<float>(i + 1),
                -0.0007f * static_cast<float>(i + 1),
                0.0003f * static_cast<float>(i + 1), 0.0f);
    }
    auto split_config = GaussianForwardConfig{
            .width = kWidth,
            .height = kHeight,
            .gaussian_capacity = 2,
            .tile_size = 16,
            .tile_bin_capacity = 32,
            .enable_backward = true,
            .enable_optimizer = true,
            .resident_view_capacity = 1,
            .sh_degree = 3,
            .enable_mcmc = true,
            .mcmc_refine_start = 0,
            .mcmc_refine_stop = 3,
            .mcmc_refine_every = 1,
            .mcmc_grow_factor = 2.0f,
            .force_split_projection_optimizer = true};
    GaussianForwardProgram monolithic_sh(split_config);
    split_config.force_split_sh_rest_storage = true;
    GaussianForwardProgram split_sh(split_config);
    const auto run_sh_storage = [&](GaussianForwardProgram& candidate) {
        candidate.uploadState(split_state);
        candidate.uploadTrainingSet({camera}, {target});
        candidate.execute();
        const float candidate_loss = candidate.downloadMeanL1();
        const auto candidate_gradients = candidate.downloadGradients();
        candidate.execute();
        return std::tuple{candidate_loss, candidate_gradients,
                          candidate.downloadState(), candidate.schedule()};
    };
    const auto [monolithic_loss, monolithic_gradients, monolithic_state,
                monolithic_schedule] = run_sh_storage(monolithic_sh);
    const auto [split_loss, split_gradients, split_result, split_schedule] =
            run_sh_storage(split_sh);
    float split_max_error = std::abs(monolithic_loss - split_loss);
    const auto accumulate_vec4_error = [&](const std::vector<glm::vec4>& a,
                                           const std::vector<glm::vec4>& b) {
        if (a.size() != b.size()) {
            split_max_error = std::numeric_limits<float>::infinity();
            return;
        }
        for (size_t i = 0; i < a.size(); i++)
            for (int c = 0; c < 4; c++)
                split_max_error = std::max(
                        split_max_error, std::abs(a[i][c] - b[i][c]));
    };
    accumulate_vec4_error(monolithic_gradients.means,
                          split_gradients.means);
    accumulate_vec4_error(monolithic_gradients.scales_opacity,
                          split_gradients.scales_opacity);
    accumulate_vec4_error(monolithic_gradients.rotations,
                          split_gradients.rotations);
    accumulate_vec4_error(monolithic_gradients.sh_dc,
                          split_gradients.sh_dc);
    accumulate_vec4_error(monolithic_gradients.sh_rest,
                          split_gradients.sh_rest);
    accumulate_vec4_error(monolithic_state.means, split_result.means);
    accumulate_vec4_error(monolithic_state.scales_opacity,
                          split_result.scales_opacity);
    accumulate_vec4_error(monolithic_state.rotations, split_result.rotations);
    accumulate_vec4_error(monolithic_state.sh_dc, split_result.sh_dc);
    accumulate_vec4_error(monolithic_state.sh_rest, split_result.sh_rest);
    if (monolithic_schedule.split_sh_rest_storage ||
        !split_schedule.split_sh_rest_storage ||
        !std::isfinite(split_max_error) || split_max_error > 1.0e-6f) {
        std::cerr << "split SH-rest lowering mismatch: "
                  << split_max_error << "\n";
        return 1;
    }

    // ABG's gradient-weighted MCMC sampling adds one persistent score buffer.
    // In the range-safe 6+6 layout it follows both split SH-rest buffers, so
    // the shader binding is 16 rather than the monolithic binding 14.
    auto abg_monolithic_config = split_config;
    abg_monolithic_config.force_split_sh_rest_storage = false;
    abg_monolithic_config.full_image_ssim = true;
    abg_monolithic_config.mcmc_clone_offset = 0.5f;
    abg_monolithic_config.mcmc_gradient_sampling = 1.5f;
    GaussianForwardProgram abg_monolithic(abg_monolithic_config);
    auto abg_split_config = abg_monolithic_config;
    abg_split_config.force_split_sh_rest_storage = true;
    GaussianForwardProgram abg_split(abg_split_config);
    const auto [abg_monolithic_loss, abg_monolithic_gradients,
                abg_monolithic_state, abg_monolithic_schedule] =
            run_sh_storage(abg_monolithic);
    const auto [abg_split_loss, abg_split_gradients, abg_split_state,
                abg_split_schedule] = run_sh_storage(abg_split);
    split_max_error = std::abs(abg_monolithic_loss - abg_split_loss);
    accumulate_vec4_error(abg_monolithic_gradients.means,
                          abg_split_gradients.means);
    accumulate_vec4_error(abg_monolithic_gradients.sh_rest,
                          abg_split_gradients.sh_rest);
    accumulate_vec4_error(abg_monolithic_state.means, abg_split_state.means);
    accumulate_vec4_error(abg_monolithic_state.sh_rest,
                          abg_split_state.sh_rest);
    if (abg_monolithic_schedule.split_sh_rest_storage ||
        !abg_split_schedule.split_sh_rest_storage ||
        !std::isfinite(split_max_error) || split_max_error > 1.0e-6f) {
        std::cerr << "ABG split SH-rest binding mismatch: "
                  << split_max_error << "\n";
        return 1;
    }

    const auto evaluate_loss = [&](const GaussianStateCpu& candidate) {
        forward.uploadState(candidate);
        forward.setCamera(camera);
        forward.execute();
        const auto rendered = forward.downloadColor();
        double sum = 0.0;
        for (size_t pixel = 0;
             pixel < static_cast<size_t>(kWidth) * kHeight; pixel++) {
            sum += std::abs(rendered.pixels[pixel * 4]);
            sum += std::abs(rendered.pixels[pixel * 4 + 1]);
            sum += std::abs(rendered.pixels[pixel * 4 + 2]);
        }
        return static_cast<float>(
                sum / (static_cast<double>(kWidth) * kHeight * 3.0));
    };
    const auto finite_difference = [&](auto perturb) {
        constexpr float epsilon = 0.001f;
        auto plus = state;
        auto minus = state;
        perturb(plus, epsilon);
        perturb(minus, -epsilon);
        return (evaluate_loss(plus) - evaluate_loss(minus)) /
               (2.0f * epsilon);
    };
    const auto close_gradient = [](float analytic, float numerical) {
        return std::isfinite(analytic) && std::isfinite(numerical) &&
               std::abs(analytic - numerical) <=
                       0.002f +
                               0.15f *
                                       std::max(std::abs(analytic),
                                                std::abs(numerical));
    };
    const float numeric_mean_x = finite_difference(
            [](GaussianStateCpu& value, float delta) {
                value.means[0].x += delta;
            });
    const float numeric_scale_y = finite_difference(
            [](GaussianStateCpu& value, float delta) {
                value.scales_opacity[0].y += delta;
            });
    const float numeric_opacity = finite_difference(
            [](GaussianStateCpu& value, float delta) {
                value.scales_opacity[0].w += delta;
            });
    const float numeric_rotation_z = finite_difference(
            [](GaussianStateCpu& value, float delta) {
                value.rotations[0].z += delta;
            });
    const float numeric_sh_x = finite_difference(
            [](GaussianStateCpu& value, float delta) {
                value.sh_dc[0].x += delta;
            });
    if (!close_gradient(gradients.means[0].x, numeric_mean_x) ||
        !close_gradient(gradients.scales_opacity[0].y, numeric_scale_y) ||
        !close_gradient(gradients.scales_opacity[0].w, numeric_opacity) ||
        !close_gradient(gradients.rotations[0].z, numeric_rotation_z) ||
        !close_gradient(gradients.sh_dc[0].x, numeric_sh_x)) {
        std::cerr << "finite difference mismatch\n"
                  << "mean.x " << gradients.means[0].x << " vs "
                  << numeric_mean_x << "\nscale.y "
                  << gradients.scales_opacity[0].y << " vs "
                  << numeric_scale_y << "\nopacity "
                  << gradients.scales_opacity[0].w << " vs "
                  << numeric_opacity << "\nrotation.z "
                  << gradients.rotations[0].z << " vs "
                  << numeric_rotation_z << "\nsh.x "
                  << gradients.sh_dc[0].x << " vs " << numeric_sh_x << "\n";
        return 1;
    }
    GaussianForwardProgram trainer(
            {.width = kWidth,
             .height = kHeight,
             .gaussian_capacity = state.size(),
             .tile_size = 16,
             .tile_bin_capacity = 32,
             .enable_backward = true,
             .enable_optimizer = true,
             .means_lr = 0.01f,
             .scales_lr = 0.01f,
             .opacity_lr = 0.01f,
             .rotation_lr = 0.005f,
             .sh_dc_lr = 0.02f,
             .scale_regularizer = 0.0f,
             .opacity_regularizer = 0.0f});
    trainer.uploadState(state);
    trainer.uploadTarget(image);
    trainer.setCamera(camera);
    trainer.execute();
    const float initial_training_loss = trainer.downloadMeanL1();
    for (int iteration = 1; iteration < 80; iteration++) {
        trainer.execute();
    }
    const float final_training_loss = trainer.downloadMeanL1();
    if (!std::isfinite(final_training_loss) ||
        final_training_loss >= initial_training_loss * 0.7f) {
        std::cerr << "Adam synthetic convergence failed: "
                  << initial_training_loss << " -> " << final_training_loss
                  << "\n";
        return 1;
    }
    std::cerr << "Adam synthetic convergence " << initial_training_loss
              << " -> " << final_training_loss << "\n";
    std::cout << "Gaussian L1 backward validated\n";
    return 0;
}
