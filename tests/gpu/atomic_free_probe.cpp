#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "cross3dgs/gaussian_forward.h"
#include "cross3dgs/gpu/gpu_program.h"

namespace {

using cross3dgs::Camera;
using cross3dgs::CameraModel;
using cross3dgs::FloatImage;
using cross3dgs::GaussianForwardConfig;
using cross3dgs::GaussianForwardProgram;
using cross3dgs::GaussianForwardStats;
using cross3dgs::GaussianGradientCpu;
using cross3dgs::GaussianStateCpu;

constexpr float kAbsoluteTolerance = 1.0e-5f;
constexpr float kRelativeTolerance = 1.0e-4f;
constexpr float kForwardAbsoluteTolerance = 1.0e-6f;
constexpr float kForwardRelativeTolerance = 1.0e-6f;
constexpr float kLossAbsoluteTolerance = 1.0e-7f;
constexpr float kLossRelativeTolerance = 1.0e-6f;
constexpr float kAdamUpdateAbsoluteTolerance = 1.0e-5f;
constexpr float kAdamUpdateRelativeTolerance = 1.0e-4f;
constexpr double kRelativeFloor = 1.0e-12;
constexpr float kShC0 = 0.28209479177387814f;
constexpr uint32_t kGradientWidth = 96;
constexpr uint32_t kGradientHeight = 80;
constexpr size_t kCameraVariantCount = 5;
constexpr size_t kStateVariantCount = 5;

struct OccupancyRegime {
    const char* name;
    size_t gaussian_count;
    float extent;
    float scale;
};

constexpr OccupancyRegime kOccupancyRegimes[] = {
        {"sparse", 4, 1.15f, 0.025f},
        {"low", 12, 0.95f, 0.045f},
        {"medium", 32, 0.60f, 0.090f},
        {"dense", 64, 0.28f, 0.180f},
};

struct ErrorGroup {
    std::string name;
    size_t count = 0;
    double sum_abs = 0.0;
    double sum_squared = 0.0;
    double max_abs = 0.0;
    double max_relative = 0.0;
    double reference_max_abs = 0.0;
    size_t reference_nonzero_count = 0;
    size_t tolerance_exceed_count = 0;
    std::string worst_case_id;
    size_t worst_element_index = 0;
    double worst_reference = 0.0;
    double worst_candidate = 0.0;

    void add(float reference,
             float candidate,
             std::string_view case_id,
             size_t element_index) {
        const double error = std::abs(static_cast<double>(reference) -
                                      static_cast<double>(candidate));
        const double relative =
                error / std::max(std::abs(static_cast<double>(reference)),
                                 kRelativeFloor);
        count++;
        sum_abs += error;
        sum_squared += error * error;
        max_relative = std::max(max_relative, relative);
        if (error > max_abs || worst_case_id.empty()) {
            max_abs = error;
            worst_case_id = case_id;
            worst_element_index = element_index;
            worst_reference = reference;
            worst_candidate = candidate;
        }
        reference_max_abs = std::max(
                reference_max_abs, std::abs(static_cast<double>(reference)));
        reference_nonzero_count += reference != 0.0f ? 1u : 0u;
        const double limit = kAbsoluteTolerance +
                             kRelativeTolerance *
                                     std::abs(static_cast<double>(reference));
        tolerance_exceed_count += error > limit ? 1u : 0u;
    }
    void merge(const ErrorGroup& other) {
        count += other.count;
        sum_abs += other.sum_abs;
        sum_squared += other.sum_squared;
        max_relative = std::max(max_relative, other.max_relative);
        reference_max_abs =
                std::max(reference_max_abs, other.reference_max_abs);
        reference_nonzero_count += other.reference_nonzero_count;
        tolerance_exceed_count += other.tolerance_exceed_count;
        if (other.max_abs > max_abs || worst_case_id.empty()) {
            max_abs = other.max_abs;
            worst_case_id = other.worst_case_id;
            worst_element_index = other.worst_element_index;
            worst_reference = other.worst_reference;
            worst_candidate = other.worst_candidate;
        }
    }
    double mean_abs() const { return count ? sum_abs / count : 0.0; }
    double rms() const {
        return count ? std::sqrt(sum_squared / count) : 0.0;
    }
};

struct ScalingRow {
    size_t gaussian_count = 0;
    uint64_t pair_count = 0;
    size_t pair_capacity = 0;
    size_t scratch_used_bytes = 0;
    size_t scratch_allocated_bytes = 0;
    size_t total_allocated_bytes = 0;
    uint32_t tile_count = 0;
    uint32_t max_tile_count = 0;
    double mean_pairs_per_tile = 0.0;
    double pairs_per_gaussian = 0.0;
    double max_tile_capacity_fraction = 0.0;
    double median_tiles_gpu_ms = 0.0;
    double median_reduce_gpu_ms = 0.0;
    double median_pair_backward_gpu_ms = 0.0;
    double median_execute_host_ms = 0.0;
};

struct GradientCaseResult {
    std::string case_id;
    std::string occupancy_regime;
    size_t state_variant = 0;
    size_t camera_variant = 0;
    size_t gaussian_count = 0;
    uint64_t pair_count = 0;
    uint32_t max_tile_count = 0;
    std::vector<ErrorGroup> groups;
};

struct SemanticMetric {
    std::string name;
    double absolute_tolerance = 0.0;
    double relative_tolerance = 0.0;
    size_t count = 0;
    size_t reference_nonzero_count = 0;
    size_t tolerance_exceed_count = 0;
    double sum_abs = 0.0;
    double sum_squared = 0.0;
    double max_abs = 0.0;
    double max_relative = 0.0;
    double reference_max_abs = 0.0;
    std::string worst_case_id;
    size_t worst_element_index = 0;
    double worst_reference = 0.0;
    double worst_candidate = 0.0;

    void add(float reference,
             float candidate,
             std::string_view case_id,
             size_t element_index) {
        const double error = std::abs(static_cast<double>(reference) -
                                      static_cast<double>(candidate));
        const double relative =
                error / std::max(std::abs(static_cast<double>(reference)),
                                 kRelativeFloor);
        count++;
        reference_nonzero_count += reference != 0.0f ? 1u : 0u;
        sum_abs += error;
        sum_squared += error * error;
        max_relative = std::max(max_relative, relative);
        reference_max_abs = std::max(
                reference_max_abs, std::abs(static_cast<double>(reference)));
        if (error > max_abs || worst_case_id.empty()) {
            max_abs = error;
            worst_case_id = case_id;
            worst_element_index = element_index;
            worst_reference = reference;
            worst_candidate = candidate;
        }
        const double limit = absolute_tolerance +
                             relative_tolerance *
                                     std::abs(static_cast<double>(reference));
        tolerance_exceed_count += error > limit ? 1u : 0u;
    }

    void merge(const SemanticMetric& other) {
        if (name != other.name || absolute_tolerance != other.absolute_tolerance ||
            relative_tolerance != other.relative_tolerance)
            throw std::runtime_error("semantic metric schema mismatch");
        count += other.count;
        reference_nonzero_count += other.reference_nonzero_count;
        tolerance_exceed_count += other.tolerance_exceed_count;
        sum_abs += other.sum_abs;
        sum_squared += other.sum_squared;
        max_relative = std::max(max_relative, other.max_relative);
        reference_max_abs =
                std::max(reference_max_abs, other.reference_max_abs);
        if (other.max_abs > max_abs || worst_case_id.empty()) {
            max_abs = other.max_abs;
            worst_case_id = other.worst_case_id;
            worst_element_index = other.worst_element_index;
            worst_reference = other.worst_reference;
            worst_candidate = other.worst_candidate;
        }
    }

    double meanAbs() const { return count ? sum_abs / count : 0.0; }
    double rms() const {
        return count ? std::sqrt(sum_squared / count) : 0.0;
    }
};

struct SemanticCaseResult {
    std::string case_id;
    std::string occupancy_regime;
    size_t state_variant = 0;
    size_t camera_variant = 0;
    size_t gaussian_count = 0;
    std::vector<SemanticMetric> metrics;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    if (values.empty()) return 0.0;
    const size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle]
                             : (values[middle - 1] + values[middle]) * 0.5;
}

Camera makeCamera(uint32_t width, uint32_t height, float focal) {
    Camera camera;
    camera.width = width;
    camera.height = height;
    camera.fx = focal;
    camera.fy = focal;
    camera.cx = static_cast<float>(width) * 0.5f;
    camera.cy = static_cast<float>(height) * 0.5f;
    camera.model = CameraModel::Pinhole;
    return camera;
}

FloatImage zeroTarget(uint32_t width, uint32_t height) {
    FloatImage target;
    target.width = width;
    target.height = height;
    target.pixels.assign(static_cast<size_t>(width) * height * 4, 0.0f);
    return target;
}

uint32_t nextRandom(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float randomUnit(uint32_t& state) {
    return static_cast<float>(nextRandom(state) & 0x00ffffffu) /
           static_cast<float>(0x01000000u);
}

Camera makeGradientCamera(size_t variant) {
    static constexpr float kFocal[] = {58.0f, 66.0f, 74.0f, 82.0f, 70.0f};
    Camera camera =
            makeCamera(kGradientWidth, kGradientHeight, kFocal[variant]);
    camera.fx *= 1.0f + 0.015f * static_cast<float>(variant);
    camera.fy *= 1.0f - 0.012f * static_cast<float>(variant);
    camera.cx += static_cast<float>(static_cast<int>(variant) - 2) * 1.25f;
    camera.cy += static_cast<float>(2 - static_cast<int>(variant)) * 0.75f;
    const float yaw = (static_cast<float>(variant) - 2.0f) * 0.035f;
    const float roll = (static_cast<float>(variant % 3) - 1.0f) * 0.018f;
    camera.world_to_camera =
            glm::rotate(glm::mat4(1.0f), yaw, glm::vec3(0.0f, 1.0f, 0.0f));
    camera.world_to_camera =
            glm::rotate(camera.world_to_camera, roll,
                        glm::vec3(0.0f, 0.0f, 1.0f));
    camera.world_to_camera[3][0] =
            (static_cast<float>(variant) - 2.0f) * 0.018f;
    camera.world_to_camera[3][1] =
            (1.0f - static_cast<float>(variant % 3)) * 0.012f;
    return camera;
}

FloatImage makeGradientTarget(size_t camera_variant) {
    FloatImage target = zeroTarget(kGradientWidth, kGradientHeight);
    for (uint32_t y = 0; y < kGradientHeight; y++) {
        for (uint32_t x = 0; x < kGradientWidth; x++) {
            const float u = static_cast<float>(x) /
                            static_cast<float>(kGradientWidth - 1);
            const float v = static_cast<float>(y) /
                            static_cast<float>(kGradientHeight - 1);
            const size_t index =
                    (static_cast<size_t>(y) * kGradientWidth + x) * 4;
            const float offset = 0.025f * static_cast<float>(camera_variant);
            target.pixels[index + 0] =
                    std::clamp(0.10f + 0.65f * u + offset, 0.0f, 1.0f);
            target.pixels[index + 1] =
                    std::clamp(0.15f + 0.60f * v - offset, 0.0f, 1.0f);
            target.pixels[index + 2] =
                    0.20f + 0.35f * (u + v) * 0.5f;
            target.pixels[index + 3] = 1.0f;
        }
    }
    return target;
}

GaussianStateCpu makeGradientState(const OccupancyRegime& regime,
                                   size_t state_variant) {
    GaussianStateCpu state;
    const size_t count = regime.gaussian_count;
    state.means.reserve(count);
    state.scales_opacity.reserve(count);
    state.rotations.reserve(count);
    state.sh_dc.reserve(count);
    state.sh_rest.assign(count * 12, glm::vec4(0.0f));
    const size_t columns = static_cast<size_t>(std::ceil(std::sqrt(count)));
    const size_t rows = (count + columns - 1) / columns;
    uint32_t rng = 0x9e3779b9u ^
                   (static_cast<uint32_t>(state_variant) + 1u) * 0x85ebca6bu ^
                   static_cast<uint32_t>(count * 97u);
    for (size_t i = 0; i < count; i++) {
        const size_t column = i % columns;
        const size_t row = i / columns;
        const float grid_x =
                ((static_cast<float>(column) + 0.5f) /
                         static_cast<float>(columns) *
                         2.0f -
                 1.0f);
        const float grid_y =
                ((static_cast<float>(row) + 0.5f) /
                         static_cast<float>(rows) *
                         2.0f -
                 1.0f);
        const float jitter_x = (randomUnit(rng) - 0.5f) * 0.08f;
        const float jitter_y = (randomUnit(rng) - 0.5f) * 0.08f;
        state.means.emplace_back(
                (grid_x + jitter_x) * regime.extent,
                (grid_y + jitter_y) * regime.extent,
                2.65f + 0.7f * randomUnit(rng), 0.0f);
        const float scale = regime.scale * (0.75f + 0.5f * randomUnit(rng));
        state.scales_opacity.emplace_back(
                scale, scale * (0.75f + 0.5f * randomUnit(rng)),
                scale * (0.75f + 0.5f * randomUnit(rng)),
                0.30f + 0.60f * randomUnit(rng));
        glm::vec4 rotation(randomUnit(rng) * 2.0f - 1.0f,
                           randomUnit(rng) * 2.0f - 1.0f,
                           randomUnit(rng) * 2.0f - 1.0f,
                           randomUnit(rng) * 2.0f - 1.0f);
        if (glm::dot(rotation, rotation) < 1.0e-6f)
            rotation = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
        state.rotations.emplace_back(glm::normalize(rotation));
        const glm::vec3 rgb(0.15f + 0.70f * randomUnit(rng),
                            0.15f + 0.70f * randomUnit(rng),
                            0.15f + 0.70f * randomUnit(rng));
        state.sh_dc.emplace_back((rgb - 0.5f) / kShC0, 0.0f);
        for (size_t coefficient = 0; coefficient < 45; coefficient++) {
            const size_t packed = i * 12 + coefficient / 4;
            const int lane = static_cast<int>(coefficient % 4);
            state.sh_rest[packed][lane] =
                    0.015f * (randomUnit(rng) * 2.0f - 1.0f);
        }
    }
    return state;
}

GaussianStateCpu makeScalingState(size_t count) {
    GaussianStateCpu state;
    state.means.reserve(count);
    state.scales_opacity.reserve(count);
    state.rotations.reserve(count);
    state.sh_dc.reserve(count);
    state.sh_rest.assign(count * 12, glm::vec4(0.0f));
    const size_t columns = static_cast<size_t>(std::ceil(std::sqrt(count)));
    const size_t rows = (count + columns - 1) / columns;
    const float scale = count >= 16384 ? 0.008f : 0.018f;
    for (size_t i = 0; i < count; i++) {
        const size_t column = i % columns;
        const size_t row = i / columns;
        const float x = (static_cast<float>(column) + 0.5f) /
                                static_cast<float>(columns) *
                                4.10f -
                        2.05f;
        const float y = (static_cast<float>(row) + 0.5f) /
                                static_cast<float>(rows) *
                                4.10f -
                        2.05f;
        state.means.emplace_back(x, y, 3.0f + 0.001f * (i % 11), 0.0f);
        state.scales_opacity.emplace_back(scale, scale, scale, 0.55f);
        state.rotations.emplace_back(1.0f, 0.0f, 0.0f, 0.0f);
        const glm::vec3 rgb(0.55f + 0.1f * static_cast<float>(i % 3),
                            0.50f, 0.45f);
        state.sh_dc.emplace_back((rgb - 0.5f) / kShC0, 0.0f);
    }
    return state;
}

std::vector<ErrorGroup> compareGradientGroups(
        const GaussianGradientCpu& reference,
        const GaussianGradientCpu& candidate,
        std::string_view case_id) {
    if (reference.means.size() != candidate.means.size() ||
        reference.scales_opacity.size() != candidate.scales_opacity.size() ||
        reference.rotations.size() != candidate.rotations.size() ||
        reference.sh_dc.size() != candidate.sh_dc.size() ||
        reference.sh_rest.size() != candidate.sh_rest.size()) {
        throw std::runtime_error("gradient array shape mismatch");
    }
    std::vector<ErrorGroup> groups = {{"mean_xyz"},
                                      {"scale_xyz"},
                                      {"opacity"},
                                      {"rotation_xyzw"},
                                      {"sh_dc_rgb"},
                                      {"sh_rest_45"}};
    for (size_t i = 0; i < reference.means.size(); i++) {
        for (int lane = 0; lane < 3; lane++)
            groups[0].add(reference.means[i][lane], candidate.means[i][lane],
                          case_id, i * 3 + static_cast<size_t>(lane));
        for (int lane = 0; lane < 3; lane++)
            groups[1].add(reference.scales_opacity[i][lane],
                          candidate.scales_opacity[i][lane], case_id,
                          i * 3 + static_cast<size_t>(lane));
        groups[2].add(reference.scales_opacity[i].w,
                      candidate.scales_opacity[i].w, case_id, i);
        for (int lane = 0; lane < 4; lane++)
            groups[3].add(reference.rotations[i][lane],
                          candidate.rotations[i][lane], case_id,
                          i * 4 + static_cast<size_t>(lane));
        for (int lane = 0; lane < 3; lane++)
            groups[4].add(reference.sh_dc[i][lane], candidate.sh_dc[i][lane],
                          case_id, i * 3 + static_cast<size_t>(lane));
        for (size_t coefficient = 0; coefficient < 45; coefficient++) {
            const size_t packed = i * 12 + coefficient / 4;
            const int lane = static_cast<int>(coefficient % 4);
            groups[5].add(reference.sh_rest[packed][lane],
                          candidate.sh_rest[packed][lane], case_id,
                          i * 45 + coefficient);
        }
    }
    return groups;
}

std::vector<SemanticMetric> makeSemanticMetrics() {
    return {{"forward_rgba", kForwardAbsoluteTolerance,
             kForwardRelativeTolerance},
            {"mean_l1_loss", kLossAbsoluteTolerance,
             kLossRelativeTolerance},
            {"adam_delta_mean_xyz", kAdamUpdateAbsoluteTolerance,
             kAdamUpdateRelativeTolerance},
            {"adam_delta_scale_xyz", kAdamUpdateAbsoluteTolerance,
             kAdamUpdateRelativeTolerance},
            {"adam_delta_opacity", kAdamUpdateAbsoluteTolerance,
             kAdamUpdateRelativeTolerance},
            {"adam_delta_rotation_xyzw", kAdamUpdateAbsoluteTolerance,
             kAdamUpdateRelativeTolerance},
            {"adam_delta_sh_dc_rgb", kAdamUpdateAbsoluteTolerance,
             kAdamUpdateRelativeTolerance},
            {"adam_delta_sh_rest_45", kAdamUpdateAbsoluteTolerance,
             kAdamUpdateRelativeTolerance}};
}

void compareForwardImage(const FloatImage& reference,
                         const FloatImage& candidate,
                         std::string_view case_id,
                         SemanticMetric& metric) {
    if (reference.width != candidate.width ||
        reference.height != candidate.height ||
        reference.channels != candidate.channels ||
        reference.pixels.size() != candidate.pixels.size())
        throw std::runtime_error("forward image shape mismatch");
    for (size_t i = 0; i < reference.pixels.size(); i++)
        metric.add(reference.pixels[i], candidate.pixels[i], case_id, i);
}

void compareAdamUpdateGroups(const GaussianStateCpu& initial,
                             const GaussianStateCpu& reference,
                             const GaussianStateCpu& candidate,
                             std::string_view case_id,
                             std::vector<SemanticMetric>& metrics) {
    const auto has_valid_shape = [](const GaussianStateCpu& state) {
        return state.scales_opacity.size() == state.size() &&
                state.rotations.size() == state.size() &&
                state.sh_dc.size() == state.size() &&
                state.sh_rest.size() == state.size() * 12;
    };
    if (!has_valid_shape(initial) || !has_valid_shape(reference) ||
        !has_valid_shape(candidate) || initial.size() != reference.size() ||
        initial.size() != candidate.size() || metrics.size() != 8)
        throw std::runtime_error("Adam update shape mismatch");
    const auto add_delta = [&](SemanticMetric& metric,
                               float initial_value,
                               float reference_value,
                               float candidate_value,
                               size_t index) {
        metric.add(reference_value - initial_value,
                   candidate_value - initial_value, case_id, index);
    };
    for (size_t i = 0; i < initial.size(); i++) {
        for (int lane = 0; lane < 3; lane++)
            add_delta(metrics[2], initial.means[i][lane],
                      reference.means[i][lane], candidate.means[i][lane],
                      i * 3 + static_cast<size_t>(lane));
        for (int lane = 0; lane < 3; lane++)
            add_delta(metrics[3], initial.scales_opacity[i][lane],
                      reference.scales_opacity[i][lane],
                      candidate.scales_opacity[i][lane],
                      i * 3 + static_cast<size_t>(lane));
        add_delta(metrics[4], initial.scales_opacity[i].w,
                  reference.scales_opacity[i].w,
                  candidate.scales_opacity[i].w, i);
        for (int lane = 0; lane < 4; lane++)
            add_delta(metrics[5], initial.rotations[i][lane],
                      reference.rotations[i][lane],
                      candidate.rotations[i][lane],
                      i * 4 + static_cast<size_t>(lane));
        for (int lane = 0; lane < 3; lane++)
            add_delta(metrics[6], initial.sh_dc[i][lane],
                      reference.sh_dc[i][lane], candidate.sh_dc[i][lane],
                      i * 3 + static_cast<size_t>(lane));
        for (size_t coefficient = 0; coefficient < 45; coefficient++) {
            const size_t packed = i * 12 + coefficient / 4;
            const int lane = static_cast<int>(coefficient % 4);
            add_delta(metrics[7], initial.sh_rest[packed][lane],
                      reference.sh_rest[packed][lane],
                      candidate.sh_rest[packed][lane],
                      i * 45 + coefficient);
        }
    }
}

double passMilliseconds(const GaussianForwardStats& stats,
                        const std::string& label) {
    for (const auto& timing : stats.timings) {
        if (timing.label == label) return timing.gpu_microseconds / 1000.0;
    }
    throw std::runtime_error("missing GPU timing: " + label);
}

bool writeGradientReports(const std::filesystem::path& output_dir,
                          const std::vector<GradientCaseResult>& cases,
                          const std::vector<ErrorGroup>& groups,
                          std::string_view reference_name,
                          std::string_view candidate_name) {
    std::ofstream csv(output_dir / "gradient_error.csv");
    csv << "parameter_group,element_count,reference_nonzero_count,"
           "reference_max_abs,max_abs_error,mean_abs_error,rms_error,"
           "tolerance,pass,max_relative_error,tolerance_exceed_count,"
           "worst_case_id,worst_element_index,worst_reference,"
           "worst_candidate,relative_tolerance\n";
    csv << std::scientific << std::setprecision(9);
    double overall = 0.0;
    size_t overall_exceed_count = 0;
    for (const auto& group : groups) {
        overall = std::max(overall, group.max_abs);
        overall_exceed_count += group.tolerance_exceed_count;
        csv << group.name << ',' << group.count << ','
            << group.reference_nonzero_count << ',' << group.reference_max_abs
            << ',' << group.max_abs << ',' << group.mean_abs() << ','
            << group.rms() << ',' << kAbsoluteTolerance << ','
            << (group.tolerance_exceed_count == 0 ? "true" : "false") << ','
            << group.max_relative << ',' << group.tolerance_exceed_count << ','
            << group.worst_case_id << ','
            << group.worst_element_index << ',' << group.worst_reference << ','
            << group.worst_candidate << ',' << kRelativeTolerance << '\n';
    }

    std::ofstream case_csv(output_dir / "gradient_error_cases.csv");
    case_csv
            << "case_id,occupancy_regime,state_variant,camera_variant,"
               "gaussian_count,pair_count,max_tile_count,parameter_group,"
               "element_count,max_abs_error,max_relative_error,mean_abs_error,"
               "rms_error,tolerance_exceed_count,pass\n";
    case_csv << std::scientific << std::setprecision(9);
    for (const auto& result : cases) {
        for (const auto& group : result.groups) {
            case_csv << result.case_id << ',' << result.occupancy_regime << ','
                     << result.state_variant << ',' << result.camera_variant
                     << ',' << result.gaussian_count << ',' << result.pair_count
                     << ',' << result.max_tile_count << ',' << group.name << ','
                     << group.count << ',' << group.max_abs << ','
                     << group.max_relative << ',' << group.mean_abs() << ','
                     << group.rms() << ',' << group.tolerance_exceed_count << ','
                     << (group.tolerance_exceed_count == 0 ? "true" : "false")
                     << '\n';
        }
    }

    std::ofstream json(output_dir / "gradient_error.json");
    json << std::scientific << std::setprecision(9)
         << "{\n  \"schema\": \"cross_3dgs.lowering_gradient_error.v3\",\n"
         << "  \"reference\": \"" << reference_name << "\",\n"
         << "  \"candidate\": \"" << candidate_name << "\",\n"
         << "  \"sh_degree\": 3,\n"
         << "  \"padding_lanes_excluded\": true,\n"
         << "  \"case_count\": " << cases.size() << ",\n"
         << "  \"camera_variant_count\": " << kCameraVariantCount << ",\n"
         << "  \"state_variant_count\": " << kStateVariantCount << ",\n"
         << "  \"occupancy_regime_count\": "
         << std::size(kOccupancyRegimes) << ",\n"
         << "  \"tolerance\": " << kAbsoluteTolerance << ",\n"
         << "  \"absolute_tolerance\": " << kAbsoluteTolerance << ",\n"
         << "  \"relative_tolerance\": " << kRelativeTolerance << ",\n"
         << "  \"acceptance\": \"abs_error <= abs_tol + rel_tol * "
            "abs(reference)\",\n"
         << "  \"overall_max_abs_error\": " << overall << ",\n"
         << "  \"tolerance_exceed_count\": " << overall_exceed_count << ",\n"
         << "  \"pass\": " << (overall_exceed_count == 0 ? "true" : "false")
         << ",\n  \"groups\": [\n";
    for (size_t i = 0; i < groups.size(); i++) {
        const auto& group = groups[i];
        json << "    {\"parameter_group\": \"" << group.name
             << "\", \"element_count\": " << group.count
             << ", \"reference_nonzero_count\": "
             << group.reference_nonzero_count
             << ", \"reference_max_abs\": " << group.reference_max_abs
             << ", \"max_abs_error\": " << group.max_abs
             << ", \"max_relative_error\": " << group.max_relative
             << ", \"mean_abs_error\": " << group.mean_abs()
             << ", \"rms_error\": " << group.rms()
             << ", \"tolerance_exceed_count\": "
             << group.tolerance_exceed_count
             << ", \"worst_case_id\": \"" << group.worst_case_id
             << "\", \"worst_element_index\": "
             << group.worst_element_index
             << ", \"worst_reference\": " << group.worst_reference
             << ", \"worst_candidate\": " << group.worst_candidate
             << ", \"pass\": "
             << (group.tolerance_exceed_count == 0 ? "true" : "false") << "}"
             << (i + 1 == groups.size() ? "\n" : ",\n");
    }
    json << "  ],\n  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); i++) {
        const auto& result = cases[i];
        size_t exceed_count = 0;
        for (const auto& group : result.groups)
            exceed_count += group.tolerance_exceed_count;
        json << "    {\"case_id\": \"" << result.case_id
             << "\", \"occupancy_regime\": \""
             << result.occupancy_regime << "\", \"state_variant\": "
             << result.state_variant << ", \"camera_variant\": "
             << result.camera_variant << ", \"gaussian_count\": "
             << result.gaussian_count << ", \"pair_count\": "
             << result.pair_count << ", \"max_tile_count\": "
             << result.max_tile_count << ", \"tolerance_exceed_count\": "
             << exceed_count << ", \"pass\": "
             << (exceed_count == 0 ? "true" : "false") << "}"
             << (i + 1 == cases.size() ? "\n" : ",\n");
    }
    json << "  ]\n}\n";
    return overall_exceed_count == 0;
}

bool writeSemanticReports(const std::filesystem::path& output_dir,
                          const std::vector<SemanticCaseResult>& cases,
                          const std::vector<SemanticMetric>& metrics,
                          std::string_view reference_name,
                          std::string_view candidate_name) {
    size_t overall_exceed_count = 0;
    size_t optimizer_groups_without_updates = 0;
    for (size_t i = 0; i < metrics.size(); i++) {
        overall_exceed_count += metrics[i].tolerance_exceed_count;
        if (i >= 2 && metrics[i].reference_nonzero_count == 0)
            optimizer_groups_without_updates++;
    }
    const bool pass = overall_exceed_count == 0 &&
                      optimizer_groups_without_updates == 0;

    std::ofstream csv(output_dir / "semantic_equivalence.csv");
    csv << "metric,element_count,reference_nonzero_count,reference_max_abs,"
           "max_abs_error,mean_abs_error,rms_error,absolute_tolerance,"
           "relative_tolerance,max_relative_error,tolerance_exceed_count,"
           "worst_case_id,worst_element_index,worst_reference,"
           "worst_candidate,pass\n";
    csv << std::scientific << std::setprecision(9);
    for (const auto& metric : metrics) {
        const bool metric_pass = metric.tolerance_exceed_count == 0 &&
                (metric.name.rfind("adam_delta_", 0) != 0 ||
                 metric.reference_nonzero_count != 0);
        csv << metric.name << ',' << metric.count << ','
            << metric.reference_nonzero_count << ','
            << metric.reference_max_abs << ',' << metric.max_abs << ','
            << metric.meanAbs() << ',' << metric.rms() << ','
            << metric.absolute_tolerance << ',' << metric.relative_tolerance
            << ',' << metric.max_relative << ','
            << metric.tolerance_exceed_count << ',' << metric.worst_case_id
            << ',' << metric.worst_element_index << ','
            << metric.worst_reference << ',' << metric.worst_candidate << ','
            << (metric_pass ? "true" : "false") << '\n';
    }

    std::ofstream case_csv(output_dir / "semantic_equivalence_cases.csv");
    case_csv << "case_id,occupancy_regime,state_variant,camera_variant,"
                "gaussian_count,metric,element_count,reference_nonzero_count,"
                "reference_max_abs,max_abs_error,max_relative_error,"
                "mean_abs_error,rms_error,absolute_tolerance,"
                "relative_tolerance,tolerance_exceed_count,pass\n";
    case_csv << std::scientific << std::setprecision(9);
    for (const auto& result : cases) {
        for (const auto& metric : result.metrics) {
            const bool metric_pass = metric.tolerance_exceed_count == 0 &&
                    (metric.name.rfind("adam_delta_", 0) != 0 ||
                     metric.reference_nonzero_count != 0);
            case_csv << result.case_id << ',' << result.occupancy_regime << ','
                     << result.state_variant << ',' << result.camera_variant
                     << ',' << result.gaussian_count << ',' << metric.name
                     << ',' << metric.count << ','
                     << metric.reference_nonzero_count << ','
                     << metric.reference_max_abs << ',' << metric.max_abs
                     << ',' << metric.max_relative << ',' << metric.meanAbs()
                     << ',' << metric.rms() << ','
                     << metric.absolute_tolerance << ','
                     << metric.relative_tolerance << ','
                     << metric.tolerance_exceed_count << ','
                     << (metric_pass ? "true" : "false") << '\n';
        }
    }

    std::ofstream json(output_dir / "semantic_equivalence.json");
    json << std::scientific << std::setprecision(9)
         << "{\n  \"schema\": \"cross_3dgs.lowering_semantic_equivalence.v2\",\n"
         << "  \"reference\": \"" << reference_name << "\",\n"
         << "  \"candidate\": \"" << candidate_name << "\",\n"
         << "  \"case_count\": " << cases.size() << ",\n"
         << "  \"sh_degree\": 3,\n"
         << "  \"optimizer\": \"Adam\",\n"
         << "  \"optimizer_step\": 3000,\n"
         << "  \"optimizer_initial_moments\": \"fresh_zero_initialized_program_per_case\",\n"
         << "  \"optimizer_compared_quantity\": \"post_step_parameters_minus_initial_parameters\",\n"
         << "  \"forward_channels\": \"RGBA\",\n"
         << "  \"loss\": \"mean_L1_RGB\",\n"
         << "  \"acceptance\": \"abs_error <= abs_tol + rel_tol * abs(reference); every Adam parameter group must have at least one nonzero reference update\",\n"
         << "  \"tolerance_exceed_count\": " << overall_exceed_count
         << ",\n  \"optimizer_groups_without_updates\": "
         << optimizer_groups_without_updates << ",\n"
         << "  \"pass\": " << (pass ? "true" : "false")
         << ",\n  \"metrics\": [\n";
    for (size_t i = 0; i < metrics.size(); i++) {
        const auto& metric = metrics[i];
        const bool metric_pass = metric.tolerance_exceed_count == 0 &&
                (metric.name.rfind("adam_delta_", 0) != 0 ||
                 metric.reference_nonzero_count != 0);
        json << "    {\"metric\": \"" << metric.name
             << "\", \"element_count\": " << metric.count
             << ", \"reference_nonzero_count\": "
             << metric.reference_nonzero_count
             << ", \"reference_max_abs\": " << metric.reference_max_abs
             << ", \"max_abs_error\": " << metric.max_abs
             << ", \"max_relative_error\": " << metric.max_relative
             << ", \"mean_abs_error\": " << metric.meanAbs()
             << ", \"rms_error\": " << metric.rms()
             << ", \"absolute_tolerance\": "
             << metric.absolute_tolerance
             << ", \"relative_tolerance\": "
             << metric.relative_tolerance
             << ", \"tolerance_exceed_count\": "
             << metric.tolerance_exceed_count
             << ", \"worst_case_id\": \"" << metric.worst_case_id
             << "\", \"worst_element_index\": "
             << metric.worst_element_index
             << ", \"worst_reference\": " << metric.worst_reference
             << ", \"worst_candidate\": " << metric.worst_candidate
             << ", \"pass\": " << (metric_pass ? "true" : "false")
             << "}" << (i + 1 == metrics.size() ? "\n" : ",\n");
    }
    json << "  ],\n  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); i++) {
        const auto& result = cases[i];
        size_t exceed_count = 0;
        size_t groups_without_updates = 0;
        for (const auto& metric : result.metrics) {
            exceed_count += metric.tolerance_exceed_count;
            if (metric.name.rfind("adam_delta_", 0) == 0 &&
                metric.reference_nonzero_count == 0)
                groups_without_updates++;
        }
        json << "    {\"case_id\": \"" << result.case_id
             << "\", \"occupancy_regime\": \""
             << result.occupancy_regime << "\", \"state_variant\": "
             << result.state_variant << ", \"camera_variant\": "
             << result.camera_variant << ", \"gaussian_count\": "
             << result.gaussian_count << ", \"tolerance_exceed_count\": "
             << exceed_count << ", \"optimizer_groups_without_updates\": "
             << groups_without_updates << ", \"pass\": "
             << (exceed_count == 0 && groups_without_updates == 0 ? "true"
                                                                  : "false")
             << "}" << (i + 1 == cases.size() ? "\n" : ",\n");
    }
    json << "  ]\n}\n";
    return pass;
}

bool runGradientMatrix(const std::filesystem::path& output_dir,
                       const std::string& candidate_name) {
    constexpr std::string_view kSplitSuffix = "_split_sh_rest";
    const bool split_storage = candidate_name.ends_with(kSplitSuffix);
    // Production-footprint series: the deployed Adreno740 standard-quality
    // backward against the exact-pair-gather semantic reference at the
    // one-million-capacity layout, so the range-derived 6+6 SH-rest split and
    // the split projection/Adam it forces are exercised by a one-step
    // series rather than only by 30k completion.
    constexpr std::string_view kNegativeControlSuffix = "_negative_control";
    // Split-axis negative control: the production-footprint tuple with the
    // candidate leg's shard-B SH-rest gradient writes deliberately dropped.
    // The gate is expected to FAIL it; it demonstrates the production-
    // footprint conformance check is falsifiable on the split axis and must
    // never enter a conformance series.
    const bool production_footprint_negative_control =
            candidate_name ==
            "per_splat_cas_production_footprint_negative_control";
    const bool production_footprint =
            candidate_name == "per_splat_cas_production_footprint" ||
            production_footprint_negative_control;
    const std::string logical_candidate = split_storage
            ? candidate_name.substr(
                    0, candidate_name.size() - kSplitSuffix.size())
            : candidate_name;
    std::vector<GradientCaseResult> cases;
    std::vector<SemanticCaseResult> semantic_cases;
    std::vector<ErrorGroup> aggregate = {{"mean_xyz"},
                                         {"scale_xyz"},
                                         {"opacity"},
                                         {"rotation_xyzw"},
                                         {"sh_dc_rgb"},
                                         {"sh_rest_45"}};
    auto semantic_aggregate = makeSemanticMetrics();
    const auto context = cross3dgs::gpu::CreateContext(false);
    for (const auto& regime : kOccupancyRegimes) {
        // The production footprint keeps the tiny synthetic active states but
        // allocates the standard-quality one-million capacity, which is what
        // decides the SH-rest storage layout (capacity-derived footprint, not
        // active count).
        const GaussianForwardConfig base = {
                .width = kGradientWidth,
                .height = kGradientHeight,
                .gaussian_capacity = production_footprint
                        ? size_t{1'000'000}
                        : regime.gaussian_count,
                .tile_size = 16,
                .tile_bin_capacity = 128,
                .enable_backward = true,
                .sh_degree = 3,
                .enable_pass_profiling = false};
        auto scatter_config=base;
        auto gather_config=base;
        scatter_config.atomic_free_exact_backward = !split_storage;
        if (split_storage) {
            // Storage-only conformance: compare the same safe backward
            // candidate with monolithic versus 6+6 SH-rest storage. This
            // deliberately does not resurrect an Adreno scatter candidate
            // that the runtime ledger rejects.
            scatter_config.atomic_free_exact_backward = true;
            gather_config.atomic_free_exact_backward = true;
            gather_config.force_split_sh_rest_storage = true;
            if (logical_candidate == "per_splat_pair_reduce") {
                scatter_config.force_per_splat_backward = true;
                gather_config.force_per_splat_backward = true;
            } else if (logical_candidate != "exact_pair_gather") {
                throw std::invalid_argument(
                        "split SH-rest conformance requires an atomic-free "
                        "candidate: " + candidate_name);
            }
        } else if (production_footprint) {
            // Reference leg: exact pair gather at the same one-million
            // layout. Candidate leg: the deployed per-splat + u32-CAS
            // backward. Both legs auto-split SH-rest on a device whose
            // storage range the capacity exceeds; assertions below fail
            // closed if the split does not engage, rather than silently
            // testing the monolithic layout.
            scatter_config.atomic_free_exact_backward = true;
            gather_config.force_per_splat_backward = true;
            gather_config.force_cas_scatter_backward = true;
            // Probe-only permission as for the small per_splat_cas tuple;
            // production schedule admission remains ledger-side.
            gather_config.unsafe_allow_rejected_schedule = true;
            if (production_footprint_negative_control) {
                gather_config
                        .unsafe_negative_control_zero_sh_rest_shard_b = true;
            }
        } else if(logical_candidate=="exact_pair_gather") {
            scatter_config.atomic_free_exact_backward=false;
            scatter_config.force_pixel_subgroup_backward=true;
            gather_config.atomic_free_exact_backward=true;
        } else if(logical_candidate=="pixel_subgroup_cas") {
            gather_config.force_pixel_subgroup_backward=true;
            gather_config.force_cas_scatter_backward=true;
        } else if(logical_candidate=="pixel_subgroup_native") {
            gather_config.force_pixel_subgroup_backward=true;
        } else if(logical_candidate=="per_splat_native") {
            gather_config.force_per_splat_backward=true;
        } else if(logical_candidate=="per_splat_cas") {
            gather_config.force_per_splat_backward=true;
            gather_config.force_cas_scatter_backward=true;
            // Probe-only permission for the tiny non-optimizer conformance
            // tuple; production schedule admission remains unchanged.
            gather_config.unsafe_allow_rejected_schedule=true;
        } else if(logical_candidate=="per_splat_pair_reduce") {
            gather_config.atomic_free_exact_backward=true;
            gather_config.force_per_splat_backward=true;
        } else if(logical_candidate=="per_splat_pair_reduce_negative_control") {
            // Negative control: identical per-splat pair-reduce candidate
            // with one tile-Gaussian partial deliberately dropped in the
            // candidate's reduce. The gate is expected to FAIL this candidate;
            // it exists only to demonstrate the conformance check is
            // falsifiable and must never enter a conformance series.
            gather_config.atomic_free_exact_backward=true;
            gather_config.force_per_splat_backward=true;
            gather_config.unsafe_negative_control_drop_partial=true;
        } else if(logical_candidate=="tiled_8x8_native") {
            gather_config.force_tiled_backward=true;
        } else if(logical_candidate=="tiled_8x8_cas") {
            gather_config.force_tiled_backward=true;
            gather_config.force_cas_scatter_backward=true;
        } else {
            throw std::invalid_argument("unknown lowering candidate: "+
                                        candidate_name);
        }
        GaussianForwardProgram scatter(scatter_config, context);
        GaussianForwardProgram gather(gather_config, context);
        const auto candidate_schedule=gather.schedule();
        const bool schedule_matches=
                (logical_candidate=="exact_pair_gather" &&
                 candidate_schedule.backward_work_assignment=="pair_partial" &&
                 candidate_schedule.backward_global_merge==
                         "gaussian_major_fixed_reduce") ||
                (logical_candidate=="pixel_subgroup_native" &&
                 candidate_schedule.backward_work_assignment==
                         "pixel_stream_subgroup" &&
                 candidate_schedule.backward_global_merge=="native_atomic") ||
                (logical_candidate=="pixel_subgroup_cas" &&
                 candidate_schedule.backward_work_assignment==
                         "pixel_stream_subgroup" &&
                 candidate_schedule.backward_global_merge=="u32_cas") ||
                (logical_candidate=="per_splat_native" &&
                 candidate_schedule.backward_work_assignment=="per_splat" &&
                 candidate_schedule.backward_global_merge=="native_atomic") ||
                (logical_candidate=="per_splat_cas" &&
                 candidate_schedule.backward_work_assignment=="per_splat" &&
                 candidate_schedule.backward_global_merge=="u32_cas") ||
                (production_footprint &&
                 candidate_schedule.backward_work_assignment=="per_splat" &&
                 candidate_schedule.backward_global_merge=="u32_cas") ||
                (logical_candidate=="per_splat_pair_reduce" &&
                 candidate_schedule.backward_work_assignment=="per_splat" &&
                 candidate_schedule.backward_global_merge==
                         "gaussian_major_fixed_reduce") ||
                (logical_candidate==
                         "per_splat_pair_reduce_negative_control" &&
                 candidate_schedule.backward_work_assignment=="per_splat" &&
                 candidate_schedule.backward_global_merge==
                         "gaussian_major_fixed_reduce") ||
                (logical_candidate=="tiled_8x8_native" &&
                 candidate_schedule.backward_work_assignment=="tiled_8x8" &&
                 candidate_schedule.backward_global_merge=="native_atomic") ||
                (logical_candidate=="tiled_8x8_cas" &&
                 candidate_schedule.backward_work_assignment=="tiled_8x8" &&
                 candidate_schedule.backward_global_merge=="u32_cas");
        if(!schedule_matches) {
            throw std::runtime_error("candidate force did not select requested "
                                     "canonical schedule: "+candidate_name);
        }
        if (production_footprint) {
            const auto reference_schedule = scatter.schedule();
            if (!candidate_schedule.split_sh_rest_storage ||
                !reference_schedule.split_sh_rest_storage) {
                throw std::runtime_error(
                        "production-footprint series requires the "
                        "range-derived 6+6 SH-rest split on both legs; this "
                        "device does not force it at the one-million "
                        "capacity");
            }
        }

        for (size_t state_variant = 0; state_variant < kStateVariantCount;
             state_variant++) {
            const auto state = makeGradientState(regime, state_variant);
            for (size_t camera_variant = 0;
                 camera_variant < kCameraVariantCount; camera_variant++) {
                const std::string case_id =
                        std::string(regime.name) + "-s" +
                        std::to_string(state_variant) + "-c" +
                        std::to_string(camera_variant);
                const auto camera = makeGradientCamera(camera_variant);
                const auto target = makeGradientTarget(camera_variant);
                // Re-upload the identical initial state for every case. The
                // non-optimizer programs do not mutate it, but doing this
                // explicitly makes case independence auditable.
                scatter.uploadState(state);
                scatter.uploadTarget(target);
                scatter.setCamera(camera);
                scatter.debugSetStep(3000);
                scatter.execute();
                const auto scatter_image = scatter.downloadColor();
                const float scatter_loss = scatter.downloadMeanL1();
                gather.uploadState(state);
                gather.uploadTarget(target);
                gather.setCamera(camera);
                gather.debugSetStep(3000);
                gather.execute();
                const auto gather_image = gather.downloadColor();
                const float gather_loss = gather.downloadMeanL1();
                const auto scatter_stats = scatter.stats();
                const auto gather_stats = gather.stats();
                if (scatter_stats.overflow_count != 0 ||
                    gather_stats.overflow_count != 0)
                    throw std::runtime_error(
                            "tile-bin overflow in gradient case " + case_id);
                if (scatter_stats.tile_reference_count !=
                    gather_stats.tile_reference_count)
                    throw std::runtime_error(
                            "scatter/gather pair-count mismatch in " + case_id);
                auto groups = compareGradientGroups(
                        scatter.downloadGradients(),
                        gather.downloadGradients(), case_id);

                auto semantic_metrics = makeSemanticMetrics();
                compareForwardImage(scatter_image, gather_image, case_id,
                                    semantic_metrics[0]);
                semantic_metrics[1].add(scatter_loss, gather_loss, case_id, 0);

                // Fresh programs guarantee zero-initialized Adam moments for
                // every case; uploadState alone intentionally does not reset
                // optimizer moment buffers. Both execute exactly one update
                // from the same state, camera, target, and optimizer step.
                auto optimizer_scatter_config = scatter_config;
                optimizer_scatter_config.enable_optimizer = true;
                GaussianForwardProgram optimizer_scatter(
                        optimizer_scatter_config, context);
                auto optimizer_gather_config = gather_config;
                optimizer_gather_config.enable_optimizer = true;
                GaussianForwardProgram optimizer_gather(
                        optimizer_gather_config, context);
                if (production_footprint) {
                    const auto scatter_opt = optimizer_scatter.schedule();
                    const auto gather_opt = optimizer_gather.schedule();
                    if (scatter_opt.projection_optimizer_fusion ||
                        gather_opt.projection_optimizer_fusion ||
                        !scatter_opt.split_sh_rest_storage ||
                        !gather_opt.split_sh_rest_storage) {
                        throw std::runtime_error(
                                "production-footprint optimizer legs must "
                                "run split SH-rest with split "
                                "projection/Adam");
                    }
                }
                optimizer_scatter.uploadState(state);
                optimizer_scatter.uploadTarget(target);
                optimizer_scatter.setCamera(camera);
                optimizer_scatter.debugSetStep(3000);
                optimizer_scatter.execute();
                const auto scatter_updated = optimizer_scatter.downloadState();
                optimizer_gather.uploadState(state);
                optimizer_gather.uploadTarget(target);
                optimizer_gather.setCamera(camera);
                optimizer_gather.debugSetStep(3000);
                optimizer_gather.execute();
                const auto gather_updated = optimizer_gather.downloadState();
                const auto optimizer_scatter_stats = optimizer_scatter.stats();
                const auto optimizer_gather_stats = optimizer_gather.stats();
                if (optimizer_scatter_stats.overflow_count != 0 ||
                    optimizer_gather_stats.overflow_count != 0)
                    throw std::runtime_error(
                            "tile-bin overflow in Adam case " + case_id);
                if (optimizer_scatter_stats.tile_reference_count !=
                    optimizer_gather_stats.tile_reference_count)
                    throw std::runtime_error(
                            "scatter/gather Adam pair-count mismatch in " +
                            case_id);
                compareAdamUpdateGroups(state, scatter_updated, gather_updated,
                                        case_id, semantic_metrics);

                for (size_t group = 0; group < groups.size(); group++)
                    aggregate[group].merge(groups[group]);
                for (size_t metric = 0; metric < semantic_metrics.size();
                     metric++)
                    semantic_aggregate[metric].merge(
                            semantic_metrics[metric]);
                cases.push_back({case_id,
                                 regime.name,
                                 state_variant,
                                 camera_variant,
                                 regime.gaussian_count,
                                 scatter_stats.tile_reference_count,
                                 scatter_stats.max_tile_count,
                                 std::move(groups)});
                semantic_cases.push_back(
                        {case_id,
                         regime.name,
                         state_variant,
                         camera_variant,
                         regime.gaussian_count,
                         std::move(semantic_metrics)});
                std::cout << "gradient case=" << case_id
                          << " pairs=" << scatter_stats.tile_reference_count
                          << " max_tile=" << scatter_stats.max_tile_count
                          << std::endl;
            }
        }
    }
    if (cases.size() < 100)
        throw std::runtime_error("gradient matrix produced fewer than 100 cases");
    const std::string reference_name = split_storage
            ? logical_candidate + "_monolithic_sh_rest"
            : logical_candidate=="exact_pair_gather"
                    ? "pixel_stream_subgroup_native"
                    : "pair_partial_gaussian_major_fixed_reduce";
    const bool gradient_pass =
            writeGradientReports(output_dir, cases, aggregate,
                                 reference_name,
                                 candidate_name);
    const bool semantic_pass = writeSemanticReports(
            output_dir, semantic_cases, semantic_aggregate,
            reference_name,
            candidate_name);
    return gradient_pass && semantic_pass;
}

ScalingRow runScalingPoint(size_t count, std::ofstream& raw_csv) {
    constexpr uint32_t kWidth = 256;
    constexpr uint32_t kHeight = 256;
    constexpr int kWarmup = 3;
    constexpr int kRepeats = 12;
    auto state = makeScalingState(count);
    GaussianForwardProgram program(
            {.width = kWidth,
             .height = kHeight,
             .gaussian_capacity = count,
             .tile_size = 16,
             .tile_bin_capacity = 512,
             .enable_backward = true,
             .sh_degree = 0,
             .atomic_free_exact_backward = true,
             .enable_pass_profiling = true});
    program.uploadState(state);
    program.uploadTarget(zeroTarget(kWidth, kHeight));
    program.setCamera(makeCamera(kWidth, kHeight, 180.0f));
    for (int i = 0; i < kWarmup; i++) program.execute();

    std::vector<double> tile_times, reduce_times, pair_times, host_times;
    GaussianForwardStats stats;
    for (int repeat = 0; repeat < kRepeats; repeat++) {
        const auto begin = std::chrono::steady_clock::now();
        program.execute();
        const auto end = std::chrono::steady_clock::now();
        stats = program.stats();
        const double tiles = passMilliseconds(
                stats, "raster_backward_exact_tiles");
        const double reduce = passMilliseconds(
                stats, "raster_backward_exact_reduce");
        const double host =
                std::chrono::duration<double, std::milli>(end - begin).count();
        tile_times.push_back(tiles);
        reduce_times.push_back(reduce);
        pair_times.push_back(tiles + reduce);
        host_times.push_back(host);
        constexpr uint32_t kTileCount = (kWidth / 16) * (kHeight / 16);
        raw_csv << count << ',' << repeat << ',' << stats.tile_reference_count
                << ',' << stats.max_tile_count << ','
                << static_cast<double>(stats.tile_reference_count) / kTileCount
                << ','
                << static_cast<double>(stats.tile_reference_count) / count
                << ',' << stats.pair_capacity << ','
                << stats.atomic_free_pair_scratch_used_bytes << ','
                << stats.atomic_free_pair_scratch_bytes << ',' << tiles << ','
                << reduce << ',' << tiles + reduce << ',' << host << '\n';
    }
    if (stats.overflow_count != 0)
        throw std::runtime_error("tile-bin overflow in scaling probe");
    constexpr uint32_t kTileCount = (kWidth / 16) * (kHeight / 16);
    return {.gaussian_count = count,
            .pair_count = stats.tile_reference_count,
            .pair_capacity = stats.pair_capacity,
            .scratch_used_bytes = stats.atomic_free_pair_scratch_used_bytes,
            .scratch_allocated_bytes = stats.atomic_free_pair_scratch_bytes,
            .total_allocated_bytes = stats.current_memory_bytes,
            .tile_count = kTileCount,
            .max_tile_count = stats.max_tile_count,
            .mean_pairs_per_tile =
                    static_cast<double>(stats.tile_reference_count) / kTileCount,
            .pairs_per_gaussian =
                    static_cast<double>(stats.tile_reference_count) / count,
            .max_tile_capacity_fraction =
                    static_cast<double>(stats.max_tile_count) / 512.0,
            .median_tiles_gpu_ms = median(tile_times),
            .median_reduce_gpu_ms = median(reduce_times),
            .median_pair_backward_gpu_ms = median(pair_times),
            .median_execute_host_ms = median(host_times)};
}

void writeScalingReports(const std::filesystem::path& output_dir) {
    std::ofstream raw(output_dir / "scratch_pair_scaling_raw.csv");
    raw << "gaussian_count,repeat,pair_count,max_tile_count,"
           "mean_pairs_per_tile,pairs_per_gaussian,pair_capacity,scratch_used_bytes,"
           "scratch_allocated_bytes,tiles_gpu_ms,reduce_gpu_ms,"
           "pair_backward_gpu_ms,execute_host_ms\n";
    raw << std::fixed << std::setprecision(9);
    std::vector<ScalingRow> rows;
    for (const size_t count : {256u, 1024u, 4096u, 16384u, 54275u}) {
        std::cout << "scaling gaussian_count=" << count << std::endl;
        rows.push_back(runScalingPoint(count, raw));
    }

    std::ofstream csv(output_dir / "scratch_pair_scaling.csv");
    csv << "gaussian_count,pair_count,tile_count,max_tile_count,"
           "mean_pairs_per_tile,pairs_per_gaussian,"
           "max_tile_capacity_fraction,pair_capacity,bytes_per_pair,"
           "scratch_used_bytes,scratch_allocated_bytes,total_allocated_bytes,"
           "median_tiles_gpu_ms,median_reduce_gpu_ms,"
           "median_pair_backward_gpu_ms,median_execute_host_ms\n";
    csv << std::fixed << std::setprecision(9);
    for (const auto& row : rows) {
        csv << row.gaussian_count << ',' << row.pair_count << ','
            << row.tile_count << ',' << row.max_tile_count << ','
            << row.mean_pairs_per_tile << ',' << row.pairs_per_gaussian << ','
            << row.max_tile_capacity_fraction << ',' << row.pair_capacity
            << ",48," << row.scratch_used_bytes << ','
            << row.scratch_allocated_bytes << ',' << row.total_allocated_bytes
            << ',' << row.median_tiles_gpu_ms << ','
            << row.median_reduce_gpu_ms << ','
            << row.median_pair_backward_gpu_ms << ','
            << row.median_execute_host_ms << '\n';
    }

    std::ofstream json(output_dir / "scratch_pair_scaling.json");
    json << std::fixed << std::setprecision(9)
         << "{\n  \"schema\": \"cross_3dgs.atomic_free_pair_scaling.v2\",\n"
         << "  \"width\": 256,\n  \"height\": 256,\n"
         << "  \"tile_size\": 16,\n  \"tile_bin_capacity\": 512,\n"
         << "  \"bytes_per_pair\": 48,\n"
         << "  \"warmup_iterations\": 3,\n"
         << "  \"measured_iterations\": 12,\n"
         << "  \"gpu_time_scope\": \"exact_tiles_plus_exact_reduce\",\n"
         << "  \"rows\": [\n";
    for (size_t i = 0; i < rows.size(); i++) {
        const auto& row = rows[i];
        json << "    {\"gaussian_count\": " << row.gaussian_count
             << ", \"pair_count\": " << row.pair_count
             << ", \"tile_count\": " << row.tile_count
             << ", \"max_tile_count\": " << row.max_tile_count
             << ", \"mean_pairs_per_tile\": " << row.mean_pairs_per_tile
             << ", \"pairs_per_gaussian\": " << row.pairs_per_gaussian
             << ", \"max_tile_capacity_fraction\": "
             << row.max_tile_capacity_fraction
             << ", \"pair_capacity\": " << row.pair_capacity
             << ", \"scratch_used_bytes\": " << row.scratch_used_bytes
             << ", \"scratch_allocated_bytes\": "
             << row.scratch_allocated_bytes
             << ", \"total_allocated_bytes\": "
             << row.total_allocated_bytes
             << ", \"median_tiles_gpu_ms\": "
             << row.median_tiles_gpu_ms
             << ", \"median_reduce_gpu_ms\": "
             << row.median_reduce_gpu_ms
             << ", \"median_pair_backward_gpu_ms\": "
             << row.median_pair_backward_gpu_ms
             << ", \"median_execute_host_ms\": "
             << row.median_execute_host_ms << "}"
             << (i + 1 == rows.size() ? "\n" : ",\n");
    }
    json << "  ]\n}\n";
}

}  // namespace

#ifdef CROSS_3DGS_ATOMIC_FREE_PROBE_LIBRARY
namespace cross3dgs_probe {

bool RunLoweringCandidateConformance(
        const std::filesystem::path& output_dir,
        const std::string& candidate_name) {
    std::filesystem::create_directories(output_dir);
    return runGradientMatrix(output_dir, candidate_name);
}

}  // namespace cross3dgs_probe
#else
int main(int argc, char** argv) {
    try {
        if ((argc != 3 && argc != 5) ||
            std::string(argv[1]) != "--output-dir" ||
            (argc==5 && std::string(argv[3])!="--candidate")) {
            std::cerr << "usage: cross_3dgs_atomic_free_probe --output-dir DIR "
                         "[--candidate exact_pair_gather|pixel_subgroup_native|"
                         "pixel_subgroup_cas|"
                         "per_splat_native|per_splat_cas|per_splat_pair_reduce|"
                         "per_splat_pair_reduce_negative_control|"
                         "per_splat_cas_production_footprint|"
                         "per_splat_cas_production_footprint_negative_control|"
                         "tiled_8x8_native|tiled_8x8_cas]\n";
            return 2;
        }
        const std::filesystem::path output_dir(argv[2]);
        const std::string candidate_name=argc==5?argv[4]:"exact_pair_gather";
        std::filesystem::create_directories(output_dir);

        const bool gradient_pass = runGradientMatrix(output_dir,candidate_name);
        if(candidate_name=="exact_pair_gather") writeScalingReports(output_dir);
        if (!gradient_pass) {
            std::cerr << "gradient or semantic acceptance failed\n";
            return 1;
        }
        std::cout << "lowering candidate conformance passed: "
                  << candidate_name << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "atomic-free probe failed: " << error.what() << '\n';
        return 1;
    }
}
#endif
