// Device-independent CPU double-precision oracle for the one-step
// update-conformance suite.
//
// A tiny two-Gaussian scene with a fixed sort order is rendered by a CPU
// double-precision reimplementation of the exact forward semantics
// (projection with clamped Jacobian and 0.3 dilation, subtile masks,
// front-to-back compositing with the 1/255 alpha and 1e-4 transmittance
// thresholds, mean-L1 loss). Reference gradients come from double-precision
// central finite differences of that CPU forward, so neither the forward
// values nor the gradients depend on any GPU implementation. The GPU exact
// scatter and atomic-free exact gather backward are then checked against
// this oracle, breaking the GPU-vs-GPU circularity of the conformance suite.
//
// Tolerances (fixed before the first reported run of this oracle):
//   forward RGBA (float32 GPU vs double CPU): max |delta| <= 1e-5
//   mean-L1 loss:                             |delta| <= 1e-6
//   gradients (GPU analytic vs CPU double FD): per scalar
//       |delta| <= 1e-6 + 1e-3 * |oracle|
// The gradient tolerance is wider than the GPU-vs-GPU conformance contract
// (1e-5 + 1e-4 |ref|) because the GPU forward accumulates float32 rounding
// that the double oracle does not share.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "cross3dgs/gaussian_forward.h"

namespace {

using cross3dgs::Camera;
using cross3dgs::CameraModel;
using cross3dgs::FloatImage;
using cross3dgs::GaussianForwardProgram;
using cross3dgs::GaussianForwardConfig;
using cross3dgs::GaussianStateCpu;

constexpr double kAlphaFloor = 0.00392156862745;
constexpr double kShC0 = 0.28209479177387814;

struct OracleCase {
    std::string name;
    uint32_t width = 48;
    uint32_t height = 32;
    uint32_t sh_degree = 0;
    uint32_t step = 0;
    // 2 runs a second optimizer step from the first step's state, so the
    // compared update starts from nonzero Adam moments.
    uint32_t adam_steps = 1;
    GaussianStateCpu state;
    Camera camera;
};

glm::dmat3 QuaternionRotation(glm::dvec4 q) {
    q = glm::normalize(q);
    const double r = q.x, x = q.y, y = q.z, z = q.w;
    glm::dmat3 m;
    m[0] = {1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + r * z),
            2.0 * (x * z - r * y)};
    m[1] = {2.0 * (x * y - r * z), 1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + r * x)};
    m[2] = {2.0 * (x * z + r * y), 2.0 * (y * z - r * x),
            1.0 - 2.0 * (x * x + y * y)};
    return m;
}

void ShBasis(const glm::dvec3& d, double* b) {
    const double x = d.x, y = d.y, z = d.z;
    const double xx = x * x, yy = y * y, zz = z * z;
    b[0] = -0.4886025119029199 * y;
    b[1] = 0.4886025119029199 * z;
    b[2] = -0.4886025119029199 * x;
    b[3] = 1.0925484305920792 * x * y;
    b[4] = -1.0925484305920792 * y * z;
    b[5] = 0.31539156525252005 * (2.0 * zz - xx - yy);
    b[6] = -1.0925484305920792 * x * z;
    b[7] = 0.5462742152960396 * (xx - yy);
    b[8] = -0.5900435899266435 * y * (3.0 * xx - yy);
    b[9] = 2.890611442640554 * x * y * z;
    b[10] = -0.4570457994644658 * y * (4.0 * zz - xx - yy);
    b[11] = 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    b[12] = -0.4570457994644658 * x * (4.0 * zz - xx - yy);
    b[13] = 1.445305721320277 * z * (xx - yy);
    b[14] = -0.5900435899266435 * x * (xx - 3.0 * yy);
}

bool EllipseOverlapsRect(const glm::dvec3& q, double x0, double x1,
                         double y0, double y1) {
    if (x0 <= 0.0 && x1 >= 0.0 && y0 <= 0.0 && y1 >= 0.0) return true;
    double minimum = 1e30;
    double x = std::clamp(-q.y * y0 / q.x, x0, x1);
    minimum = std::min(minimum,
                       q.x * x * x + 2.0 * q.y * x * y0 + q.z * y0 * y0);
    x = std::clamp(-q.y * y1 / q.x, x0, x1);
    minimum = std::min(minimum,
                       q.x * x * x + 2.0 * q.y * x * y1 + q.z * y1 * y1);
    double y = std::clamp(-q.y * x0 / q.z, y0, y1);
    minimum = std::min(minimum,
                       q.x * x0 * x0 + 2.0 * q.y * x0 * y + q.z * y * y);
    y = std::clamp(-q.y * x1 / q.z, y0, y1);
    minimum = std::min(minimum,
                       q.x * x1 * x1 + 2.0 * q.y * x1 * y + q.z * y * y);
    return minimum <= 1.0001;
}

struct ProjectedGaussian {
    bool visible = false;
    glm::dvec2 screen{};
    double depth = 0.0;
    glm::dvec3 conic{};
    double opacity = 0.0;
    glm::dvec3 color{};
    glm::ivec2 tile_min{};
    glm::ivec2 tile_max{};
};

// CPU double replica of ProjectionShader + compute_subtile_mask +
// RasterShader compositing + the mean-L1 loss against a zero target.
struct CpuForwardResult {
    std::vector<glm::dvec4> image;  // RGBA per pixel
    double mean_l1 = 0.0;
};

CpuForwardResult CpuForward(const OracleCase& scene) {
    const uint32_t width = scene.width, height = scene.height;
    const uint32_t grid_width = (width + 15) / 16;
    const uint32_t grid_height = (height + 15) / 16;
    const glm::dmat4 w2c(scene.camera.world_to_camera);
    const double fx = scene.camera.fx, fy = scene.camera.fy;
    const double cx = scene.camera.cx, cy = scene.camera.cy;
    const uint32_t active_degree =
            std::min(scene.sh_degree, scene.step / 1000u);
    const size_t count = scene.state.size();

    std::vector<ProjectedGaussian> gaussians(count);
    for (size_t i = 0; i < count; i++) {
        auto& g = gaussians[i];
        const glm::dvec3 mean(scene.state.means[i]);
        const glm::dvec3 view =
                glm::dvec3(w2c * glm::dvec4(mean, 1.0));
        if (view.z <= 0.01) continue;
        const glm::dvec2 screen(fx * view.x / view.z + cx - 0.5,
                                fy * view.y / view.z + cy - 0.5);
        const glm::dvec4 so(scene.state.scales_opacity[i]);
        const glm::dvec3 scale(so);
        const glm::dmat3 rotation =
                QuaternionRotation(glm::dvec4(scene.state.rotations[i]));
        const glm::dmat3 covariance = rotation *
                glm::dmat3(scale.x * scale.x, 0, 0, 0, scale.y * scale.y,
                           0, 0, 0, scale.z * scale.z) *
                glm::transpose(rotation);
        const glm::dmat3 wvr(w2c);
        const glm::dvec3 row0(wvr[0][0], wvr[1][0], wvr[2][0]);
        const glm::dvec3 row1(wvr[0][1], wvr[1][1], wvr[2][1]);
        const glm::dvec3 row2(wvr[0][2], wvr[1][2], wvr[2][2]);
        const double inv_z = 1.0 / view.z;
        const double tan_fovx = 0.5 * width / fx;
        const double tan_fovy = 0.5 * height / fy;
        const double limit_x_positive = (width - cx) / fx + 0.3 * tan_fovx;
        const double limit_x_negative = cx / fx + 0.3 * tan_fovx;
        const double limit_y_positive = (height - cy) / fy + 0.3 * tan_fovy;
        const double limit_y_negative = cy / fy + 0.3 * tan_fovy;
        const double covariance_x =
                std::clamp(view.x * inv_z, -limit_x_negative,
                           limit_x_positive) * view.z;
        const double covariance_y =
                std::clamp(view.y * inv_z, -limit_y_negative,
                           limit_y_positive) * view.z;
        const glm::dvec3 t0 = (fx * inv_z) * row0 +
                (-fx * covariance_x * inv_z * inv_z) * row2;
        const glm::dvec3 t1 = (fy * inv_z) * row1 +
                (-fy * covariance_y * inv_z * inv_z) * row2;
        const glm::dvec3 cov_t0 = covariance * t0;
        const glm::dvec3 cov_t1 = covariance * t1;
        const double xx = glm::dot(t0, cov_t0) + 0.3;
        const double xy = glm::dot(t0, cov_t1);
        const double yy = glm::dot(t1, cov_t1) + 0.3;
        const double determinant = xx * yy - xy * xy;
        if (!(determinant > 0.0) || !std::isfinite(determinant)) continue;
        const double opacity = so.w;
        if (opacity <= kAlphaFloor) continue;
        const double extend =
                std::sqrt(2.0 * std::log(std::min(opacity, 1.0) /
                                         kAlphaFloor));
        const glm::dvec2 radii(
                std::ceil(std::sqrt(std::max(xx, 0.0)) * extend),
                std::ceil(std::sqrt(std::max(yy, 0.0)) * extend));
        glm::ivec2 minimum(
                static_cast<int>(std::floor((screen.x - radii.x) / 16.0)),
                static_cast<int>(std::floor((screen.y - radii.y) / 16.0)));
        glm::ivec2 maximum(
                static_cast<int>(std::ceil((screen.x + radii.x) / 16.0)),
                static_cast<int>(std::ceil((screen.y + radii.y) / 16.0)));
        minimum = glm::clamp(minimum, glm::ivec2(0),
                             glm::ivec2(grid_width, grid_height));
        maximum = glm::clamp(maximum, glm::ivec2(0),
                             glm::ivec2(grid_width, grid_height));
        if (minimum.x >= maximum.x || minimum.y >= maximum.y) continue;
        g.visible = true;
        g.screen = screen;
        g.depth = view.z;
        g.conic = glm::dvec3(yy, -xy, xx) / determinant;
        g.opacity = opacity;
        glm::dvec3 color = glm::dvec3(0.5) +
                kShC0 * glm::dvec3(scene.state.sh_dc[i]);
        if (active_degree > 0) {
            const glm::dmat3 world_rotation(w2c);
            const glm::dvec3 camera_position =
                    -glm::transpose(world_rotation) *
                    glm::dvec3(w2c[3]);
            double basis[15];
            ShBasis(glm::normalize(mean - camera_position), basis);
            const uint32_t coefficient_count =
                    active_degree == 1 ? 3u : (active_degree == 2 ? 8u
                                                                  : 15u);
            for (uint32_t c = 1; c <= coefficient_count; c++) {
                const uint32_t scalar = (c - 1) * 3;
                const auto sh_scalar = [&](uint32_t s) {
                    return double(scene.state
                                          .sh_rest[i * 12 + s / 4][s % 4]);
                };
                color += basis[c - 1] *
                        glm::dvec3(sh_scalar(scalar),
                                   sh_scalar(scalar + 1),
                                   sh_scalar(scalar + 2));
            }
        }
        g.color = glm::max(glm::dvec3(0.0), color);
        g.tile_min = minimum;
        g.tile_max = maximum;
    }

    // Pair emission + radix order: per tile, depth ascending (the sort key
    // is monotonic in view depth; depths are distinct in the oracle cases).
    struct Pair { double depth; size_t id; uint32_t mask; };
    std::vector<std::vector<Pair>> tiles(
            static_cast<size_t>(grid_width) * grid_height);
    for (size_t i = 0; i < count; i++) {
        const auto& g = gaussians[i];
        if (!g.visible) continue;
        const double extend_squared =
                2.0 * std::log(std::max(g.opacity, kAlphaFloor) /
                               kAlphaFloor);
        const glm::dvec3 normalized =
                g.conic / std::max(extend_squared, 1e-6);
        const double mask_det = normalized.x * normalized.z -
                normalized.y * normalized.y;
        for (int ty = g.tile_min.y; ty < g.tile_max.y; ty++) {
            for (int tx = g.tile_min.x; tx < g.tile_max.x; tx++) {
                const uint32_t tile = ty * grid_width + tx;
                uint32_t mask = 0;
                if (!(mask_det > 0.0)) {
                    mask = 0xffu;
                } else {
                    const glm::dvec2 tile_origin(tx * 16.0, ty * 16.0);
                    const glm::dvec2 center = g.screen - tile_origin;
                    for (uint32_t sy = 0; sy < 4; sy++) {
                        for (uint32_t sx = 0; sx < 4; sx++) {
                            const double x0 = sx * 4.0 - center.x;
                            const double y0 = sy * 4.0 - center.y;
                            if (EllipseOverlapsRect(normalized, x0,
                                                    x0 + 3.0, y0,
                                                    y0 + 3.0)) {
                                mask |= 1u << (sy * 4 + sx);
                            }
                        }
                    }
                }
                tiles[tile].push_back({g.depth, i, mask});
            }
        }
    }
    for (auto& pairs : tiles) {
        std::sort(pairs.begin(), pairs.end(),
                  [](const Pair& a, const Pair& b) {
                      return a.depth != b.depth ? a.depth < b.depth
                                                : a.id < b.id;
                  });
    }

    CpuForwardResult result;
    result.image.assign(static_cast<size_t>(width) * height,
                        glm::dvec4(0.0));
    double loss_sum = 0.0;
    for (uint32_t py = 0; py < height; py++) {
        for (uint32_t px = 0; px < width; px++) {
            const uint32_t tile =
                    (py / 16) * grid_width + (px / 16);
            const uint32_t subtile =
                    (py & 15u) / 4u * 4u + (px & 15u) / 4u;
            glm::dvec3 accumulated(0.0);
            double transmittance = 1.0;
            for (const auto& pair : tiles[tile]) {
                if ((pair.mask & (1u << subtile)) == 0) continue;
                const auto& g = gaussians[pair.id];
                const glm::dvec2 d =
                        glm::dvec2(px, py) - g.screen;
                const double power = -0.5 *
                        (g.conic.x * d.x * d.x +
                         2.0 * g.conic.y * d.x * d.y +
                         g.conic.z * d.y * d.y);
                if (power > 0.0) continue;
                const double alpha =
                        std::min(0.99, g.opacity * std::exp(power));
                if (alpha < kAlphaFloor) continue;
                const double next_transmittance =
                        transmittance * (1.0 - alpha);
                if (next_transmittance < 0.0001) break;
                accumulated += transmittance * alpha * g.color;
                transmittance = next_transmittance;
            }
            result.image[static_cast<size_t>(py) * width + px] =
                    glm::dvec4(accumulated, 1.0 - transmittance);
            loss_sum += (std::abs(accumulated.x) +
                         std::abs(accumulated.y) +
                         std::abs(accumulated.z)) / 3.0;
        }
    }
    result.mean_l1 = loss_sum / (static_cast<double>(width) * height);
    return result;
}

// Enumerate the differentiable scalars of the case (per Gaussian: mean xyz,
// scale xyz, opacity, rotation xyzw, SH-DC xyz, and 45 SH-rest scalars at
// SH3), with accessors into a GaussianStateCpu copy.
struct Scalar {
    std::string group;
    size_t gaussian;
    size_t component;
    std::function<float*(GaussianStateCpu&)> ref;
};

std::vector<Scalar> EnumerateScalars(const OracleCase& scene) {
    std::vector<Scalar> scalars;
    for (size_t i = 0; i < scene.state.size(); i++) {
        for (size_t c = 0; c < 3; c++) {
            scalars.push_back({"mean", i, c, [i, c](GaussianStateCpu& s) {
                                   return &s.means[i][static_cast<int>(c)];
                               }});
        }
        for (size_t c = 0; c < 3; c++) {
            scalars.push_back(
                    {"scale", i, c, [i, c](GaussianStateCpu& s) {
                         return &s.scales_opacity[i][static_cast<int>(c)];
                     }});
        }
        scalars.push_back({"opacity", i, 0, [i](GaussianStateCpu& s) {
                               return &s.scales_opacity[i].w;
                           }});
        for (size_t c = 0; c < 4; c++) {
            scalars.push_back(
                    {"rotation", i, c, [i, c](GaussianStateCpu& s) {
                         return &s.rotations[i][static_cast<int>(c)];
                     }});
        }
        for (size_t c = 0; c < 3; c++) {
            scalars.push_back({"sh_dc", i, c, [i, c](GaussianStateCpu& s) {
                                   return &s.sh_dc[i][static_cast<int>(c)];
                               }});
        }
        if (scene.sh_degree == 3) {
            for (size_t s = 0; s < 45; s++) {
                scalars.push_back(
                        {"sh_rest", i, s, [i, s](GaussianStateCpu& state) {
                             return &state.sh_rest[i * 12 + s / 4]
                                     [static_cast<int>(s % 4)];
                         }});
            }
        }
    }
    return scalars;
}

// Multiplies the central-difference step h; the reported oracle uses 1.0.
// Values away from 1.0 are for the FD step-sensitivity check only, so a
// truncation- or cancellation-dominated h cannot silently shape the oracle.
double FdEpsilonScale() {
    static const double scale = [] {
        if (const char* env = std::getenv("CROSS3DGS_FD_EPSILON_SCALE")) {
            const double parsed = std::atof(env);
            if (std::isfinite(parsed) && parsed > 0.0) return parsed;
        }
        return 1.0;
    }();
    return scale;
}

double FiniteDifference(const OracleCase& scene, const Scalar& scalar,
                        double step_scale = 1.0) {
    OracleCase perturbed = scene;
    float* value = scalar.ref(perturbed.state);
    const double base = *value;
    const double epsilon = step_scale * FdEpsilonScale() *
            std::max(1e-6, 1e-4 * std::abs(base));
    *value = static_cast<float>(base + epsilon);
    const double plus = CpuForward(perturbed).mean_l1;
    *value = static_cast<float>(base - epsilon);
    const double minus = CpuForward(perturbed).mean_l1;
    return (plus - minus) / (2.0 * epsilon);
}

double GpuGradient(const cross3dgs::GaussianGradientCpu& gradients,
                   const Scalar& scalar) {
    const int c = static_cast<int>(scalar.component);
    if (scalar.group == "mean") return gradients.means[scalar.gaussian][c];
    if (scalar.group == "scale")
        return gradients.scales_opacity[scalar.gaussian][c];
    if (scalar.group == "opacity")
        return gradients.scales_opacity[scalar.gaussian].w;
    if (scalar.group == "rotation")
        return gradients.rotations[scalar.gaussian][c];
    if (scalar.group == "sh_dc") return gradients.sh_dc[scalar.gaussian][c];
    return gradients.sh_rest[scalar.gaussian * 12 + scalar.component / 4]
            [static_cast<int>(scalar.component % 4)];
}

// Counts of scalars whose central difference changed with the step size.
struct FdRefinement {
    int refined = 0;     // h and h/4 disagreed; h/4 and h/16 agreed
    int unresolved = 0;  // h/4 and h/16 also disagreed; h/16 is used
};

// With `refinement`, a scalar whose difference at h and h/4 differs by more
// than 5% (the stencil straddles a coverage boundary, not rounding noise)
// takes the smaller step instead; without it, every scalar uses h as before.
cross3dgs::GaussianGradientCpu CpuFiniteDifferenceGradients(
        const OracleCase& scene, const std::vector<Scalar>& scalars,
        FdRefinement* refinement = nullptr) {
    const auto agree = [](double a, double b) {
        return std::abs(a - b) <= 1e-5 + 0.05 * std::abs(b);
    };
    cross3dgs::GaussianGradientCpu gradients;
    gradients.means.assign(scene.state.size(), glm::vec4(0.0f));
    gradients.scales_opacity.assign(scene.state.size(), glm::vec4(0.0f));
    gradients.rotations.assign(scene.state.size(), glm::vec4(0.0f));
    gradients.sh_dc.assign(scene.state.size(), glm::vec4(0.0f));
    gradients.sh_rest.assign(scene.state.size() * 12, glm::vec4(0.0f));
    for (const auto& scalar : scalars) {
        double difference = FiniteDifference(scene, scalar);
        if (refinement) {
            const double quarter = FiniteDifference(scene, scalar, 0.25);
            if (!agree(difference, quarter)) {
                difference = FiniteDifference(scene, scalar, 0.0625);
                if (agree(quarter, difference)) refinement->refined++;
                else refinement->unresolved++;
            }
        }
        const float value = static_cast<float>(difference);
        const int c = static_cast<int>(scalar.component);
        if (scalar.group == "mean") gradients.means[scalar.gaussian][c] = value;
        else if (scalar.group == "scale")
            gradients.scales_opacity[scalar.gaussian][c] = value;
        else if (scalar.group == "opacity")
            gradients.scales_opacity[scalar.gaussian].w = value;
        else if (scalar.group == "rotation")
            gradients.rotations[scalar.gaussian][c] = value;
        else if (scalar.group == "sh_dc")
            gradients.sh_dc[scalar.gaussian][c] = value;
        else
            gradients.sh_rest[scalar.gaussian * 12 + scalar.component / 4]
                    [static_cast<int>(scalar.component % 4)] = value;
    }
    return gradients;
}

// Adam moments carried between CpuAdamUpdate calls, one entry per updated
// scalar in the update's fixed visiting order.
struct AdamMoments {
    std::vector<double> m;
    std::vector<double> v;
    size_t cursor = 0;
};

GaussianStateCpu CpuAdamUpdate(
        const OracleCase& scene,
        const cross3dgs::GaussianGradientCpu& gradients,
        AdamMoments* moments = nullptr) {
    GaussianStateCpu updated = scene.state;
    const double step = static_cast<double>(scene.step) + 1.0;
    const double inverse_bias_m = 1.0 / (1.0 - std::pow(0.9, step));
    const double inverse_bias_v = 1.0 / (1.0 - std::pow(0.999, step));
    const double progress = std::min(double(scene.step) / 30000.0, 1.0);
    const double means_lr = 0.00016 * std::pow(0.01, progress);
    if (moments) moments->cursor = 0;
    const auto update_scalar = [&](double x, double gradient, double lr) {
        double m = 0.1 * gradient;
        double v = 0.001 * gradient * gradient;
        if (moments) {
            const size_t k = moments->cursor++;
            if (k == moments->m.size()) {
                moments->m.push_back(0.0);
                moments->v.push_back(0.0);
            }
            m += 0.9 * moments->m[k];
            v += 0.999 * moments->v[k];
            moments->m[k] = m;
            moments->v[k] = v;
        }
        return x - lr * (m * inverse_bias_m) /
                (std::sqrt(v * inverse_bias_v) + 1e-15);
    };
    for (size_t i = 0; i < updated.size(); i++) {
        for (int c = 0; c < 3; c++) {
            updated.means[i][c] = static_cast<float>(update_scalar(
                    scene.state.means[i][c], gradients.means[i][c],
                    means_lr));
        }
        glm::dvec4 unconstrained(
                std::log(scene.state.scales_opacity[i].x),
                std::log(scene.state.scales_opacity[i].y),
                std::log(scene.state.scales_opacity[i].z),
                std::log(scene.state.scales_opacity[i].w /
                         (1.0 - scene.state.scales_opacity[i].w)));
        glm::dvec4 unconstrained_gradient(
                gradients.scales_opacity[i].x *
                        scene.state.scales_opacity[i].x,
                gradients.scales_opacity[i].y *
                        scene.state.scales_opacity[i].y,
                gradients.scales_opacity[i].z *
                        scene.state.scales_opacity[i].z,
                gradients.scales_opacity[i].w *
                        scene.state.scales_opacity[i].w *
                        (1.0 - scene.state.scales_opacity[i].w));
        for (int c = 0; c < 3; c++) {
            unconstrained[c] = update_scalar(
                    unconstrained[c], unconstrained_gradient[c], 0.005);
            updated.scales_opacity[i][c] =
                    static_cast<float>(std::exp(unconstrained[c]));
        }
        unconstrained.w = update_scalar(
                unconstrained.w, unconstrained_gradient.w, 0.05);
        updated.scales_opacity[i].w = static_cast<float>(
                1.0 / (1.0 + std::exp(-unconstrained.w)));
        glm::dvec4 rotation(scene.state.rotations[i]);
        for (int c = 0; c < 4; c++) {
            rotation[c] = update_scalar(
                    rotation[c], gradients.rotations[i][c], 0.001);
        }
        rotation = glm::normalize(rotation);
        updated.rotations[i] = glm::vec4(rotation);
        for (int c = 0; c < 3; c++) {
            updated.sh_dc[i][c] = static_cast<float>(update_scalar(
                    scene.state.sh_dc[i][c], gradients.sh_dc[i][c], 0.0025));
        }
        for (size_t packed = 0; packed < 12; packed++) {
            for (int c = 0; c < 4; c++) {
                const size_t index = i * 12 + packed;
                updated.sh_rest[index][c] = static_cast<float>(update_scalar(
                        scene.state.sh_rest[index][c],
                        gradients.sh_rest[index][c], 0.000125));
            }
        }
    }
    return updated;
}

OracleCase MakeCase(uint32_t sh_degree, uint32_t variant = 0) {
    OracleCase scene;
    scene.name = (sh_degree == 0 ? "sh0_two_gaussians_v"
                                 : "sh3_two_gaussians_v") +
            std::to_string(variant);
    scene.sh_degree = sh_degree;
    scene.step = sh_degree == 0 ? 0 : 3000;
    scene.camera.width = scene.width;
    scene.camera.height = scene.height;
    scene.camera.fx = 40.0f;
    scene.camera.fy = 40.0f;
    scene.camera.cx = 24.0f;
    scene.camera.cy = 16.0f;
    scene.camera.model = CameraModel::Pinhole;
    auto& state = scene.state;
    state.means.emplace_back(0.05f, -0.03f, 3.0f, 0.0f);
    state.scales_opacity.emplace_back(0.25f, 0.18f, 0.21f, 0.7f);
    state.rotations.emplace_back(
            glm::normalize(glm::vec4(0.93f, 0.11f, -0.31f, 0.17f)));
    state.sh_dc.emplace_back(0.4f, -0.2f, 0.3f, 0.0f);
    state.means.emplace_back(-0.12f, 0.08f, 4.0f, 0.0f);
    state.scales_opacity.emplace_back(0.3f, 0.22f, 0.15f, 0.55f);
    state.rotations.emplace_back(
            glm::normalize(glm::vec4(0.85f, -0.23f, 0.19f, -0.41f)));
    state.sh_dc.emplace_back(-0.1f, 0.25f, 0.1f, 0.0f);
    state.sh_rest.assign(24, glm::vec4(0.0f));
    if (sh_degree == 3) {
        for (size_t g = 0; g < 2; g++) {
            for (size_t s = 0; s < 45; s++) {
                const float sign = (s % 2 == 0) ? 1.0f : -1.0f;
                state.sh_rest[g * 12 + s / 4][static_cast<int>(s % 4)] =
                        sign * (0.02f + 0.002f * static_cast<float>(s) +
                                0.05f * static_cast<float>(g));
            }
        }
    }
    // Keep every matrix member geometrically distinct and away from the
    // repeated projection positions that can put a finite-difference sample
    // exactly on a raster coverage boundary.
    const float signed_offset =
            (static_cast<float>(variant) - 4.5f) * 0.0067f;
    scene.camera.fx += 0.35f * static_cast<float>(variant);
    scene.camera.fy += 0.20f * static_cast<float>(variant % 4);
    scene.camera.cx += 0.08f * static_cast<float>(variant % 3);
    scene.camera.cy -= 0.06f * static_cast<float>(variant % 4);
    state.means[0].x += signed_offset;
    state.means[0].y -= 0.5f * signed_offset;
    state.means[1].x -= 0.75f * signed_offset;
    state.means[1].y += signed_offset;
    state.scales_opacity[0].x += 0.003f * static_cast<float>(variant);
    state.scales_opacity[1].y += 0.002f * static_cast<float>(variant % 5);
    state.scales_opacity[0].w -= 0.01f * static_cast<float>(variant % 4);
    state.scales_opacity[1].w += 0.008f * static_cast<float>(variant % 3);
    return scene;
}

struct GroupError {
    double max_error = 0.0;
    double worst_reference = 0.0;
    std::string worst_scalar;
    int violations = 0;
    int scalar_count = 0;
    int reference_nonzero_count = 0;
};

int RunCase(const OracleCase& scene, std::ostream& report,
            bool selected_schedule_only = false,
            bool force_split_sh_rest_storage = false) {
    const auto oracle = CpuForward(scene);
    const auto scalars = EnumerateScalars(scene);
    // The two-step update is sensitive to gradient magnitudes, so its
    // references refine the finite-difference step where it straddles a
    // coverage boundary; the one-step oracle keeps the plain step.
    FdRefinement fd_refinement;
    const auto cpu_gradients = CpuFiniteDifferenceGradients(
            scene, scalars,
            scene.adam_steps == 2 ? &fd_refinement : nullptr);
    GaussianStateCpu cpu_updated;
    if (scene.adam_steps == 2) {
        // Second step: finite-difference gradients at the first step's state,
        // then Adam with the first step's moments.
        AdamMoments moments;
        OracleCase second = scene;
        second.state = CpuAdamUpdate(scene, cpu_gradients, &moments);
        second.step = scene.step + 1;
        const auto second_gradients = CpuFiniteDifferenceGradients(
                second, EnumerateScalars(second), &fd_refinement);
        cpu_updated = CpuAdamUpdate(second, second_gradients, &moments);
    } else {
        cpu_updated = CpuAdamUpdate(scene, cpu_gradients);
    }
    int failures = 0;

    const auto evaluate = [&](int mode) -> int {
        const bool atomic_free = mode == 2;
        GaussianForwardProgram program(
                {.width = scene.width,
                 .height = scene.height,
                 .gaussian_capacity = scene.state.size(),
                 .tile_size = 16,
                 .tile_bin_capacity = 32,
                 .enable_backward = true,
                 .sh_degree = scene.sh_degree,
                 .force_split_sh_rest_storage =
                         force_split_sh_rest_storage,
                 .atomic_free_exact_backward = atomic_free});
        program.uploadState(scene.state);
        FloatImage target;
        target.width = scene.width;
        target.height = scene.height;
        target.pixels.assign(
                static_cast<size_t>(scene.width) * scene.height * 4, 0.0f);
        program.uploadTarget(target);
        program.setCamera(scene.camera);
        if (scene.step > 0) program.debugSetStep(scene.step);
        program.execute();
        const auto image = program.downloadColor();
        const double gpu_loss = program.downloadMeanL1();
        const auto gradients = program.downloadGradients();
        const std::string backward = mode == 0
                ? program.schedule().schedule_candidate_id
                : atomic_free ? "exact_pair_gather" : "exact_scatter";

        double forward_max = 0.0;
        bool finite_outputs = std::isfinite(gpu_loss) &&
                std::isfinite(oracle.mean_l1);
        for (size_t p = 0; p < oracle.image.size(); p++) {
            for (int c = 0; c < 4; c++) {
                finite_outputs = finite_outputs &&
                        std::isfinite(image.pixels[p * 4 + c]) &&
                        std::isfinite(oracle.image[p][c]);
                forward_max = std::max(
                        forward_max,
                        std::abs(double(image.pixels[p * 4 + c]) -
                                 oracle.image[p][c]));
            }
        }
        const double loss_error = std::abs(gpu_loss - oracle.mean_l1);
        int local_failures = 0;
        if (!finite_outputs || !std::isfinite(forward_max) ||
            forward_max > 1e-5) local_failures++;
        if (!std::isfinite(loss_error) || loss_error > 1e-6) {
            local_failures++;
        }

        std::vector<std::pair<std::string, GroupError>> groups;
        for (const auto& scalar : scalars) {
            const double reference = GpuGradient(cpu_gradients, scalar);
            const double measured = GpuGradient(gradients, scalar);
            const double error = std::abs(measured - reference);
            const double tolerance = 1e-6 + 1e-3 * std::abs(reference);
            const bool finite_scalar = std::isfinite(reference) &&
                    std::isfinite(measured) && std::isfinite(error) &&
                    std::isfinite(tolerance);
            auto it = std::find_if(groups.begin(), groups.end(),
                                   [&](const auto& g) {
                                       return g.first == scalar.group;
                                   });
            if (it == groups.end()) {
                groups.push_back({scalar.group, {}});
                it = groups.end() - 1;
            }
            it->second.scalar_count++;
            if (error > it->second.max_error) {
                it->second.max_error = error;
                it->second.worst_reference = reference;
                it->second.worst_scalar =
                        scalar.group + "[" +
                        std::to_string(scalar.gaussian) + "][" +
                        std::to_string(scalar.component) + "]";
            }
            if (!finite_scalar || error > tolerance) {
                it->second.violations++;
                local_failures++;
            }
        }

        auto optimizer_config = GaussianForwardConfig{
                .width = scene.width,
                .height = scene.height,
                .gaussian_capacity = scene.state.size(),
                .tile_size = 16,
                .tile_bin_capacity = 32,
                .enable_backward = true,
                .enable_optimizer = true,
                .sh_degree = scene.sh_degree,
                .force_split_sh_rest_storage =
                        force_split_sh_rest_storage,
                .atomic_free_exact_backward = atomic_free};
        GaussianForwardProgram optimizer(optimizer_config);
        optimizer.uploadState(scene.state);
        optimizer.uploadTarget(target);
        optimizer.setCamera(scene.camera);
        if (scene.step > 0) optimizer.debugSetStep(scene.step);
        optimizer.execute();
        if (scene.adam_steps == 2) optimizer.execute();
        auto gpu_updated = optimizer.downloadState();
        auto cpu_updated_copy = cpu_updated;
        auto initial_copy = scene.state;
        std::vector<std::pair<std::string, GroupError>> adam_groups;
        for (const auto& scalar : scalars) {
            const double initial = *scalar.ref(initial_copy);
            const double reference = *scalar.ref(cpu_updated_copy) - initial;
            const double measured = *scalar.ref(gpu_updated) - initial;
            const double error = std::abs(measured - reference);
            const double tolerance = 1e-5 + 1e-2 * std::abs(reference);
            auto it = std::find_if(adam_groups.begin(), adam_groups.end(),
                                   [&](const auto& g) {
                                       return g.first == scalar.group;
                                   });
            if (it == adam_groups.end()) {
                adam_groups.push_back({scalar.group, {}});
                it = adam_groups.end() - 1;
            }
            auto& group = it->second;
            group.scalar_count++;
            group.reference_nonzero_count += reference != 0.0 ? 1 : 0;
            if (error > group.max_error) {
                group.max_error = error;
                group.worst_reference = reference;
                group.worst_scalar = scalar.group + "[" +
                        std::to_string(scalar.gaussian) + "][" +
                        std::to_string(scalar.component) + "]";
            }
            if (!std::isfinite(reference) || !std::isfinite(measured) ||
                !std::isfinite(error) || error > tolerance) {
                group.violations++;
                local_failures++;
            }
        }
        for (auto& [name, group] : adam_groups) {
            if (group.reference_nonzero_count == 0) {
                group.violations++;
                local_failures++;
            }
        }

        report << "    {\n      \"backward\": \"" << backward << "\",\n"
               << "      \"forward_rgba_max_abs_error\": " << forward_max
               << ",\n      \"loss_abs_error\": " << loss_error
               << ",\n      \"cpu_loss\": " << oracle.mean_l1
               << ",\n      \"gpu_loss\": " << gpu_loss
               << ",\n      \"finite_outputs\": "
               << (finite_outputs ? "true" : "false")
               << ",\n      \"groups\": [\n";
        for (size_t g = 0; g < groups.size(); g++) {
            const auto& [name, err] = groups[g];
            report << "        {\"group\": \"" << name
                   << "\", \"scalars\": " << err.scalar_count
                   << ", \"max_abs_error\": " << err.max_error
                   << ", \"worst_scalar\": \"" << err.worst_scalar
                   << "\", \"worst_reference\": " << err.worst_reference
                   << ", \"violations\": " << err.violations << "}"
                   << (g + 1 < groups.size() ? ",\n" : "\n");
        }
        report << "      ],\n      \"adam_update_groups\": [\n";
        for (size_t g = 0; g < adam_groups.size(); g++) {
            const auto& [name, err] = adam_groups[g];
            report << "        {\"group\": \"" << name
                   << "\", \"scalars\": " << err.scalar_count
                   << ", \"reference_nonzero_count\": "
                   << err.reference_nonzero_count
                   << ", \"max_abs_error\": " << err.max_error
                   << ", \"worst_scalar\": \"" << err.worst_scalar
                   << "\", \"worst_reference_delta\": "
                   << err.worst_reference
                   << ", \"violations\": " << err.violations << "}"
                   << (g + 1 < adam_groups.size() ? ",\n" : "\n");
        }
        report << "      ]\n    }";
        std::cerr << scene.name << "/" << backward << ": forward "
                  << forward_max << ", loss " << loss_error
                  << ", failures " << local_failures << "\n";
        return local_failures;
    };

    report << "  {\n  \"case\": \"" << scene.name << "\",\n"
           << "  \"gaussians\": " << scene.state.size() << ",\n"
           << "  \"image\": [" << scene.width << ", " << scene.height
           << "],\n  \"sh_degree\": " << scene.sh_degree
           << ",\n  \"step\": " << scene.step
           << ",\n  \"adam_steps\": " << scene.adam_steps;
    if (scene.adam_steps == 2) {
        report << ",\n  \"fd_refined_scalars\": " << fd_refinement.refined
               << ",\n  \"fd_unresolved_scalars\": "
               << fd_refinement.unresolved;
    }
    report << ",\n  \"backends\": [\n";
    if (selected_schedule_only) {
        failures += evaluate(0);
    } else {
        failures += evaluate(1);
        report << ",\n";
        failures += evaluate(2);
    }
    report << "\n  ]\n  }";
    return failures;
}

std::pair<std::string, int> RunSelectedOracleMatrix(
        bool force_split_sh_rest_storage = false, uint32_t adam_steps = 1) {
    std::ostringstream report;
    report << "{\n\"schema\": \"cross_3dgs.cpu_double_oracle.v2\",\n"
           << "\"fd_epsilon_scale\": " << FdEpsilonScale() << ",\n"
           << "\"tolerances\": {\"forward_rgba_abs\": 1e-5, "
           << "\"loss_abs\": 1e-6, "
           << "\"gradient\": \"1e-6 + 1e-3 * |oracle|\", "
           << "\"adam_delta\": \"1e-5 + 1e-2 * |oracle_delta|\"},\n"
           << "\"oracle\": \"CPU double forward + central finite "
              "differences + independent "
           << (adam_steps == 2 ? "two-step Adam update with carried moments"
                               : "zero-moment Adam update")
           << "\",\n"
           << "\"storage_layout\": \""
           << (force_split_sh_rest_storage ? "sh_rest_6_plus_6"
                                           : "device_default")
           << "\",\n"
           << "\"case_count\": 20,\n\"cases\": [\n";
    int failures = 0;
    size_t case_index = 0;
    for (const uint32_t sh_degree : {0u, 3u}) {
        for (uint32_t variant = 0; variant < 10; variant++) {
            if (case_index++ != 0) report << ",\n";
            OracleCase scene = MakeCase(sh_degree, variant);
            if (adam_steps == 2) {
                scene.adam_steps = 2;
                scene.name += "_adam2";
            }
            failures += RunCase(scene, report, true,
                                force_split_sh_rest_storage);
        }
    }
    report << "\n],\n\"failures\": " << failures
           << ",\n\"pass\": " << (failures == 0 ? "true" : "false")
           << "\n}\n";
    return {report.str(), failures};
}

}  // namespace

#ifdef CROSS_3DGS_CPU_ORACLE_LIBRARY
namespace cross3dgs_probe {

bool RunSelectedScheduleCpuOracle(
        const std::filesystem::path& output_dir,
        bool force_split_sh_rest_storage) {
    std::filesystem::create_directories(output_dir);
    const auto [report, failures] =
            RunSelectedOracleMatrix(force_split_sh_rest_storage);
    std::ofstream output(output_dir / "cpu_oracle.json");
    output << report;
    return failures == 0 && static_cast<bool>(output);
}

}  // namespace cross3dgs_probe
#else
int main(int argc, char** argv) {
    std::filesystem::path output_dir;
    bool selected_matrix = false;
    bool force_split_sh_rest_storage = false;
    uint32_t adam_steps = 1;
    for (int i = 1; i < argc; i++) {
        const std::string argument = argv[i];
        if (argument == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (argument == "--selected-matrix") {
            selected_matrix = true;
        } else if (argument == "--force-split-sh-rest-storage") {
            force_split_sh_rest_storage = true;
        } else if (argument == "--two-step-adam") {
            adam_steps = 2;
        }
    }
    if (selected_matrix) {
        const auto [selected_report, selected_failures] =
                RunSelectedOracleMatrix(force_split_sh_rest_storage,
                                        adam_steps);
        if (!output_dir.empty()) {
            std::filesystem::create_directories(output_dir);
            std::ofstream output(output_dir /
                                 (adam_steps == 2
                                          ? "cpu_oracle_two_step_adam.json"
                                          : "cpu_oracle.json"));
            output << selected_report;
        }
        return selected_failures == 0 ? 0 : 1;
    }
    std::ostringstream report;
    report << "{\n\"schema\": \"cross_3dgs.cpu_double_oracle.v1\",\n"
           << "\"tolerances\": {\"forward_rgba_abs\": 1e-5, "
           << "\"loss_abs\": 1e-6, "
           << "\"gradient\": \"1e-6 + 1e-3 * |oracle|\"},\n"
           << "\"oracle\": \"CPU double forward replica + double central "
           << "finite differences (device independent)\",\n"
           << "\"cases\": [\n";
    int failures = RunCase(MakeCase(0), report);
    report << ",\n";
    failures += RunCase(MakeCase(3), report);
    report << "\n],\n\"failures\": " << failures << "\n}\n";
    if (!output_dir.empty()) {
        std::filesystem::create_directories(output_dir);
        std::ofstream output(output_dir / "cpu_oracle.json");
        output << report.str();
    }
    if (failures != 0) {
        std::cerr << "CPU oracle mismatches: " << failures << "\n";
        return 1;
    }
    std::cout << "CPU double oracle validated\n";
    return 0;
}
#endif
