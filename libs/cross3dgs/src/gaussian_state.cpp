// Initialization intentionally matches VkSplat's sparse-point KNN policy.
// The reference implementation is Apache-2.0; see THIRD_PARTY_NOTICES.md.
#include "cross3dgs/gaussian_state.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <memory>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>

#include <glm/gtc/quaternion.hpp>

namespace cross3dgs {

namespace {

struct KdNode {
    glm::vec3 point{};
    size_t index = 0;
    uint32_t axis = 0;
    std::unique_ptr<KdNode> left;
    std::unique_ptr<KdNode> right;
};

std::unique_ptr<KdNode> BuildTree(
        std::vector<std::pair<glm::vec3, size_t>>::iterator begin,
        std::vector<std::pair<glm::vec3, size_t>>::iterator end,
        uint32_t depth) {
    if (begin == end) {
        return nullptr;
    }
    const uint32_t axis = depth % 3;
    auto middle = begin + (end - begin) / 2;
    std::nth_element(begin, middle, end,
                     [axis](const auto& left, const auto& right) {
                         return left.first[axis] < right.first[axis];
                     });
    auto node = std::make_unique<KdNode>();
    node->point = middle->first;
    node->index = middle->second;
    node->axis = axis;
    node->left = BuildTree(begin, middle, depth + 1);
    node->right = BuildTree(middle + 1, end, depth + 1);
    return node;
}

struct Neighbor {
    float distance = 0.0f;
    size_t index = 0;
    bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

void Search(const KdNode* node, const glm::vec3& query, size_t count,
            std::priority_queue<Neighbor>& neighbors) {
    if (node == nullptr) {
        return;
    }
    const float distance = glm::distance(node->point, query);
    if (neighbors.size() < count) {
        neighbors.push({distance, node->index});
    } else if (distance < neighbors.top().distance) {
        neighbors.pop();
        neighbors.push({distance, node->index});
    }
    const float difference = query[node->axis] - node->point[node->axis];
    const KdNode* first = difference < 0.0f ? node->left.get()
                                           : node->right.get();
    const KdNode* second = difference < 0.0f ? node->right.get()
                                            : node->left.get();
    Search(first, query, count, neighbors);
    if (neighbors.size() < count ||
        std::abs(difference) < neighbors.top().distance) {
        Search(second, query, count, neighbors);
    }
}

std::vector<float> InitialScales(const Scene& scene, float multiplier) {
    std::vector<std::pair<glm::vec3, size_t>> indexed;
    indexed.reserve(scene.points.size());
    for (size_t i = 0; i < scene.points.size(); i++) {
        indexed.emplace_back(scene.points[i].position, i);
    }
    auto root = BuildTree(indexed.begin(), indexed.end(), 0);
    std::vector<float> result(scene.points.size());
    for (size_t i = 0; i < scene.points.size(); i++) {
        std::priority_queue<Neighbor> neighbors;
        Search(root.get(), scene.points[i].position, 4, neighbors);
        float squared_sum = 0.0f;
        while (!neighbors.empty()) {
            squared_sum += neighbors.top().distance * neighbors.top().distance;
            neighbors.pop();
        }
        result[i] = multiplier * std::sqrt(squared_sum / 3.0f);
    }
    return result;
}

}  // namespace

GaussianStateCpu InitializeFromSparsePoints(const Scene& scene,
                                            float scale_multiplier,
                                            float opacity, uint32_t seed) {
    if (scene.points.empty()) {
        throw std::invalid_argument("cannot initialize Gaussians without points");
    }
    constexpr float kShC0 = 0.28209479177387814f;
    const auto scales = InitialScales(scene, scale_multiplier);
    // VkSplat uses the implementation's default_random_engine, and matching it
    // is part of the Windows/PC benchmark contract.  std::default_random_engine
    // is implementation-defined, so that contract only held while both builds
    // were MSVC: it aliases mt19937 there but minstd_rand0 on libstdc++, which
    // gave Linux an entirely different set of initial rotations from the same
    // seed.  Name mt19937 explicitly -- identical to what MSVC was already
    // choosing (so the Windows/VkSplat numbers are unchanged), and now the same
    // sequence everywhere.
    //
    // This pins the engine, not the whole pipeline: std::normal_distribution's
    // algorithm is also implementation-defined, so exact cross-platform parity
    // would additionally require hand-rolling the normal transform.
    std::mt19937 random(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    GaussianStateCpu state;
    state.means.reserve(scene.points.size());
    state.scales_opacity.reserve(scene.points.size());
    state.rotations.reserve(scene.points.size());
    state.sh_dc.reserve(scene.points.size());
    state.sh_rest.assign(scene.points.size() * 12, glm::vec4(0.0f));
    for (size_t i = 0; i < scene.points.size(); i++) {
        const auto& point = scene.points[i];
        state.means.emplace_back(point.position, 0.0f);
        state.scales_opacity.emplace_back(glm::vec3(scales[i]), opacity);
        glm::vec4 quaternion(normal(random), normal(random), normal(random),
                             normal(random));
        quaternion = glm::normalize(quaternion);
        state.rotations.push_back(quaternion);
        const glm::vec3 color = glm::vec3(point.color[0], point.color[1],
                                          point.color[2]) /
                                255.0f;
        state.sh_dc.emplace_back((color - 0.5f) / kShC0, 0.0f);
    }
    return state;
}

namespace {

constexpr int kShRestCoefficients = 15;
using ShRotation =
        std::array<std::array<double, kShRestCoefficients>, kShRestCoefficients>;

// sh_rest stores coefficient-major RGB triples, four scalars per vec4.
float& ShRestScalar(GaussianStateCpu& state, size_t gaussian, int coefficient,
                    int channel) {
    const int scalar = coefficient * 3 + channel;
    return state.sh_rest[gaussian * 12 + scalar / 4][scalar % 4];
}

float ShRestScalar(const GaussianStateCpu& state, size_t gaussian,
                   int coefficient, int channel) {
    return ShRestScalar(const_cast<GaussianStateCpu&>(state), gaussian,
                        coefficient, channel);
}

// SH degrees 1 to 3 in the renderer's order and signs (shaders_forward.cpp).
std::array<double, kShRestCoefficients> ShBasis(const glm::dvec3& d) {
    const double x = d.x, y = d.y, z = d.z;
    const double xx = x * x, yy = y * y, zz = z * z;
    return {-0.4886025119029199 * y,
            0.4886025119029199 * z,
            -0.4886025119029199 * x,
            1.0925484305920792 * x * y,
            -1.0925484305920792 * y * z,
            0.31539156525252005 * (2.0 * zz - xx - yy),
            -1.0925484305920792 * x * z,
            0.5462742152960396 * (xx - yy),
            -0.5900435899266435 * y * (3.0 * xx - yy),
            2.890611442640554 * x * y * z,
            -0.4570457994644658 * y * (4.0 * zz - xx - yy),
            0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy),
            -0.4570457994644658 * x * (4.0 * zz - xx - yy),
            1.445305721320277 * z * (xx - yy),
            -0.5900435899266435 * x * (xx - 3.0 * yy)};
}

// The matrix M with (M c) . Y(d) = c . Y(R^T d) for every direction d, so
// colors seen along d after the rotation are those seen along R^T d before
// it. Least squares over directions spread across the sphere; the basis
// functions are polynomials, so the fit is exact.
ShRotation MakeShRotation(const glm::dmat3& rotation) {
    constexpr int kDirections = 256;
    constexpr int n = kShRestCoefficients;
    std::array<std::array<double, 2 * n>, n> system{};
    const double golden_angle = 3.14159265358979323846 * (3.0 - std::sqrt(5.0));
    for (int i = 0; i < kDirections; i++) {
        const double y = 1.0 - 2.0 * (i + 0.5) / kDirections;
        const double radius = std::sqrt(1.0 - y * y);
        const double angle = golden_angle * i;
        const glm::dvec3 direction(radius * std::cos(angle), y,
                                   radius * std::sin(angle));
        const auto a = ShBasis(direction);
        const auto b = ShBasis(glm::transpose(rotation) * direction);
        for (int row = 0; row < n; row++) {
            for (int column = 0; column < n; column++) {
                system[row][column] += a[row] * a[column];
                system[row][n + column] += a[row] * b[column];
            }
        }
    }
    // Gauss-Jordan on the normal equations [A^T A | A^T B].
    for (int pivot = 0; pivot < n; pivot++) {
        int best = pivot;
        for (int row = pivot + 1; row < n; row++) {
            if (std::abs(system[row][pivot]) > std::abs(system[best][pivot])) {
                best = row;
            }
        }
        std::swap(system[pivot], system[best]);
        const double scale = 1.0 / system[pivot][pivot];
        for (double& value : system[pivot]) value *= scale;
        for (int row = 0; row < n; row++) {
            if (row == pivot) continue;
            const double factor = system[row][pivot];
            for (int column = 0; column < 2 * n; column++) {
                system[row][column] -= factor * system[pivot][column];
            }
        }
    }
    ShRotation result{};
    for (int row = 0; row < n; row++) {
        for (int column = 0; column < n; column++) {
            result[row][column] = system[row][n + column];
        }
    }
    return result;
}

}  // namespace

glm::mat3 RotationToDownPositiveY(const glm::vec3& world_down) {
    const glm::vec3 down = glm::normalize(world_down);
    const glm::vec3 target(0.0f, 1.0f, 0.0f);
    const float cosine = glm::dot(down, target);
    if (cosine < -1.0f + 1e-6f) {
        // Exactly upside down: any half turn about a horizontal axis does.
        return glm::mat3(glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f),
                         glm::vec3(0.0f, 0.0f, -1.0f));
    }
    const glm::vec3 v = glm::cross(down, target);
    const glm::mat3 skew(glm::vec3(0.0f, v.z, -v.y), glm::vec3(-v.z, 0.0f, v.x),
                         glm::vec3(v.y, -v.x, 0.0f));
    return glm::mat3(1.0f) + skew + skew * skew * (1.0f / (1.0f + cosine));
}

GaussianStateCpu RotateGaussianState(const GaussianStateCpu& state,
                                     const glm::mat3& rotation) {
    GaussianStateCpu rotated = state;
    const glm::quat turn = glm::normalize(glm::quat_cast(rotation));
    const ShRotation sh = MakeShRotation(glm::dmat3(rotation));
    for (size_t index = 0; index < state.size(); index++) {
        const glm::vec3 mean = rotation * glm::vec3(state.means[index]);
        rotated.means[index] = glm::vec4(mean, state.means[index].w);
        // Stored as (w, x, y, z) in the vec4's x, y, z, w lanes.
        const glm::vec4& q = state.rotations[index];
        const glm::quat turned = turn * glm::quat(q.x, q.y, q.z, q.w);
        rotated.rotations[index] =
                glm::vec4(turned.w, turned.x, turned.y, turned.z);
        for (int channel = 0; channel < 3; channel++) {
            for (int row = 0; row < kShRestCoefficients; row++) {
                double value = 0.0;
                for (int column = 0; column < kShRestCoefficients; column++) {
                    value += sh[row][column] *
                             ShRestScalar(state, index, column, channel);
                }
                ShRestScalar(rotated, index, row, channel) =
                        static_cast<float>(value);
            }
        }
    }
    return rotated;
}

void SaveGaussianPly(const std::filesystem::path& path,
                     const GaussianStateCpu& state) {
    const size_t count = state.size();
    if (state.scales_opacity.size() != count ||
        state.rotations.size() != count || state.sh_dc.size() != count ||
        state.sh_rest.size() != count * 12) {
        throw std::invalid_argument("Gaussian state arrays have different sizes");
    }
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to create Gaussian PLY: " +
                                 path.string());
    }
    output << "ply\n"
           << "format binary_little_endian 1.0\n"
           << "element vertex " << count << "\n"
           << "property float x\n"
           << "property float y\n"
           << "property float z\n"
           << "property float f_dc_0\n"
           << "property float f_dc_1\n"
           << "property float f_dc_2\n";
    for (int coefficient = 0; coefficient < 45; coefficient++) {
        output << "property float f_rest_" << coefficient << "\n";
    }
    output << "property float opacity\n"
           << "property float scale_0\n"
           << "property float scale_1\n"
           << "property float scale_2\n"
           << "property float rot_0\n"
           << "property float rot_1\n"
           << "property float rot_2\n"
           << "property float rot_3\n"
           << "end_header\n";
    constexpr float kMinimum = 1.0e-6f;
    constexpr float kMaximumOpacity = 1.0f - kMinimum;
    const auto write_float = [&](float value) {
        output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    };
    for (size_t index = 0; index < count; index++) {
        for (int component = 0; component < 3; component++) {
            write_float(state.means[index][component]);
        }
        for (int component = 0; component < 3; component++) {
            write_float(state.sh_dc[index][component]);
        }
        for (int channel = 0; channel < 3; channel++) {
            for (int coefficient = 0; coefficient < kShRestCoefficients;
                 coefficient++) {
                write_float(ShRestScalar(state, index, coefficient, channel));
            }
        }
        const float opacity = std::clamp(
                state.scales_opacity[index].w, kMinimum, kMaximumOpacity);
        write_float(std::log(opacity / (1.0f - opacity)));
        for (int component = 0; component < 3; component++) {
            write_float(std::log(std::max(
                    state.scales_opacity[index][component], kMinimum)));
        }
        for (int component = 0; component < 4; component++) {
            write_float(state.rotations[index][component]);
        }
    }
    if (!output) {
        throw std::runtime_error("failed to write Gaussian PLY: " +
                                 path.string());
    }
}

void SaveGaussianCheckpoint(const std::filesystem::path& path,
                            const GaussianStateCpu& state) {
    const size_t count=state.size();
    if (state.scales_opacity.size()!=count||state.rotations.size()!=count||
        state.sh_dc.size()!=count||state.sh_rest.size()!=count*12)
        throw std::invalid_argument("Gaussian checkpoint state shape mismatch");
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path,std::ios::binary);
    if(!output) throw std::runtime_error("failed to create Gaussian checkpoint: "+path.string());
    const std::array<char,8> magic{'D','3','G','S','C','H','K','1'};
    const uint64_t stored_count=static_cast<uint64_t>(count);
    output.write(magic.data(),magic.size());
    output.write(reinterpret_cast<const char*>(&stored_count),sizeof(stored_count));
    const auto write=[&](const std::vector<glm::vec4>& values){
        output.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size()*sizeof(glm::vec4)));
    };
    write(state.means);write(state.scales_opacity);write(state.rotations);
    write(state.sh_dc);write(state.sh_rest);
    if(!output) throw std::runtime_error("failed to write Gaussian checkpoint: "+path.string());
}

GaussianStateCpu LoadGaussianCheckpoint(const std::filesystem::path& path) {
    std::ifstream input(path,std::ios::binary);
    if(!input) throw std::runtime_error("failed to open Gaussian checkpoint: "+path.string());
    std::array<char,8> magic{};uint64_t count=0;
    input.read(magic.data(),magic.size());
    input.read(reinterpret_cast<char*>(&count),sizeof(count));
    const std::array<char,8> expected{'D','3','G','S','C','H','K','1'};
    if(magic!=expected||count==0||count>100000000)
        throw std::runtime_error("invalid Gaussian checkpoint header: "+path.string());
    GaussianStateCpu state;
    const auto read=[&](std::vector<glm::vec4>& values,size_t size){
        values.resize(size);
        input.read(reinterpret_cast<char*>(values.data()),
                   static_cast<std::streamsize>(values.size()*sizeof(glm::vec4)));
    };
    read(state.means,count);read(state.scales_opacity,count);
    read(state.rotations,count);read(state.sh_dc,count);read(state.sh_rest,count*12);
    if(!input||input.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error("invalid Gaussian checkpoint payload: "+path.string());
    return state;
}

}  // namespace cross3dgs
