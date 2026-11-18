#ifndef CROSS3DGS_GAUSSIAN_STATE_H
#define CROSS3DGS_GAUSSIAN_STATE_H

#include <cstddef>
#include <filesystem>
#include <vector>

#include <glm/glm.hpp>

#include "cross3dgs/dataset.h"

namespace cross3dgs {

struct GaussianStateCpu {
    std::vector<glm::vec4> means;
    std::vector<glm::vec4> scales_opacity;
    // Quaternion component order is (w, x, y, z), matching VkSplat.
    std::vector<glm::vec4> rotations;
    std::vector<glm::vec4> sh_dc;
    // SH coefficients 1..15, RGB-interleaved and packed into 12 vec4 values
    // per Gaussian. The final three scalar lanes are padding.
    std::vector<glm::vec4> sh_rest;

    size_t size() const { return means.size(); }
};

GaussianStateCpu InitializeFromSparsePoints(const Scene& scene,
                                            float scale_multiplier = 0.1f,
                                            float opacity = 0.5f,
                                            uint32_t seed = 42);
// Writes the standard 3DGS PLY: f_rest_* holds each color channel's 15 SH
// coefficients in turn (channel * 15 + coefficient), not the interleaved
// order of GaussianStateCpu::sh_rest.
void SaveGaussianPly(const std::filesystem::path& path,
                     const GaussianStateCpu& state);
// The rotation taking unit vector `world_down` to +Y, the down direction of
// the COLMAP/3DGS convention, by the smallest angle.
glm::mat3 RotationToDownPositiveY(const glm::vec3& world_down);
// The same Gaussians in a world rotated by `rotation`: means, orientations,
// and view-dependent color all turn with it.
GaussianStateCpu RotateGaussianState(const GaussianStateCpu& state,
                                     const glm::mat3& rotation);
void SaveGaussianCheckpoint(const std::filesystem::path& path,
                            const GaussianStateCpu& state);
GaussianStateCpu LoadGaussianCheckpoint(const std::filesystem::path& path);

}  // namespace cross3dgs

#endif
