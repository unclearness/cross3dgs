// Platform-neutral pieces of the capture-to-view pipeline, ported from the
// Android :e2e module so both apps derive SfM priors and training scenes the
// same way:
//   PosesFile        <- org.unclearness.cross3dgs.capture.PosesFile (Kotlin)
//   Colmap* helpers  <- org.unclearness.cross3dgs.ColmapBinary (Kotlin)
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace cross3dgs_ios {

// Parser for arkit_poses.txt (the same layout as Android's arcore_poses.txt).
// Body rows are TUM trajectory rows `timestamp_ns tx ty tz qx qy qz qw`,
// world_from_camera in meters, in the AR camera convention (+X right, +Y up,
// -Z forward), which ARKit and ARCore share. Header comments carry
// `video_start_ns`, `intrinsics fx fy cx cy width height` and
// `gravity_world`.
class PosesFile {
public:
    struct Prior {
        std::array<double, 3> position{};
        // Gravity ("down") in the COLMAP camera frame (+X right, +Y down,
        // +Z forward).
        std::array<double, 3> gravity_camera{};
        // Position std-dev [m]; inflated when the tracker skipped frames
        // around this timestamp.
        double sigma = 0.0;
    };

    static std::optional<PosesFile> parse(const std::filesystem::path& path);

    int64_t videoStartNs() const { return m_video_start_ns; }
    bool hasIntrinsics() const { return m_has_intrinsics; }
    // fx fy cx cy at intrinsicsSize().
    const std::array<double, 4>& intrinsics() const { return m_intrinsics; }
    const std::array<int, 2>& intrinsicsSize() const { return m_intrinsics_size; }
    size_t sampleCount() const { return m_times.size(); }

    // Interpolated prior at absolute capture time t_ns, or nullopt when no
    // pose sample is close enough.
    std::optional<Prior> priorAt(int64_t t_ns) const;

private:
    Prior makePrior(const std::array<double, 3>& position,
                    const std::array<double, 4>& q_world_from_camera,
                    double sigma) const;

    int64_t m_video_start_ns = 0;
    bool m_has_intrinsics = false;
    std::array<double, 4> m_intrinsics{};
    std::array<int, 2> m_intrinsics_size{};
    std::array<double, 3> m_gravity_world{0.0, -1.0, 0.0};
    std::vector<int64_t> m_times;
    std::vector<std::array<double, 3>> m_positions;
    std::vector<std::array<double, 4>> m_quaternions;  // x y z w
};

// Registered image names in a COLMAP images.bin, in file order.
std::vector<std::string> ColmapRegisteredImageNames(
        const std::filesystem::path& images_bin);
// Mean of COLMAP's stored per-point reprojection errors; nullopt if empty.
std::optional<double> ColmapMeanReprojectionError(
        const std::filesystem::path& points_bin);

// Lowercase hex SHA-256 of a file's bytes.
std::string Sha256File(const std::filesystem::path& path);

}  // namespace cross3dgs_ios
