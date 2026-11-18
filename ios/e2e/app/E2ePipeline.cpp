#include "E2ePipeline.h"

#include <CommonCrypto/CommonDigest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace cross3dgs_ios {
namespace {

constexpr int64_t kMaxGapNs = 150'000'000;          // 150 ms
constexpr int64_t kStableBracketNs = 66'000'000;    // ~2 tracker frames
constexpr double kSigmaStable = 0.08;
constexpr double kSigmaUnstable = 0.20;

using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>;  // x y z w

Vec3 Rotate(const Quat& q, const Vec3& v) {
    const double x = q[0], y = q[1], z = q[2], w = q[3];
    // t = 2 q_vec x v; v' = v + w t + q_vec x t
    const double tx = 2 * (y * v[2] - z * v[1]);
    const double ty = 2 * (z * v[0] - x * v[2]);
    const double tz = 2 * (x * v[1] - y * v[0]);
    return {v[0] + w * tx + (y * tz - z * ty),
            v[1] + w * ty + (z * tx - x * tz),
            v[2] + w * tz + (x * ty - y * tx)};
}

Quat Slerp(const Quat& qa, Quat qb, double a) {
    double dot = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3];
    if (dot < 0) {
        for (double& value : qb) value = -value;
        dot = -dot;
    }
    Quat out{};
    if (dot > 0.9995) {
        double norm = 0;
        for (int i = 0; i < 4; i++) {
            out[i] = qa[i] * (1 - a) + qb[i] * a;
            norm += out[i] * out[i];
        }
        norm = std::sqrt(norm);
        for (double& value : out) value /= norm;
        return out;
    }
    const double theta = std::acos(std::clamp(dot, -1.0, 1.0));
    const double s = std::sin(theta);
    const double wa = std::sin((1 - a) * theta) / s;
    const double wb = std::sin(a * theta) / s;
    for (int i = 0; i < 4; i++) out[i] = qa[i] * wa + qb[i] * wb;
    return out;
}

uint64_t ReadU64(std::istream& input) {
    unsigned char bytes[8];
    if (!input.read(reinterpret_cast<char*>(bytes), 8)) {
        throw std::runtime_error("truncated COLMAP binary");
    }
    uint64_t value = 0;
    for (int shift = 0; shift < 8; shift++) {
        value |= static_cast<uint64_t>(bytes[shift]) << (8 * shift);
    }
    return value;
}

void SkipExact(std::istream& input, uint64_t bytes) {
    input.seekg(static_cast<std::streamoff>(bytes), std::ios::cur);
    if (!input) throw std::runtime_error("truncated COLMAP binary");
}

}  // namespace

std::optional<PosesFile> PosesFile::parse(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) return std::nullopt;
    PosesFile poses;
    std::string raw;
    while (std::getline(file, raw)) {
        const auto first = raw.find_first_not_of(" \t\r");
        if (first == std::string::npos) continue;
        std::string line = raw.substr(first);
        if (line[0] == '#') {
            std::istringstream tokens(line.substr(1));
            std::string key;
            tokens >> key;
            if (key == "video_start_ns") {
                tokens >> poses.m_video_start_ns;
            } else if (key == "intrinsics") {
                if (tokens >> poses.m_intrinsics[0] >> poses.m_intrinsics[1] >>
                    poses.m_intrinsics[2] >> poses.m_intrinsics[3] >>
                    poses.m_intrinsics_size[0] >> poses.m_intrinsics_size[1]) {
                    poses.m_has_intrinsics = true;
                }
            } else if (key == "gravity_world") {
                tokens >> poses.m_gravity_world[0] >> poses.m_gravity_world[1] >>
                        poses.m_gravity_world[2];
            }
            continue;
        }
        std::istringstream tokens(line);
        int64_t time = 0;
        Vec3 position{};
        Quat quaternion{};
        if (!(tokens >> time >> position[0] >> position[1] >> position[2] >>
              quaternion[0] >> quaternion[1] >> quaternion[2] >>
              quaternion[3])) {
            continue;
        }
        poses.m_times.push_back(time);
        poses.m_positions.push_back(position);
        poses.m_quaternions.push_back(quaternion);
    }
    if (poses.m_times.empty()) return std::nullopt;
    return poses;
}

std::optional<PosesFile::Prior> PosesFile::priorAt(int64_t t_ns) const {
    // Index of the last sample at or before t_ns (-1 if none).
    const auto upper = std::upper_bound(m_times.begin(), m_times.end(), t_ns);
    const long idx = static_cast<long>(upper - m_times.begin()) - 1;
    const long count = static_cast<long>(m_times.size());
    if (idx < 0) {
        if (std::llabs(t_ns - m_times.front()) > kMaxGapNs) return std::nullopt;
        return makePrior(m_positions.front(), m_quaternions.front(),
                         kSigmaUnstable);
    }
    if (idx == count - 1) {
        if (std::llabs(t_ns - m_times[idx]) > kMaxGapNs) return std::nullopt;
        return makePrior(m_positions[idx], m_quaternions[idx], kSigmaUnstable);
    }
    const int64_t t0 = m_times[idx];
    const int64_t t1 = m_times[idx + 1];
    if (t_ns - t0 > kMaxGapNs && t1 - t_ns > kMaxGapNs) return std::nullopt;
    const double sigma =
            t1 - t0 > kStableBracketNs ? kSigmaUnstable : kSigmaStable;
    const double a = static_cast<double>(t_ns - t0) /
                     static_cast<double>(t1 - t0);
    Vec3 position{};
    for (int i = 0; i < 3; i++) {
        position[i] = m_positions[idx][i] * (1 - a) +
                      m_positions[idx + 1][i] * a;
    }
    return makePrior(position,
                     Slerp(m_quaternions[idx], m_quaternions[idx + 1], a),
                     sigma);
}

PosesFile::Prior PosesFile::makePrior(const Vec3& position,
                                      const Quat& q_world_from_camera,
                                      double sigma) const {
    // Gravity in the AR camera frame: g_cam = R_wc^T g_world.
    const Quat conjugate{-q_world_from_camera[0], -q_world_from_camera[1],
                         -q_world_from_camera[2], q_world_from_camera[3]};
    const Vec3 g_ar = Rotate(conjugate, m_gravity_world);
    // AR camera (+X right, +Y up, -Z forward) -> COLMAP camera (+X right,
    // +Y down, +Z forward): negate Y and Z.
    Prior prior;
    prior.position = position;
    prior.gravity_camera = {g_ar[0], -g_ar[1], -g_ar[2]};
    prior.sigma = sigma;
    return prior;
}

std::vector<std::string> ColmapRegisteredImageNames(
        const std::filesystem::path& images_bin) {
    std::ifstream input(images_bin, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + images_bin.string());
    const uint64_t count = ReadU64(input);
    if (count > 1'000'000) {
        throw std::runtime_error("implausible COLMAP image count");
    }
    std::vector<std::string> names;
    names.reserve(count);
    for (uint64_t index = 0; index < count; index++) {
        SkipExact(input, 4 + 7 * 8 + 4);
        std::string name;
        std::getline(input, name, '\0');
        if (!input) throw std::runtime_error("truncated COLMAP image name");
        names.push_back(name);
        const uint64_t points = ReadU64(input);
        if (points > 100'000'000) {
            throw std::runtime_error("implausible COLMAP observation count");
        }
        SkipExact(input, points * 24);
    }
    return names;
}

std::optional<double> ColmapMeanReprojectionError(
        const std::filesystem::path& points_bin) {
    std::ifstream input(points_bin, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + points_bin.string());
    const uint64_t count = ReadU64(input);
    if (count == 0) return std::nullopt;
    if (count > 100'000'000) {
        throw std::runtime_error("implausible COLMAP point count");
    }
    double error_sum = 0.0;
    for (uint64_t index = 0; index < count; index++) {
        SkipExact(input, 8 + 3 * 8 + 3);
        const uint64_t bits = ReadU64(input);
        double error = 0.0;
        static_assert(sizeof(error) == sizeof(bits));
        std::memcpy(&error, &bits, sizeof(error));
        error_sum += error;
        const uint64_t track_length = ReadU64(input);
        if (track_length > 100'000'000) {
            throw std::runtime_error("implausible COLMAP track length");
        }
        SkipExact(input, track_length * 8);
    }
    return error_sum / static_cast<double>(count);
}

std::string Sha256File(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    std::vector<char> buffer(1 << 20);
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto read = input.gcount();
        if (read > 0) {
            CC_SHA256_Update(&context, buffer.data(),
                             static_cast<CC_LONG>(read));
        }
    }
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);
    std::ostringstream hex;
    for (unsigned char byte : digest) {
        hex << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<int>(byte);
    }
    return hex.str();
}

}  // namespace cross3dgs_ios
