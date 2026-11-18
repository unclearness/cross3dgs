#ifndef CROSS3DGS_DATASET_H
#define CROSS3DGS_DATASET_H

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace cross3dgs {

enum class CameraModel {
    Pinhole,
    OpenCv,
    OpenCvFisheye,
};

struct Camera {
    uint32_t image_id = 0;
    uint32_t camera_id = 0;
    std::string image_name;
    std::filesystem::path image_path;
    uint32_t width = 0;
    uint32_t height = 0;
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    std::array<float, 4> distortion{};
    CameraModel model = CameraModel::Pinhole;
    glm::mat4 world_to_camera{1.0f};
    bool validation = false;
};

struct SparsePoint {
    uint64_t id = 0;
    glm::vec3 position{};
    std::array<uint8_t, 3> color{};
};

struct Scene {
    std::filesystem::path scene_path;
    std::string image_dir;
    std::string sparse_dir;
    std::vector<Camera> cameras;
    std::vector<SparsePoint> points;
    glm::mat4 dataset_transform{1.0f};

    size_t trainCameraCount() const;
    size_t validationCameraCount() const;
};

struct FloatImage {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 4;
    std::vector<float> pixels;
};

Scene LoadColmapScene(const std::filesystem::path& scene_path,
                      const std::string& image_dir,
                      const std::string& sparse_dir,
                      uint32_t eval_interval = 8,
                      bool normalize_world = true);

// Matches VkSplat/GSplat: maximum camera-center distance from the mean center,
// expanded by ten percent. All cameras, including validation views, contribute.
float CameraExtentSceneScale(const std::vector<Camera>& cameras);

FloatImage LoadImageRgbaF32(const std::filesystem::path& path);
void SaveImagePng(const std::filesystem::path& path, const FloatImage& image);

}  // namespace cross3dgs

#endif
