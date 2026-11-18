// The COLMAP normalization sequence is adapted from VkSplat
// (https://github.com/harry7557558/vksplat, Apache-2.0) so both backends use
// the same coordinate system for the fairness benchmark.
#include "cross3dgs/dataset.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <numeric>
#include <stdexcept>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace cross3dgs {

namespace {

template <typename T>
T Read(std::istream& stream) {
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!stream) {
        throw std::runtime_error("unexpected end of COLMAP binary");
    }
    return value;
}

std::string ReadString(std::istream& stream) {
    std::string value;
    for (;;) {
        const char c = Read<char>(stream);
        if (c == '\0') {
            return value;
        }
        value.push_back(c);
    }
}

struct RawCamera {
    uint32_t id = 0;
    int32_t model = 0;
    uint64_t width = 0;
    uint64_t height = 0;
    std::vector<double> params;
};

struct RawImage {
    uint32_t id = 0;
    glm::dquat rotation{};
    glm::dvec3 translation{};
    uint32_t camera_id = 0;
    std::string name;
};

const std::map<int32_t, size_t> kCameraParameterCounts{
        {0, 3}, {1, 4}, {2, 4}, {3, 5}, {4, 8}, {5, 8},
        {6, 12}, {7, 5}, {8, 4}, {9, 5}, {10, 12}, {11, 16}};

std::map<uint32_t, RawCamera> ReadCameras(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open " + path.string());
    }
    std::map<uint32_t, RawCamera> result;
    const uint64_t count = Read<uint64_t>(stream);
    for (uint64_t i = 0; i < count; i++) {
        RawCamera camera;
        camera.id = Read<uint32_t>(stream);
        camera.model = Read<int32_t>(stream);
        camera.width = Read<uint64_t>(stream);
        camera.height = Read<uint64_t>(stream);
        const auto found = kCameraParameterCounts.find(camera.model);
        if (found == kCameraParameterCounts.end()) {
            throw std::runtime_error("unknown COLMAP camera model " +
                                     std::to_string(camera.model));
        }
        camera.params.resize(found->second);
        stream.read(reinterpret_cast<char*>(camera.params.data()),
                    static_cast<std::streamsize>(camera.params.size() *
                                                 sizeof(double)));
        if (!stream) {
            throw std::runtime_error("truncated cameras.bin");
        }
        result.emplace(camera.id, std::move(camera));
    }
    return result;
}

std::vector<RawImage> ReadImages(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open " + path.string());
    }
    std::vector<RawImage> result;
    const uint64_t count = Read<uint64_t>(stream);
    result.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; i++) {
        RawImage image;
        image.id = Read<uint32_t>(stream);
        const double qw = Read<double>(stream);
        const double qx = Read<double>(stream);
        const double qy = Read<double>(stream);
        const double qz = Read<double>(stream);
        image.rotation = glm::dquat(qw, qx, qy, qz);
        image.translation = {Read<double>(stream), Read<double>(stream),
                             Read<double>(stream)};
        image.camera_id = Read<uint32_t>(stream);
        image.name = ReadString(stream);
        const uint64_t point_count = Read<uint64_t>(stream);
        stream.seekg(static_cast<std::streamoff>(point_count * 24),
                     std::ios::cur);
        if (!stream) {
            throw std::runtime_error("truncated images.bin");
        }
        result.push_back(std::move(image));
    }
    std::sort(result.begin(), result.end(),
              [](const RawImage& left, const RawImage& right) {
                  return left.name < right.name;
              });
    return result;
}

std::vector<SparsePoint> ReadPoints(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open " + path.string());
    }
    std::vector<SparsePoint> result;
    const uint64_t count = Read<uint64_t>(stream);
    result.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; i++) {
        SparsePoint point;
        point.id = Read<uint64_t>(stream);
        point.position = {static_cast<float>(Read<double>(stream)),
                          static_cast<float>(Read<double>(stream)),
                          static_cast<float>(Read<double>(stream))};
        point.color = {Read<uint8_t>(stream), Read<uint8_t>(stream),
                       Read<uint8_t>(stream)};
        (void)Read<double>(stream);
        const uint64_t track_count = Read<uint64_t>(stream);
        stream.seekg(static_cast<std::streamoff>(track_count * 8),
                     std::ios::cur);
        if (!stream) {
            throw std::runtime_error("truncated points3D.bin");
        }
        result.push_back(point);
    }
    return result;
}

float Median(std::vector<float> values) {
    if (values.empty()) {
        return 0.0f;
    }
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    const float upper = values[middle];
    if (values.size() % 2 != 0) {
        return upper;
    }
    const float lower = *std::max_element(values.begin(),
                                          values.begin() + middle);
    return 0.5f * (lower + upper);
}

glm::vec3 MedianVec3(const std::vector<glm::vec3>& values) {
    std::vector<float> x, y, z;
    x.reserve(values.size());
    y.reserve(values.size());
    z.reserve(values.size());
    for (const auto& value : values) {
        x.push_back(value.x);
        y.push_back(value.y);
        z.push_back(value.z);
    }
    return {Median(std::move(x)), Median(std::move(y)), Median(std::move(z))};
}

void JacobiEigen(glm::dmat3 matrix, glm::dmat3& vectors,
                 glm::dvec3& values) {
    vectors = glm::dmat3(1.0);
    for (int iteration = 0; iteration < 16; iteration++) {
        int p = 0;
        int q = 1;
        double maximum = std::abs(matrix[p][q]);
        for (int i = 0; i < 3; i++) {
            for (int j = i + 1; j < 3; j++) {
                if (std::abs(matrix[i][j]) > maximum) {
                    maximum = std::abs(matrix[i][j]);
                    p = i;
                    q = j;
                }
            }
        }
        if (maximum < 1e-10) {
            break;
        }
        const double theta =
                0.5 * std::atan2(2.0 * matrix[p][q],
                                 matrix[q][q] - matrix[p][p]);
        const double c = std::cos(theta);
        const double s = std::sin(theta);
        glm::dmat3 rotation(1.0);
        rotation[p][p] = c;
        rotation[p][q] = -s;
        rotation[q][p] = s;
        rotation[q][q] = c;
        matrix = glm::transpose(rotation) * matrix * rotation;
        vectors = vectors * rotation;
    }
    values = {matrix[0][0], matrix[1][1], matrix[2][2]};
}

glm::mat4 SimilarityFromCameras(const std::vector<glm::mat4>& cameras) {
    std::vector<glm::vec3> positions, ups, forwards;
    for (const auto& camera : cameras) {
        positions.emplace_back(camera[3]);
        const glm::mat3 rotation(camera);
        ups.push_back(rotation * glm::vec3(0.0f, -1.0f, 0.0f));
        forwards.push_back(rotation * glm::vec3(0.0f, 0.0f, 1.0f));
    }
    glm::vec3 world_up(0.0f);
    for (const auto& up : ups) {
        world_up += up;
    }
    world_up = glm::normalize(world_up / static_cast<float>(ups.size()));
    const glm::vec3 camera_up(0.0f, -1.0f, 0.0f);
    const float cosine = glm::dot(camera_up, world_up);
    const glm::vec3 cross = glm::cross(world_up, camera_up);
    glm::mat3 align(1.0f);
    if (cosine > -1.0f + 1e-6f) {
        glm::mat3 skew(0.0f);
        skew[1][0] = -cross.z;
        skew[2][0] = cross.y;
        skew[0][1] = cross.z;
        skew[2][1] = -cross.x;
        skew[0][2] = -cross.y;
        skew[1][2] = cross.x;
        align = glm::mat3(1.0f) + skew +
                (skew * skew) * (1.0f / (1.0f + cosine));
    } else {
        align = glm::mat3(-1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                          0.0f, 0.0f, 1.0f);
    }
    std::vector<glm::vec3> nearest;
    for (size_t i = 0; i < positions.size(); i++) {
        positions[i] = align * positions[i];
        forwards[i] = align * forwards[i];
        nearest.push_back(positions[i] +
                          glm::dot(-positions[i], forwards[i]) * forwards[i]);
    }
    const glm::vec3 translation = -MedianVec3(nearest);
    std::vector<float> distances;
    for (const auto& position : positions) {
        distances.push_back(glm::length(position + translation));
    }
    const float scale = 1.0f / Median(std::move(distances));
    glm::mat4 transform = glm::mat4(align) *
                          glm::scale(glm::mat4(1.0f), glm::vec3(scale));
    transform[3] = glm::vec4(translation * scale, 1.0f);
    return transform;
}

glm::mat4 PrincipalAxes(const std::vector<SparsePoint>& points) {
    std::vector<glm::vec3> positions;
    positions.reserve(points.size());
    for (const auto& point : points) {
        positions.push_back(point.position);
    }
    const glm::vec3 center = MedianVec3(positions);
    glm::dmat3 covariance(0.0);
    glm::dvec3 mean(0.0);
    for (const auto& position : positions) {
        const glm::dvec3 value = glm::dvec3(position - center);
        mean += value;
        covariance += glm::outerProduct(value, value);
    }
    covariance /= static_cast<double>(positions.size() - 1);
    mean /= static_cast<double>(positions.size());
    covariance -= glm::outerProduct(mean, mean);
    glm::dmat3 vectors;
    glm::dvec3 values;
    JacobiEigen(covariance, vectors, values);
    std::array<int, 3> order{0, 1, 2};
    std::sort(order.begin(), order.end(),
              [&](int left, int right) { return values[left] > values[right]; });
    glm::mat3 sorted;
    for (int i = 0; i < 3; i++) {
        sorted[i] = glm::vec3(vectors[order[i]]);
    }
    if (glm::determinant(sorted) < 0.0f) {
        sorted[0] = -sorted[0];
    }
    const glm::mat3 rotation = glm::transpose(sorted);
    glm::mat4 transform(rotation);
    transform[3] = glm::vec4(-rotation * center, 1.0f);
    return transform;
}

void ApplyTransform(const glm::mat4& transform,
                    std::vector<glm::mat4>& camera_to_world,
                    std::vector<SparsePoint>& points) {
    for (auto& camera : camera_to_world) {
        camera = transform * camera;
        const float scale = glm::length(glm::vec3(camera[0]));
        for (int column = 0; column < 3; column++) {
            camera[column] /= scale;
        }
    }
    for (auto& point : points) {
        point.position = glm::vec3(transform * glm::vec4(point.position, 1.0f));
    }
}

Camera MakeCamera(const RawCamera& raw_camera, const RawImage& raw_image,
                  const std::filesystem::path& image_path, uint32_t index,
                  uint32_t eval_interval,
                  const std::pair<int, int>& fallback_dimensions) {
    int width = 0;
    int height = 0;
    int channels = 0;
    if (!stbi_info(image_path.string().c_str(), &width, &height, &channels)) {
        width = fallback_dimensions.first;
        height = fallback_dimensions.second;
        if (width <= 0 || height <= 0) {
            throw std::runtime_error("cannot inspect image " +
                                     image_path.string());
        }
    }
    Camera camera;
    camera.image_id = raw_image.id;
    camera.camera_id = raw_image.camera_id;
    camera.image_name = raw_image.name;
    camera.image_path = image_path;
    camera.width = static_cast<uint32_t>(width);
    camera.height = static_cast<uint32_t>(height);
    camera.validation = index % eval_interval == 0;
    const float sx = static_cast<float>(width) /
                     static_cast<float>(raw_camera.width);
    const float sy = static_cast<float>(height) /
                     static_cast<float>(raw_camera.height);
    if (raw_camera.model == 0 || raw_camera.model == 2 ||
        raw_camera.model == 3 || raw_camera.model == 8 ||
        raw_camera.model == 9) {
        camera.fx = camera.fy = static_cast<float>(raw_camera.params[0]);
        camera.cx = static_cast<float>(raw_camera.params[1]);
        camera.cy = static_cast<float>(raw_camera.params[2]);
    } else {
        camera.fx = static_cast<float>(raw_camera.params[0]);
        camera.fy = static_cast<float>(raw_camera.params[1]);
        camera.cx = static_cast<float>(raw_camera.params[2]);
        camera.cy = static_cast<float>(raw_camera.params[3]);
    }
    camera.fx *= sx;
    camera.fy *= sy;
    camera.cx *= sx;
    camera.cy *= sy;
    if (raw_camera.model == 2 || raw_camera.model == 3) {
        camera.model = CameraModel::OpenCv;
        camera.distortion[0] = static_cast<float>(raw_camera.params[3]);
        if (raw_camera.model == 3) {
            camera.distortion[1] = static_cast<float>(raw_camera.params[4]);
        }
    } else if (raw_camera.model == 4 || raw_camera.model == 6) {
        camera.model = CameraModel::OpenCv;
        for (int i = 0; i < 4; i++) {
            camera.distortion[i] = static_cast<float>(raw_camera.params[4 + i]);
        }
    } else if (raw_camera.model == 5 || raw_camera.model == 8 ||
               raw_camera.model == 9 || raw_camera.model == 10) {
        camera.model = CameraModel::OpenCvFisheye;
        for (size_t i = 0; i < camera.distortion.size() &&
                           3 + i < raw_camera.params.size();
             i++) {
            camera.distortion[i] = static_cast<float>(raw_camera.params[3 + i]);
        }
    }
    const glm::dmat4 rotation = glm::toMat4(raw_image.rotation);
    glm::dmat4 world_to_camera = rotation;
    world_to_camera[3] = glm::dvec4(raw_image.translation, 1.0);
    camera.world_to_camera = glm::mat4(world_to_camera);
    return camera;
}

}  // namespace

size_t Scene::trainCameraCount() const {
    return static_cast<size_t>(std::count_if(
            cameras.begin(), cameras.end(),
            [](const Camera& camera) { return !camera.validation; }));
}

size_t Scene::validationCameraCount() const {
    return cameras.size() - trainCameraCount();
}

Scene LoadColmapScene(const std::filesystem::path& scene_path,
                      const std::string& image_dir,
                      const std::string& sparse_dir, uint32_t eval_interval,
                      bool normalize_world) {
    if (eval_interval == 0) {
        throw std::invalid_argument("eval_interval must be positive");
    }
    const auto sparse_path = scene_path / sparse_dir;
    const auto raw_cameras = ReadCameras(sparse_path / "cameras.bin");
    const auto raw_images = ReadImages(sparse_path / "images.bin");
    Scene scene;
    scene.scene_path = std::filesystem::absolute(scene_path);
    scene.image_dir = image_dir;
    scene.sparse_dir = sparse_dir;
    scene.points = ReadPoints(sparse_path / "points3D.bin");
    // Mobile bundles may intentionally carry only the selected training
    // images while retaining every COLMAP pose for identical normalization.
    // Resolve one real image size per camera and use it for missing siblings.
    std::map<uint32_t, std::pair<int, int>> dimensions_by_camera;
    for (const auto& raw_image : raw_images) {
        if (dimensions_by_camera.contains(raw_image.camera_id)) {
            continue;
        }
        int width = 0;
        int height = 0;
        int channels = 0;
        const auto image_path =
                scene.scene_path / image_dir / raw_image.name;
        if (stbi_info(image_path.string().c_str(), &width, &height,
                      &channels)) {
            dimensions_by_camera.emplace(raw_image.camera_id,
                                         std::pair{width, height});
        }
    }
    scene.cameras.reserve(raw_images.size());
    std::vector<glm::mat4> camera_to_world;
    camera_to_world.reserve(raw_images.size());
    for (uint32_t index = 0; index < raw_images.size(); index++) {
        const auto& raw_image = raw_images[index];
        const auto found = raw_cameras.find(raw_image.camera_id);
        if (found == raw_cameras.end()) {
            throw std::runtime_error("image references an unknown camera");
        }
        const auto dimensions = dimensions_by_camera.find(raw_image.camera_id);
        if (dimensions == dimensions_by_camera.end()) {
            throw std::runtime_error(
                    "no bundled image is available for camera " +
                    std::to_string(raw_image.camera_id));
        }
        scene.cameras.push_back(MakeCamera(
                found->second, raw_image, scene.scene_path / image_dir /
                                                   raw_image.name,
                index, eval_interval, dimensions->second));
        camera_to_world.push_back(glm::inverse(scene.cameras.back().world_to_camera));
    }
    if (normalize_world) {
        const glm::mat4 first = SimilarityFromCameras(camera_to_world);
        ApplyTransform(first, camera_to_world, scene.points);
        const glm::mat4 second = PrincipalAxes(scene.points);
        ApplyTransform(second, camera_to_world, scene.points);
        scene.dataset_transform = second * first;
        std::vector<float> z;
        z.reserve(scene.points.size());
        float mean = 0.0f;
        for (const auto& point : scene.points) {
            z.push_back(point.position.z);
            mean += point.position.z;
        }
        mean /= static_cast<float>(scene.points.size());
        if (Median(z) > mean) {
            glm::mat4 flip(1.0f);
            flip[1][1] = -1.0f;
            flip[2][2] = -1.0f;
            ApplyTransform(flip, camera_to_world, scene.points);
            scene.dataset_transform = flip * scene.dataset_transform;
        }
        for (size_t i = 0; i < scene.cameras.size(); i++) {
            scene.cameras[i].world_to_camera = glm::inverse(camera_to_world[i]);
        }
    }
    return scene;
}

float CameraExtentSceneScale(const std::vector<Camera>& cameras) {
    if (cameras.empty()) {
        throw std::invalid_argument("cannot compute scene scale without cameras");
    }
    glm::vec3 center(0.0f);
    for (const auto& camera : cameras) {
        center += glm::vec3(glm::inverse(camera.world_to_camera)[3]);
    }
    center /= static_cast<float>(cameras.size());
    float maximum_distance = 0.0f;
    for (const auto& camera : cameras) {
        const glm::vec3 position =
                glm::vec3(glm::inverse(camera.world_to_camera)[3]);
        maximum_distance =
                std::max(maximum_distance, glm::distance(position, center));
    }
    return maximum_distance * 1.1f;
}

FloatImage LoadImageRgbaF32(const std::filesystem::path& path) {
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* data = stbi_load(path.string().c_str(), &width, &height, &channels, 4);
    if (!data) {
        throw std::runtime_error("cannot load image " + path.string());
    }
    FloatImage image;
    image.width = static_cast<uint32_t>(width);
    image.height = static_cast<uint32_t>(height);
    image.pixels.resize(static_cast<size_t>(width) * height * 4);
    std::transform(data, data + image.pixels.size(), image.pixels.begin(),
                   [](uint8_t value) { return static_cast<float>(value) / 255.0f; });
    stbi_image_free(data);
    return image;
}

void SaveImagePng(const std::filesystem::path& path, const FloatImage& image) {
    if (image.width == 0 || image.height == 0 || image.channels < 3 ||
        image.pixels.size() <
                static_cast<size_t>(image.width) * image.height *
                        image.channels) {
        throw std::invalid_argument("invalid float image");
    }
    std::vector<uint8_t> bytes(
            static_cast<size_t>(image.width) * image.height * 3);
    for (size_t pixel = 0;
         pixel < static_cast<size_t>(image.width) * image.height; pixel++) {
        for (size_t channel = 0; channel < 3; channel++) {
            const float value = std::clamp(
                    image.pixels[pixel * image.channels + channel], 0.0f, 1.0f);
            bytes[pixel * 3 + channel] =
                    static_cast<uint8_t>(value * 255.0f + 0.5f);
        }
    }
    std::filesystem::create_directories(path.parent_path());
    const auto path_string = path.string();
    if (stbi_write_png(path_string.c_str(), static_cast<int>(image.width),
                       static_cast<int>(image.height), 3, bytes.data(),
                       static_cast<int>(image.width * 3)) == 0) {
        throw std::runtime_error("failed to write PNG: " + path_string);
    }
}

}  // namespace cross3dgs
