#include "cross3dgs/orbit_view.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <utility>

#include <glm/gtc/matrix_transform.hpp>

namespace cross3dgs {

std::unordered_map<std::string, glm::vec3> ReadFramePriorGravity(
        const std::filesystem::path& frame_priors_path) {
    std::unordered_map<std::string, glm::vec3> gravity;
    std::ifstream file(frame_priors_path);
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream fields(line);
        std::string name;
        double tx, ty, tz, gx, gy, gz;
        if (!(fields >> name >> tx >> ty >> tz >> gx >> gy >> gz)) continue;
        const glm::vec3 g(static_cast<float>(gx), static_cast<float>(gy),
                          static_cast<float>(gz));
        // A (near-)zero vector means "no gravity", as in the SfM reader.
        if (glm::length(g) < 0.5f) continue;
        gravity.emplace(std::move(name), glm::normalize(g));
    }
    return gravity;
}

std::optional<WorldDownEstimate> EstimateWorldDown(
        const std::vector<Camera>& cameras,
        const std::unordered_map<std::string, glm::vec3>& gravity_by_image) {
    glm::vec3 sum(0.0f);
    uint32_t count = 0;
    for (const auto& camera : cameras) {
        const auto found = gravity_by_image.find(camera.image_name);
        if (found == gravity_by_image.end()) continue;
        const glm::mat3 camera_to_world(glm::inverse(camera.world_to_camera));
        sum += glm::normalize(camera_to_world * found->second);
        count++;
    }
    if (count < 3) return std::nullopt;
    const glm::vec3 mean = sum / static_cast<float>(count);
    const float agreement = glm::length(mean);
    if (agreement < 0.9f) return std::nullopt;
    return WorldDownEstimate{mean / agreement, count, agreement};
}

glm::vec3 CameraDownAxis(const Camera& camera) {
    return glm::normalize(
            glm::vec3(glm::inverse(camera.world_to_camera)[1]));
}

Camera LevelCamera(const Camera& reference, const glm::vec3& world_down) {
    const glm::mat4 camera_to_world = glm::inverse(reference.world_to_camera);
    const glm::vec3 right = glm::normalize(glm::vec3(camera_to_world[0]));
    const glm::vec3 down = glm::normalize(glm::vec3(camera_to_world[1]));
    const glm::vec3 forward = glm::normalize(glm::vec3(camera_to_world[2]));
    const glm::vec3 g = glm::normalize(world_down);
    // Gravity projected onto the image plane; undefined when looking along
    // gravity (straight down or up), where any roll is as level as another.
    const glm::vec3 in_plane = g - glm::dot(g, forward) * forward;
    if (glm::length(in_plane) < 0.1f) return reference;
    const glm::vec3 new_down = glm::normalize(in_plane);
    const glm::vec3 new_right = glm::cross(new_down, forward);

    Camera level = reference;
    const float roll_cos = glm::dot(down, new_down);
    const float roll_sin = glm::dot(right, new_down);
    const bool swap_axes = std::abs(roll_sin) > std::abs(roll_cos);
    const float width = swap_axes ? static_cast<float>(reference.height)
                                  : static_cast<float>(reference.width);
    const float height = swap_axes ? static_cast<float>(reference.width)
                                   : static_cast<float>(reference.height);
    if (swap_axes) {
        level.width = reference.height;
        level.height = reference.width;
        level.fx = reference.fy;
        level.fy = reference.fx;
    }
    // Principal-point offset from the image center, re-expressed in the
    // rolled image axes (exact for 90-degree rolls, approximate otherwise
    // only when fx != fy).
    const glm::vec3 offset =
            (reference.cx - 0.5f * reference.width) * right +
            (reference.cy - 0.5f * reference.height) * down;
    level.cx = 0.5f * width + glm::dot(offset, new_right);
    level.cy = 0.5f * height + glm::dot(offset, new_down);

    glm::mat4 level_to_world(1.0f);
    level_to_world[0] = glm::vec4(new_right, 0.0f);
    level_to_world[1] = glm::vec4(new_down, 0.0f);
    level_to_world[2] = glm::vec4(forward, 0.0f);
    level_to_world[3] = camera_to_world[3];
    level.world_to_camera = glm::inverse(level_to_world);
    return level;
}

OrbitPivot EstimateOrbitPivot(const Camera& reference,
                              const std::vector<SparsePoint>& points) {
    constexpr size_t kMinPoints = 16;
    const glm::mat4 camera_to_world = glm::inverse(reference.world_to_camera);
    const glm::vec3 position(camera_to_world[3]);
    const glm::vec3 forward = glm::normalize(glm::vec3(camera_to_world[2]));
    std::vector<float> central, visible;
    for (const auto& point : points) {
        const glm::vec3 p(reference.world_to_camera *
                          glm::vec4(point.position, 1.0f));
        if (p.z <= 1e-4f) continue;
        const float u = (reference.fx * p.x / p.z + reference.cx) /
                        static_cast<float>(reference.width);
        const float v = (reference.fy * p.y / p.z + reference.cy) /
                        static_cast<float>(reference.height);
        if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) continue;
        visible.push_back(p.z);
        if (std::abs(u - 0.5f) < 0.2f && std::abs(v - 0.5f) < 0.2f) {
            central.push_back(p.z);
        }
    }
    auto& depths = central.size() >= kMinPoints ? central : visible;
    OrbitPivot pivot;
    if (depths.size() >= kMinPoints) {
        const auto middle = depths.begin() + depths.size() / 2;
        std::nth_element(depths.begin(), middle, depths.end());
        pivot.depth = *middle;
        pivot.points = static_cast<uint32_t>(depths.size());
    } else {
        const float along = glm::dot(-position, forward);
        pivot.depth = along > 1e-3f ? along : std::max(glm::length(position), 1e-3f);
    }
    pivot.position = position + pivot.depth * forward;
    return pivot;
}

OrbitReference MakeOrbitReference(
        const std::vector<Camera>& scene_cameras, const Camera& reference,
        const std::filesystem::path& frame_priors_path,
        const glm::mat4& dataset_transform,
        const std::vector<SparsePoint>& points) {
    const auto gravity = frame_priors_path.empty()
                                 ? std::unordered_map<std::string, glm::vec3>{}
                                 : ReadFramePriorGravity(frame_priors_path);
    const auto estimate = EstimateWorldDown(scene_cameras, gravity);
    // The pivot lies on the optical axis, which levelling keeps, so it is
    // the same for the levelled camera.
    const OrbitPivot pivot = EstimateOrbitPivot(reference, points);
    std::ostringstream pivot_text;
    pivot_text.setf(std::ios::fixed);
    pivot_text.precision(3);
    pivot_text << "; orbit pivot at depth " << pivot.depth;
    if (pivot.points != 0) {
        pivot_text << " (median of " << pivot.points << " sparse points)";
    } else {
        pivot_text << " (no sparse points in view)";
    }
    if (!estimate) {
        OrbitReference orbit;
        orbit.camera = reference;
        orbit.yaw_axis = CameraDownAxis(reference);
        orbit.pivot = pivot.position;
        orbit.summary = "Viewer: no consistent gravity priors (" +
                        std::to_string(gravity.size()) +
                        " rows); orbiting about the reference camera's down axis" +
                        pivot_text.str();
        return orbit;
    }
    const Camera level = LevelCamera(reference, estimate->down);
    const glm::mat3 before(glm::inverse(reference.world_to_camera));
    const glm::mat3 after(glm::inverse(level.world_to_camera));
    const float roll_degrees = glm::degrees(std::atan2(
            glm::dot(glm::cross(before[1], after[1]), before[2]),
            glm::dot(before[1], after[1])));
    std::ostringstream summary;
    summary.setf(std::ios::fixed);
    summary.precision(3);
    summary << "Viewer: gravity-aligned from " << estimate->cameras
            << " frame priors (agreement " << estimate->agreement << ")";
    summary.precision(1);
    summary << ", reference rolled " << roll_degrees << " deg to a level horizon";
    // Scene::dataset_transform maps COLMAP model coordinates to the scene
    // frame (rotation times uniform scale), so its inverse takes directions
    // back to the sparse model.
    const glm::vec3 colmap_down = glm::normalize(
            glm::inverse(glm::mat3(dataset_transform)) * estimate->down);
    const auto vector = [](const glm::vec3& v) {
        std::ostringstream text;
        text.setf(std::ios::fixed);
        text.precision(4);
        text << '(' << v.x << ", " << v.y << ", " << v.z << ')';
        return text.str();
    };
    summary << "; world down " << vector(estimate->down)
            << " in the scene (3DGS) frame, " << vector(colmap_down)
            << " in the COLMAP sparse frame" << pivot_text.str();

    // Pitch p turns the optical axis about the level right axis, so the
    // angle it looks down by is look_down - p (OrbitCamera's sign). Keep
    // that between -10 degrees (looking slightly up from below) and 85
    // degrees (almost straight down; past 90 the view flips over the top).
    const glm::vec3 forward = glm::normalize(
            glm::vec3(glm::inverse(level.world_to_camera)[2]));
    const float look_down = std::asin(
            std::clamp(glm::dot(forward, estimate->down), -1.0f, 1.0f));
    OrbitReference orbit;
    orbit.camera = level;
    orbit.yaw_axis = estimate->down;
    orbit.pivot = pivot.position;
    orbit.min_pitch = std::clamp(look_down - glm::radians(85.0f),
                                 -kMaxOrbitPitch, 0.0f);
    orbit.max_pitch = std::clamp(look_down + glm::radians(10.0f), 0.0f,
                                 kMaxOrbitPitch);
    orbit.gravity_aligned = true;
    summary.precision(2);
    summary << "; pitch range [" << orbit.min_pitch << ", " << orbit.max_pitch
            << "] rad";
    orbit.summary = summary.str();
    return orbit;
}

Camera OrbitCamera(const OrbitReference& orbit, float yaw, float pitch,
                   float zoom, float pan_x, float pan_y,
                   uint32_t requested_width, uint32_t requested_height) {
    return OrbitCamera(orbit.camera, orbit.yaw_axis, orbit.pivot, yaw,
                       std::clamp(pitch, orbit.min_pitch, orbit.max_pitch),
                       zoom, pan_x, pan_y, requested_width, requested_height);
}

Camera OrbitCamera(const Camera& reference, const glm::vec3& yaw_axis,
                   const glm::vec3& pivot, float yaw, float pitch, float zoom,
                   float pan_x, float pan_y, uint32_t requested_width,
                   uint32_t requested_height) {
    Camera camera = reference;
    const glm::mat4 base_camera_to_world =
            glm::inverse(reference.world_to_camera);
    const glm::vec3 base_offset = glm::vec3(base_camera_to_world[3]) - pivot;
    const float base_distance = std::max(glm::length(base_offset), 1e-3f);
    // The loader's normalization rotates the world onto the point cloud's
    // principal axes, so world Y is not vertical; the caller supplies the
    // vertical (gravity, or the reference camera's down axis).
    const glm::mat4 yaw_rotation =
            glm::rotate(glm::mat4(1.0f), yaw, glm::normalize(yaw_axis));
    const glm::vec3 base_right = glm::normalize(
            glm::vec3(base_camera_to_world[0]));
    const glm::mat4 pitch_rotation = glm::rotate(
            glm::mat4(1.0f), pitch, base_right);
    const glm::mat4 rotation = yaw_rotation * pitch_rotation;
    glm::mat4 camera_to_world = rotation * base_camera_to_world;
    const glm::vec3 right = glm::normalize(glm::vec3(camera_to_world[0]));
    const glm::vec3 down = glm::normalize(glm::vec3(camera_to_world[1]));
    const glm::vec3 pan = base_distance * (right * pan_x + down * pan_y);
    const float distance_scale = std::clamp(zoom, 0.2f, 5.0f);
    camera_to_world[3] = glm::vec4(
            pivot + pan +
                    glm::vec3(rotation * glm::vec4(base_offset, 0.0f)) *
                            distance_scale,
            1.0f);
    camera.world_to_camera = glm::inverse(camera_to_world);

    constexpr float kMaxRenderExtent = 960.0f;
    const float requested_max = static_cast<float>(
            std::max(requested_width, requested_height));
    const float extent_scale = std::min(
            1.0f, kMaxRenderExtent / std::max(requested_max, 1.0f));
    const uint32_t width = std::max(
            64u, static_cast<uint32_t>(
                         std::lround(requested_width * extent_scale)));
    const uint32_t height = std::max(
            64u, static_cast<uint32_t>(
                         std::lround(requested_height * extent_scale)));
    // Keep square pixels and the reference camera's field of view. When the
    // view aspect differs from the captured image, expand the view on the
    // surplus axis instead of stretching the scene.
    const float projection_scale = std::min(
            static_cast<float>(width) / reference.width,
            static_cast<float>(height) / reference.height);
    const float offset_x =
            (width - reference.width * projection_scale) * 0.5f;
    const float offset_y =
            (height - reference.height * projection_scale) * 0.5f;
    camera.width = width;
    camera.height = height;
    camera.fx = reference.fx * projection_scale;
    camera.fy = reference.fy * projection_scale;
    camera.cx = reference.cx * projection_scale + offset_x;
    camera.cy = reference.cy * projection_scale + offset_y;
    return camera;
}

}  // namespace cross3dgs
