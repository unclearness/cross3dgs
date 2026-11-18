#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "cross3dgs/orbit_view.h"

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        g_failures++;
    }
}

// World down of the synthetic scene: deliberately not a world axis, like the
// principal-axes-normalized world of LoadColmapScene.
const glm::vec3 kDown = glm::normalize(glm::vec3(0.3f, 0.8f, -0.5f));

// Camera at `position` looking at the origin, rolled by `roll` radians about
// its optical axis away from the level pose (a tilted horizon).
cross3dgs::Camera MakeCamera(const glm::vec3& position, float roll,
                             const std::string& name) {
    const glm::vec3 forward = glm::normalize(-position);
    const glm::vec3 level_right = glm::normalize(glm::cross(kDown, forward));
    const glm::vec3 level_down = glm::cross(forward, level_right);
    const glm::vec3 right =
            std::cos(roll) * level_right + std::sin(roll) * level_down;
    const glm::vec3 down =
            -std::sin(roll) * level_right + std::cos(roll) * level_down;
    glm::mat4 camera_to_world(1.0f);
    camera_to_world[0] = glm::vec4(right, 0.0f);
    camera_to_world[1] = glm::vec4(down, 0.0f);
    camera_to_world[2] = glm::vec4(forward, 0.0f);
    camera_to_world[3] = glm::vec4(position, 1.0f);
    cross3dgs::Camera camera;
    camera.image_name = name;
    camera.width = 1920;
    camera.height = 1080;
    camera.fx = 1400.0f;
    camera.fy = 1400.0f;
    camera.cx = 980.0f;   // off-center principal point
    camera.cy = 530.0f;
    camera.world_to_camera = glm::inverse(camera_to_world);
    return camera;
}

glm::vec3 GravityInCamera(const cross3dgs::Camera& camera) {
    return glm::mat3(camera.world_to_camera) * kDown;
}

glm::vec2 Project(const cross3dgs::Camera& camera, const glm::vec3& world) {
    const glm::vec3 p(camera.world_to_camera * glm::vec4(world, 1.0f));
    return {camera.fx * p.x / p.z + camera.cx, camera.fy * p.y / p.z + camera.cy};
}

// World point seen at pixel (u, v) of `camera` at depth 3.
glm::vec3 Unproject(const cross3dgs::Camera& camera, float u, float v) {
    const glm::vec3 ray((u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy,
                        1.0f);
    return glm::vec3(glm::inverse(camera.world_to_camera) *
                     glm::vec4(3.0f * ray, 1.0f));
}

void TestReadFramePriorGravity() {
    const auto path = std::filesystem::temp_directory_path() /
                      "cross3dgs_orbit_view_priors.txt";
    {
        std::ofstream file(path);
        file << "# intrinsics 1400 1400 960 540 1920 1080\n"
             << "frame_0000.jpg 0 0 0 0 2 0 0.08\n"
             << "frame_0001.jpg 1 0 0 0 0 0 0.08\n"   // no gravity
             << "frame_0002.jpg 1 0 0 0.6 0.8 0 0.2\n"
             << "malformed\n";
    }
    const auto gravity = cross3dgs::ReadFramePriorGravity(path);
    std::filesystem::remove(path);
    Check(gravity.size() == 2, "priors: two usable gravity rows");
    Check(gravity.contains("frame_0000.jpg") &&
                  glm::length(gravity.at("frame_0000.jpg") -
                              glm::vec3(0, 1, 0)) < 1e-6f,
          "priors: gravity normalized");
    Check(!gravity.contains("frame_0001.jpg"), "priors: zero gravity skipped");
    Check(cross3dgs::ReadFramePriorGravity("does/not/exist.txt").empty(),
          "priors: missing file is empty");
}

void TestEstimateWorldDown() {
    std::vector<cross3dgs::Camera> cameras;
    std::unordered_map<std::string, glm::vec3> gravity;
    for (int i = 0; i < 12; i++) {
        const float angle = 0.5f * static_cast<float>(i);
        const glm::vec3 position(4.0f * std::cos(angle), 1.0f + 0.2f * i,
                                 4.0f * std::sin(angle));
        const std::string name = "frame_" + std::to_string(i) + ".jpg";
        cameras.push_back(MakeCamera(position, 0.3f * (i % 5) - 0.6f, name));
        // Small per-frame VIO noise.
        const glm::vec3 noise(0.01f * std::sin(3.0f * i), 0.01f * std::cos(5.0f * i),
                              0.01f * std::sin(7.0f * i));
        gravity[name] = glm::normalize(GravityInCamera(cameras.back()) + noise);
    }
    const auto estimate = cross3dgs::EstimateWorldDown(cameras, gravity);
    Check(estimate.has_value(), "world down: estimated");
    if (estimate) {
        Check(glm::dot(estimate->down, kDown) > 0.9999f, "world down: direction");
        Check(estimate->cameras == 12, "world down: all cameras used");
    }

    std::unordered_map<std::string, glm::vec3> two(gravity.begin(), gravity.end());
    while (two.size() > 2) two.erase(two.begin());
    Check(!cross3dgs::EstimateWorldDown(cameras, two).has_value(),
          "world down: fewer than three cameras rejected");

    auto inconsistent = gravity;
    int flip = 0;
    for (auto& [name, g] : inconsistent) {
        if (flip++ % 2 == 0) g = -g;
    }
    Check(!cross3dgs::EstimateWorldDown(cameras, inconsistent).has_value(),
          "world down: disagreeing priors rejected");
}

void TestLevelCamera() {
    // Looking down at the subject with a tilted phone.
    const glm::vec3 position = 3.0f * glm::normalize(glm::vec3(1.0f, 0.0f, 1.0f)) -
                               1.5f * kDown;
    for (const float roll : {0.0f, 0.35f, -0.6f, 1.2f, -1.4f, 3.0f}) {
        const auto camera = MakeCamera(position, roll, "a");
        const auto level = cross3dgs::LevelCamera(camera, kDown);
        const glm::mat4 c2w = glm::inverse(level.world_to_camera);
        const glm::mat4 ref_c2w = glm::inverse(camera.world_to_camera);
        const std::string label = " (roll " + std::to_string(roll) + ")";
        Check(std::abs(glm::dot(glm::vec3(c2w[0]), kDown)) < 1e-4f,
              "level: right axis horizontal" + label);
        Check(glm::dot(glm::vec3(c2w[1]), kDown) > 0.0f,
              "level: image-down points with gravity" + label);
        Check(glm::length(glm::vec3(c2w[2]) - glm::vec3(ref_c2w[2])) < 1e-5f,
              "level: optical axis unchanged" + label);
        Check(glm::length(glm::vec3(c2w[3]) - position) < 1e-4f,
              "level: position unchanged" + label);
        const bool portrait = std::abs(std::sin(roll)) > std::abs(std::cos(roll));
        Check((level.width == 1080) == portrait,
              "level: aspect swaps only near 90 degrees" + label);
    }

    // Exact 90-degree roll (portrait capture): every reference pixel maps to
    // the rotated pixel of the swapped image, so the corners land on corners.
    const auto portrait = MakeCamera(position, 1.5707964f, "p");
    const auto level = cross3dgs::LevelCamera(portrait, kDown);
    Check(level.width == 1080 && level.height == 1920, "portrait: swapped size");
    for (const glm::vec2 corner : {glm::vec2(0, 0), glm::vec2(1920, 0),
                                   glm::vec2(0, 1080), glm::vec2(1920, 1080)}) {
        const glm::vec2 p = Project(level, Unproject(portrait, corner.x, corner.y));
        const bool on_corner =
                (std::abs(p.x) < 0.05f || std::abs(p.x - 1080.0f) < 0.05f) &&
                (std::abs(p.y) < 0.05f || std::abs(p.y - 1920.0f) < 0.05f);
        Check(on_corner, "portrait: corner maps to corner");
    }

    // Looking straight down: any roll is as level as another, so the camera
    // is kept unchanged.
    cross3dgs::Camera nadir = portrait;
    const glm::vec3 right = glm::normalize(glm::cross(kDown, glm::vec3(0, 0, 1)));
    glm::mat4 nadir_to_world(1.0f);
    nadir_to_world[0] = glm::vec4(right, 0.0f);
    nadir_to_world[1] = glm::vec4(glm::cross(kDown, right), 0.0f);
    nadir_to_world[2] = glm::vec4(kDown, 0.0f);
    nadir_to_world[3] = glm::vec4(-3.0f * kDown, 1.0f);
    nadir.world_to_camera = glm::inverse(nadir_to_world);
    const auto kept = cross3dgs::LevelCamera(nadir, kDown);
    Check(kept.world_to_camera == nadir.world_to_camera && kept.width == nadir.width,
          "nadir: camera kept");
}

void TestOrbitCamera() {
    const glm::vec3 position = 3.0f * glm::normalize(glm::vec3(1.0f, 0.0f, 1.0f)) -
                               1.5f * kDown;
    const auto level = cross3dgs::LevelCamera(MakeCamera(position, 0.4f, "a"), kDown);
    const float distance = glm::length(position);
    for (const float yaw : {0.0f, 0.7f, 2.0f, -2.5f}) {
        for (const float pitch : {0.0f, 0.3f, -0.4f}) {
            const auto view = cross3dgs::OrbitCamera(level, kDown, glm::vec3(0.0f),
                                                     yaw, pitch, 1.0f, 0.0f, 0.0f,
                                                     1920, 1080);
            const glm::mat4 c2w = glm::inverse(view.world_to_camera);
            Check(std::abs(glm::dot(glm::vec3(c2w[0]), kDown)) < 1e-4f,
                  "orbit: horizon stays level");
            Check(std::abs(glm::length(glm::vec3(c2w[3])) - distance) < 1e-3f,
                  "orbit: distance to the pivot preserved");
        }
    }
    // The fallback (yaw about the camera's own down axis) tilts the horizon
    // for a camera that looks down at the subject: the behavior replaced
    // when gravity is known.
    const auto tilted = cross3dgs::OrbitCamera(
            level, cross3dgs::CameraDownAxis(level), glm::vec3(0.0f), 1.5f, 0.0f,
            1.0f, 0.0f, 0.0f, 1920, 1080);
    const glm::mat4 c2w = glm::inverse(tilted.world_to_camera);
    Check(std::abs(glm::dot(glm::vec3(c2w[0]), kDown)) > 0.1f,
          "orbit: camera-down yaw axis tilts the horizon (fallback)");
}

// Sparse points: a subject cluster around `subject` plus a larger cluster
// behind the test cameras that drags the median of all points (the world
// origin after LoadColmapScene) away from the subject, like a partial
// walk-around.
std::vector<cross3dgs::SparsePoint> MakePoints(const glm::vec3& subject) {
    std::vector<cross3dgs::SparsePoint> points;
    for (int i = 0; i < 200; i++) {
        const float t = static_cast<float>(i);
        cross3dgs::SparsePoint point;
        point.position = subject + 0.15f * glm::vec3(std::sin(1.3f * t),
                                                     std::cos(1.7f * t),
                                                     std::sin(2.9f * t));
        points.push_back(point);
    }
    for (int i = 0; i < 400; i++) {
        const float t = static_cast<float>(i);
        cross3dgs::SparsePoint point;
        point.position = glm::vec3(6.0f, 0.0f, 6.0f) +
                         4.0f * glm::vec3(std::sin(0.7f * t), std::cos(1.1f * t),
                                          std::sin(0.3f * t));
        points.push_back(point);
    }
    return points;
}

void TestOrbitPivot() {
    const glm::vec3 subject(0.6f, 0.2f, -0.4f);
    // Camera looking at the subject from 3 units away, slightly from above.
    const glm::vec3 position = subject + 3.0f * glm::normalize(glm::vec3(1.0f, 0.0f, 1.0f) -
                                                               0.6f * kDown);
    cross3dgs::Camera camera = MakeCamera(position - subject, 0.2f, "a");
    glm::mat4 c2w = glm::inverse(camera.world_to_camera);
    c2w[3] = glm::vec4(position, 1.0f);
    camera.world_to_camera = glm::inverse(c2w);
    camera.cx = 960.0f;
    camera.cy = 540.0f;

    const auto pivot = cross3dgs::EstimateOrbitPivot(camera, MakePoints(subject));
    Check(pivot.points >= 16, "pivot: from central sparse points");
    Check(std::abs(pivot.depth - 3.0f) < 0.1f,
          "pivot: subject depth (" + std::to_string(pivot.depth) + ")");
    Check(glm::distance(pivot.position, subject) < 0.1f, "pivot: at the subject");

    // No points: the axis point nearest the origin; origin behind the
    // camera: the axis point at the origin's distance.
    const auto empty = cross3dgs::EstimateOrbitPivot(camera, {});
    Check(empty.points == 0, "pivot: no points");
    const glm::vec3 forward(c2w[2]);
    Check(std::abs(empty.depth - glm::dot(-position, forward)) < 1e-4f,
          "pivot: nearest axis point to the origin without points");
    glm::mat4 away = c2w;
    away[2] = -away[2];
    away[0] = -away[0];
    cross3dgs::Camera backwards = camera;
    backwards.world_to_camera = glm::inverse(away);
    const auto behind = cross3dgs::EstimateOrbitPivot(backwards, {});
    Check(std::abs(behind.depth - glm::length(position)) < 1e-4f,
          "pivot: origin behind the camera keeps the pivot in front");

    // Orbiting the pivot keeps it on the optical axis: the subject stays at
    // the principal point while the horizon stays level.
    const auto level = cross3dgs::LevelCamera(camera, kDown);
    for (const float yaw : {0.0f, 0.8f, -2.0f}) {
        for (const float pitch : {0.0f, 0.5f, -0.6f}) {
            const auto view = cross3dgs::OrbitCamera(level, kDown, pivot.position, yaw,
                                                     pitch, 1.0f, 0.0f, 0.0f, 1920, 1080);
            const glm::vec2 p = Project(view, pivot.position);
            Check(glm::length(p - glm::vec2(view.cx, view.cy)) < 0.05f,
                  "orbit: pivot stays at the principal point");
            const glm::mat4 v2w = glm::inverse(view.world_to_camera);
            Check(std::abs(glm::distance(glm::vec3(v2w[3]), pivot.position) -
                           pivot.depth) < 1e-3f,
                  "orbit: distance to the pivot preserved");
            Check(std::abs(glm::dot(glm::vec3(v2w[0]), kDown)) < 1e-4f,
                  "orbit: horizon level around the pivot");
        }
    }
}

void TestMakeOrbitReference() {
    std::vector<cross3dgs::Camera> cameras;
    const auto path = std::filesystem::temp_directory_path() /
                      "cross3dgs_orbit_reference_priors.txt";
    {
        std::ofstream file(path);
        file << "# intrinsics 1400 1400 960 540 1920 1080\n";
        for (int i = 0; i < 6; i++) {
            const float angle = 0.8f * static_cast<float>(i);
            const std::string name = "frame_" + std::to_string(i) + ".jpg";
            cameras.push_back(MakeCamera(
                    glm::vec3(4.0f * std::cos(angle), -1.0f, 4.0f * std::sin(angle)),
                    0.5f, name));
            const glm::vec3 g = GravityInCamera(cameras.back());
            file << name << " 0 0 0 " << g.x << ' ' << g.y << ' ' << g.z << " 0.08\n";
        }
    }
    // COLMAP -> scene: uniform scale 2 and 90 degrees about Z, as
    // LoadColmapScene's similarity would produce.
    glm::mat4 dataset_transform(1.0f);
    dataset_transform[0] = glm::vec4(0.0f, 2.0f, 0.0f, 0.0f);
    dataset_transform[1] = glm::vec4(-2.0f, 0.0f, 0.0f, 0.0f);
    dataset_transform[2] = glm::vec4(0.0f, 0.0f, 2.0f, 0.0f);
    const glm::vec3 subject(0.3f, -0.2f, 0.1f);
    const auto points = MakePoints(subject);
    const auto aligned = cross3dgs::MakeOrbitReference(cameras, cameras[2], path,
                                                       dataset_transform, points);
    std::filesystem::remove(path);
    const auto format = [](const glm::vec3& v) {
        char text[96];
        std::snprintf(text, sizeof(text), "(%.4f, %.4f, %.4f)", v.x, v.y, v.z);
        return std::string(text);
    };
    // Scene down (x, y, z) is COLMAP down (y, -x, z) under this rotation.
    const glm::vec3 colmap_down(kDown.y, -kDown.x, kDown.z);
    Check(aligned.summary.find("world down " + format(kDown) +
                               " in the scene (3DGS) frame, " +
                               format(colmap_down) + " in the COLMAP sparse frame") !=
                  std::string::npos,
          "reference: summary records the down vector in both frames (" +
                  aligned.summary + ")");
    Check(aligned.gravity_aligned, "reference: gravity aligned");
    Check(glm::dot(aligned.yaw_axis, kDown) > 0.9999f, "reference: yaw axis is gravity");
    Check(std::abs(glm::dot(glm::vec3(glm::inverse(aligned.camera.world_to_camera)[0]),
                            kDown)) < 1e-4f,
          "reference: levelled");
    Check(aligned.summary.find("rolled -28.6 deg") != std::string::npos,
          "reference: summary reports the roll (" + aligned.summary + ")");
    const glm::vec3 axis(glm::inverse(aligned.camera.world_to_camera)[2]);
    const glm::vec3 eye(glm::inverse(aligned.camera.world_to_camera)[3]);
    Check(glm::length(glm::cross(glm::normalize(aligned.pivot - eye), axis)) < 1e-4f,
          "reference: pivot on the optical axis");
    // This camera looks at the origin, so the subject is off-axis: the pivot
    // is the axis point at the subject's depth.
    Check(std::abs(glm::dot(aligned.pivot - eye, axis) -
                   glm::dot(subject - eye, axis)) < 0.1f,
          "reference: pivot at the subject depth, not the background median");
    // The pitch range keeps the view between 10 degrees below the horizon
    // and 85 degrees above it, and contains the initial view.
    Check(aligned.min_pitch <= 0.0f && aligned.max_pitch >= 0.0f,
          "reference: pitch range contains the initial view");
    for (const float pitch : {-10.0f, aligned.min_pitch, aligned.max_pitch, 10.0f}) {
        const auto view = cross3dgs::OrbitCamera(aligned, 0.7f, pitch, 1.0f, 0.0f, 0.0f,
                                                 1920, 1080);
        const glm::mat4 v2w = glm::inverse(view.world_to_camera);
        const float look_down = glm::degrees(
                std::asin(glm::dot(glm::vec3(v2w[2]), kDown)));
        Check(look_down < 85.01f && look_down > -10.01f,
              "reference: pitch clamped (" + std::to_string(look_down) + " deg)");
        Check(glm::dot(glm::vec3(v2w[1]), kDown) > 0.0f,
              "reference: never upside down");
    }
    const auto lowest = cross3dgs::OrbitCamera(aligned, 0.0f, 10.0f, 1.0f, 0.0f, 0.0f,
                                               1920, 1080);
    Check(std::abs(glm::degrees(std::asin(glm::dot(
                  glm::vec3(glm::inverse(lowest.world_to_camera)[2]), kDown))) +
                  10.0f) < 0.01f,
          "reference: lowest view looks 10 degrees up");

    const auto fallback = cross3dgs::MakeOrbitReference(cameras, cameras[2], {},
                                                        glm::mat4(1.0f), points);
    Check(!fallback.gravity_aligned, "reference: no priors -> fallback");
    Check(fallback.camera.world_to_camera == cameras[2].world_to_camera,
          "reference: fallback keeps the camera");
    Check(glm::length(fallback.yaw_axis - cross3dgs::CameraDownAxis(cameras[2])) < 1e-6f,
          "reference: fallback yaw axis is the camera's down axis");
    Check(glm::distance(fallback.pivot, aligned.pivot) < 1e-4f,
          "reference: fallback orbits the same pivot");
    Check(fallback.min_pitch == -cross3dgs::kMaxOrbitPitch &&
                  fallback.max_pitch == cross3dgs::kMaxOrbitPitch,
          "reference: fallback keeps the full pitch range");
}

}  // namespace

int main() {
    TestReadFramePriorGravity();
    TestEstimateWorldDown();
    TestLevelCamera();
    TestOrbitCamera();
    TestOrbitPivot();
    TestMakeOrbitReference();
    if (g_failures != 0) {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "orbit view tests passed\n";
    return 0;
}
