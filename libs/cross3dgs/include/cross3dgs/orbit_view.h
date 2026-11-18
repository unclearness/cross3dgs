#pragma once

// Free-view orbit camera shared by the Android and iOS capture-to-view
// viewers. The view orbits the point the reference camera looks at (on its
// optical axis, at the depth of the sparse points near the image center), so
// the subject stays in place while turning. When the capture recorded
// gravity, the view is also gravity-aligned: the initial camera is rolled so
// its horizon is level, a horizontal drag turns about the world vertical, and
// the vertical turn stops short of the zenith and of looking far up from
// below. Without gravity it falls back to turning about the reference
// camera's own down axis.

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include "cross3dgs/dataset.h"

namespace cross3dgs {

// Per-image gravity ("down") in the COLMAP camera frame, read from a
// capture-to-view frame_priors.txt (`<image> tx ty tz gx gy gz sigma`).
// Images without a usable gravity prior are omitted; a missing file gives an
// empty map.
std::unordered_map<std::string, glm::vec3> ReadFramePriorGravity(
        const std::filesystem::path& frame_priors_path);

struct WorldDownEstimate {
    glm::vec3 down{0.0f, 1.0f, 0.0f};  // unit, in the scene's world frame
    uint32_t cameras = 0;              // cameras that contributed
    float agreement = 0.0f;            // length of the mean unit vector, (0, 1]
};

// Gravity in the scene's world frame (after any LoadColmapScene
// normalization), averaged over cameras whose image_name has a gravity
// prior. nullopt when fewer than three cameras contribute or they disagree
// (agreement below 0.9), e.g. for a failed reconstruction.
std::optional<WorldDownEstimate> EstimateWorldDown(
        const std::vector<Camera>& cameras,
        const std::unordered_map<std::string, glm::vec3>& gravity_by_image);

// Rolls `reference` about its optical axis so that image-down follows world
// down (a level horizon). Width/height and the focal lengths swap when the
// roll is nearer to +-90 degrees than to 0/180 (e.g. a portrait capture
// recorded in sensor orientation); the principal point keeps its offset from
// the image center. Unchanged when the camera looks almost straight up or
// down.
Camera LevelCamera(const Camera& reference, const glm::vec3& world_down);

struct OrbitPivot {
    glm::vec3 position{0.0f};  // world point on the reference optical axis
    float depth = 0.0f;        // distance from the reference camera
    uint32_t points = 0;       // sparse points the depth came from; 0 = none
};

// Orbit center for `reference`: on its optical axis at the median depth of
// the sparse points that project into the central 40% of the image (all
// points in the image when fewer than 16 are central). Without points it is
// the axis point nearest the world origin, or, when the origin is behind the
// camera, the axis point at the origin's distance. LoadColmapScene centers
// the world on the median of all points, which can sit far off the subject
// (or behind the camera for a partial walk-around), so the origin itself is
// not used.
OrbitPivot EstimateOrbitPivot(const Camera& reference,
                              const std::vector<SparsePoint>& points);

// Orbit view around `pivot`: yaw about `yaw_axis` (world down when gravity is
// known, otherwise the reference camera's down axis), then pitch about the
// reference camera's right axis, pan in the image plane and zoom as a scale
// of the distance to the pivot. The output keeps square pixels and the
// reference field of view in a render target capped at 960 pixels.
Camera OrbitCamera(const Camera& reference, const glm::vec3& yaw_axis,
                   const glm::vec3& pivot, float yaw, float pitch, float zoom,
                   float pan_x, float pan_y, uint32_t requested_width,
                   uint32_t requested_height);

// Down axis of a camera in world coordinates (the fallback yaw axis).
glm::vec3 CameraDownAxis(const Camera& camera);

// Pitch range of the viewers' vertical drag when gravity is unknown.
constexpr float kMaxOrbitPitch = 1.35f;

struct OrbitReference {
    Camera camera;              // initial view (levelled when gravity is known)
    glm::vec3 yaw_axis;         // pass to OrbitCamera
    glm::vec3 pivot{0.0f};      // orbit center
    float min_pitch = -kMaxOrbitPitch;  // radians; the UI clamps its pitch
    float max_pitch = kMaxOrbitPitch;   // to this range (always contains 0)
    bool gravity_aligned = false;
    std::string summary;  // one log line describing the choice
};

// OrbitCamera with the reference's axis and pivot, pitch clamped to its range.
Camera OrbitCamera(const OrbitReference& orbit, float yaw, float pitch,
                   float zoom, float pan_x, float pan_y,
                   uint32_t requested_width, uint32_t requested_height);

// The viewer set-up both apps run: gravity from `frame_priors_path` (may be
// empty or missing), estimated over `scene_cameras`, then LevelCamera on
// `reference` and world down as the yaw axis; otherwise `reference` as is
// with its own down axis. Either way the pivot comes from EstimateOrbitPivot
// over `points` (Scene::points). With gravity, the pitch range keeps the
// view between 10 degrees below the horizon and 85 degrees above it (looking
// down at the pivot). The summary records the estimated down vector in the
// scene (3DGS) frame and, through the inverse of `dataset_transform`
// (Scene::dataset_transform), in the COLMAP sparse-model frame.
OrbitReference MakeOrbitReference(
        const std::vector<Camera>& scene_cameras, const Camera& reference,
        const std::filesystem::path& frame_priors_path,
        const glm::mat4& dataset_transform = glm::mat4(1.0f),
        const std::vector<SparsePoint>& points = {});

}  // namespace cross3dgs
