// Live SfM preview renderer (see cross3dgs_sfm_set_preview). Private to the
// library; cross3dgs_sfm.cc owns one instance per run.
#ifndef CROSS3DGS_SFM_PREVIEW_H
#define CROSS3DGS_SFM_PREVIEW_H

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

#include "cross3dgs_sfm.h"

struct sqlite3;

namespace colmap {
class Reconstruction;
}

namespace cross3dgs_sfm {

struct PreviewTarget {
  cross3dgs_sfm_preview_fn fn = nullptr;
  void* user_data = nullptr;
  int width = 0;
  int height = 0;
};

class LivePreview {
 public:
  // gravity_by_image: per-image "down" in the COLMAP camera frame (frame
  // priors); used to keep the mapping turntable upright.
  LivePreview(const PreviewTarget& target,
              std::filesystem::path image_path,
              std::filesystem::path database_path,
              std::unordered_map<std::string, Eigen::Vector3d> gravity_by_image);
  // Stops the thread; when the stage is DONE, one final frame is emitted
  // first. No callback happens after the destructor returns.
  ~LivePreview();

  LivePreview(const LivePreview&) = delete;
  LivePreview& operator=(const LivePreview&) = delete;

  void SetStage(int stage);
  // Total pairs the matcher will verify (0 if unknown, e.g. sequential).
  void SetExpectedPairs(size_t pairs);
  // Copies what the turntable needs; call from the mapping thread.
  void OnReconstruction(const colmap::Reconstruction& reconstruction);

 private:
  struct Camera {
    Eigen::Vector3f center;
    Eigen::Matrix3f camera_to_world;  // columns: right, down, forward
    std::string name;
  };
  struct Model {
    std::vector<Eigen::Vector3f> points;
    std::vector<std::array<uint8_t, 3>> colors;
    std::vector<Camera> cameras;  // sorted by image name (capture order)
    std::unordered_set<std::string> newest;
    Eigen::Vector3f down = Eigen::Vector3f::Zero();  // zero: unknown
  };
  // One verified image pair: COLMAP's pair id and its inlier count.
  struct MatchedPair {
    int64_t pair_id = -1;
    int inliers = 0;
  };
  // An input image scaled to fit one slot of the matched-pair view.
  struct Thumbnail {
    int width = 0;
    int height = 0;
    float scale = 1.0f;  // thumbnail pixels per source pixel
    std::vector<uint8_t> rgb;
  };

  void Loop();
  bool OpenDatabase();
  bool RenderFeatures();
  bool RenderMatching();
  bool RenderMapping(bool final_frame);
  std::shared_ptr<const Thumbnail> LoadThumbnail(const std::string& name,
                                                 int max_width, int max_height);
  void Emit(int stage);

  const PreviewTarget target_;
  const std::filesystem::path image_path_;
  const std::filesystem::path database_path_;
  const std::unordered_map<std::string, Eigen::Vector3d> gravity_by_image_;
  size_t num_input_images_ = 0;

  std::atomic<int> stage_{CROSS3DGS_SFM_STAGE_FEATURES};
  std::atomic<size_t> expected_pairs_{0};

  // Swapped, never copied, under the lock: the mapping thread must not wait
  // on the (low-priority) preview thread.
  std::mutex model_mutex_;
  std::shared_ptr<const Model> model_;
  std::unordered_set<std::string> registered_;  // mapping thread only

  // Preview thread only.
  sqlite3* database_ = nullptr;
  int64_t last_keypoint_image_ = -1;
  int64_t last_pair_count_ = -1;
  std::unordered_set<int64_t> seen_pairs_;
  // Pairs of the newest database batch still to show, best last. A batch can
  // land at once, so one pair per frame keeps the view moving between them.
  std::vector<MatchedPair> pending_pairs_;
  MatchedPair shown_pair_;
  std::unordered_map<std::string, std::shared_ptr<const Thumbnail>> thumbnails_;
  std::vector<uint8_t> canvas_;
  std::string caption_;
  std::chrono::steady_clock::time_point started_;

  std::mutex stop_mutex_;
  std::condition_variable stop_cv_;
  bool stop_ = false;
  std::thread thread_;
};

}  // namespace cross3dgs_sfm

#endif
