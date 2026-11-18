#include "sfm_preview.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

#include <sqlite3.h>

#if defined(__ANDROID__) || defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#endif

#include "colmap/scene/reconstruction.h"
#include "colmap/sensor/bitmap.h"
#include "colmap/util/logging.h"

namespace cross3dgs_sfm {
namespace {

using Clock = std::chrono::steady_clock;
constexpr auto kFrameInterval = std::chrono::milliseconds(250);
constexpr uint8_t kBackground[3] = {14, 15, 18};
constexpr float kPi = 3.14159265358979f;
// Matched-pair view: lines drawn per pair, and input images kept scaled in
// memory. Exhaustive matching revisits the same images, so a small cache
// avoids most JPEG decodes.
constexpr int kMaxMatchLines = 100;
constexpr size_t kMaxThumbnails = 32;

struct Rgb {
  float r, g, b;
};

// ---- Minimal RGBA canvas ----

class Canvas {
 public:
  Canvas(std::vector<uint8_t>* pixels, int width, int height)
      : pixels_(pixels), width_(width), height_(height) {
    pixels_->assign(static_cast<size_t>(width) * height * 4, 255);
  }
  int width() const { return width_; }
  int height() const { return height_; }

  void Clear() {
    for (size_t i = 0; i < pixels_->size(); i += 4) {
      (*pixels_)[i] = kBackground[0];
      (*pixels_)[i + 1] = kBackground[1];
      (*pixels_)[i + 2] = kBackground[2];
    }
  }
  void Blend(int x, int y, Rgb c, float alpha) {
    if (x < 0 || y < 0 || x >= width_ || y >= height_) return;
    uint8_t* p = pixels_->data() + (static_cast<size_t>(y) * width_ + x) * 4;
    p[0] = static_cast<uint8_t>(p[0] + (c.r - p[0]) * alpha);
    p[1] = static_cast<uint8_t>(p[1] + (c.g - p[1]) * alpha);
    p[2] = static_cast<uint8_t>(p[2] + (c.b - p[2]) * alpha);
  }
  void Set(int x, int y, Rgb c) { Blend(x, y, c, 1.0f); }
  void FillRect(int x0, int y0, int x1, int y1, Rgb c, float alpha = 1.0f) {
    for (int y = std::max(y0, 0); y < std::min(y1, height_); ++y) {
      for (int x = std::max(x0, 0); x < std::min(x1, width_); ++x) {
        Blend(x, y, c, alpha);
      }
    }
  }
  void Line(float x0, float y0, float x1, float y1, Rgb c, float alpha) {
    const float steps = std::ceil(std::max(std::abs(x1 - x0), std::abs(y1 - y0)));
    if (!(steps < 4096.0f)) return;  // also rejects NaN
    const int n = std::max(1, static_cast<int>(steps));
    for (int i = 0; i <= n; ++i) {
      const float t = static_cast<float>(i) / n;
      Blend(static_cast<int>(std::lround(x0 + (x1 - x0) * t)),
            static_cast<int>(std::lround(y0 + (y1 - y0) * t)), c, alpha);
    }
  }
  void Circle(float cx, float cy, float radius, Rgb c, float alpha) {
    const int n = std::clamp(static_cast<int>(radius * 6.0f), 8, 48);
    float px = cx + radius, py = cy;
    for (int i = 1; i <= n; ++i) {
      const float a = 2.0f * kPi * i / n;
      const float x = cx + radius * std::cos(a), y = cy + radius * std::sin(a);
      Line(px, py, x, y, c, alpha);
      px = x;
      py = y;
    }
  }

 private:
  std::vector<uint8_t>* pixels_;
  int width_, height_;
};

// Perceptually ordered dark-to-bright ramp (inferno-like), t in [0, 1].
Rgb Inferno(float t) {
  static constexpr float kStops[6][3] = {{20, 11, 52},   {87, 16, 110},
                                         {158, 41, 99},  {219, 80, 59},
                                         {250, 152, 21}, {252, 255, 164}};
  t = std::clamp(t, 0.0f, 1.0f) * 5.0f;
  const int i = std::min(static_cast<int>(t), 4);
  const float f = t - i;
  return {kStops[i][0] + (kStops[i + 1][0] - kStops[i][0]) * f,
          kStops[i][1] + (kStops[i + 1][1] - kStops[i][1]) * f,
          kStops[i][2] + (kStops[i + 1][2] - kStops[i][2]) * f};
}

void ProgressBar(Canvas& canvas, double fraction) {
  const int h = std::max(4, canvas.height() / 90);
  const int y0 = canvas.height() - h;
  canvas.FillRect(0, y0, canvas.width(), canvas.height(), {40, 42, 48});
  const int filled = static_cast<int>(canvas.width() * std::clamp(fraction, 0.0, 1.0));
  canvas.FillRect(0, y0, filled, canvas.height(), {80, 200, 255});
}

std::string WithThousands(size_t value) {
  std::string digits = std::to_string(value);
  for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) {
    digits.insert(static_cast<size_t>(i), ",");
  }
  return digits;
}

bool IsImageFile(const std::filesystem::path& path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png";
}

}  // namespace

LivePreview::LivePreview(
    const PreviewTarget& target,
    std::filesystem::path image_path,
    std::filesystem::path database_path,
    std::unordered_map<std::string, Eigen::Vector3d> gravity_by_image)
    : target_(target),
      image_path_(std::move(image_path)),
      database_path_(std::move(database_path)),
      gravity_by_image_(std::move(gravity_by_image)),
      started_(Clock::now()) {
  std::error_code error;
  for (const auto& entry : std::filesystem::directory_iterator(image_path_, error)) {
    if (entry.is_regular_file(error) && IsImageFile(entry.path())) ++num_input_images_;
  }
  LOG(INFO) << "Live preview " << target_.width << "x" << target_.height
            << " (at most 4 frames/s, read-only)";
  thread_ = std::thread([this]() { Loop(); });
}

LivePreview::~LivePreview() {
  {
    std::lock_guard<std::mutex> lock(stop_mutex_);
    stop_ = true;
  }
  stop_cv_.notify_all();
  if (thread_.joinable()) thread_.join();
  if (database_ != nullptr) sqlite3_close(database_);
}

void LivePreview::SetStage(int stage) {
  stage_ = stage;
  stop_cv_.notify_all();  // render the new stage without waiting a tick
}

void LivePreview::SetExpectedPairs(size_t pairs) { expected_pairs_ = pairs; }

void LivePreview::OnReconstruction(const colmap::Reconstruction& reconstruction) {
  Model model;
  model.points.reserve(reconstruction.NumPoints3D());
  model.colors.reserve(reconstruction.NumPoints3D());
  for (const auto& [id, point] : reconstruction.Points3D()) {
    model.points.push_back(point.xyz.cast<float>());
    model.colors.push_back({point.color[0], point.color[1], point.color[2]});
  }
  Eigen::Vector3d down_sum = Eigen::Vector3d::Zero();
  int down_count = 0;
  std::unordered_set<std::string> registered;
  for (const colmap::image_t image_id : reconstruction.RegImageIds()) {
    const colmap::Image& image = reconstruction.Image(image_id);
    const colmap::Rigid3d cam_from_world = image.CamFromWorld();
    const Eigen::Matrix3d world_from_cam =
        cam_from_world.rotation().toRotationMatrix().transpose();
    Camera camera;
    camera.center = image.ProjectionCenter().cast<float>();
    camera.camera_to_world = world_from_cam.cast<float>();
    camera.name = image.Name();
    model.cameras.push_back(std::move(camera));
    registered.insert(image.Name());
    if (!registered_.count(image.Name())) model.newest.insert(image.Name());
    const auto gravity = gravity_by_image_.find(image.Name());
    if (gravity != gravity_by_image_.end()) {
      down_sum += world_from_cam * gravity->second.normalized();
      ++down_count;
    }
  }
  registered_ = std::move(registered);
  std::sort(model.cameras.begin(), model.cameras.end(),
            [](const Camera& a, const Camera& b) { return a.name < b.name; });
  // Gravity from the frame priors when they agree; otherwise the cameras'
  // mean image-down axis (phones are held roughly upright).
  if (down_count >= 2 && down_sum.norm() > 0.9 * down_count) {
    model.down = down_sum.normalized().cast<float>();
  } else if (!model.cameras.empty()) {
    Eigen::Vector3f sum = Eigen::Vector3f::Zero();
    for (const auto& camera : model.cameras) sum += camera.camera_to_world.col(1);
    if (sum.norm() > 1e-3f) model.down = sum.normalized();
  }
  auto shared = std::make_shared<const Model>(std::move(model));
  std::lock_guard<std::mutex> lock(model_mutex_);
  model_.swap(shared);
}

void LivePreview::Loop() {
  // Background priority: the preview only gets CPU that SfM leaves idle, so
  // when every core is busy it drops frames instead of slowing SfM down.
#if defined(__ANDROID__) || defined(__linux__)
  setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), 10);
#elif defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#endif
  auto next = Clock::now();
  int shown_stage = 0;
  while (true) {
    bool stopping = false;
    {
      std::unique_lock<std::mutex> lock(stop_mutex_);
      stop_cv_.wait_until(lock, next, [&]() {
        return stop_ || stage_.load() != shown_stage;
      });
      stopping = stop_;
    }
    const int stage = stage_.load();
    if (stopping) {
      if (stage == CROSS3DGS_SFM_STAGE_DONE && RenderMapping(true)) Emit(stage);
      return;
    }
    next = Clock::now() + kFrameInterval;
    shown_stage = stage;
    bool produced = false;
    try {
      switch (stage) {
        case CROSS3DGS_SFM_STAGE_FEATURES: produced = RenderFeatures(); break;
        case CROSS3DGS_SFM_STAGE_MATCHING: produced = RenderMatching(); break;
        case CROSS3DGS_SFM_STAGE_MAPPING: produced = RenderMapping(false); break;
        default: break;
      }
    } catch (const std::exception& error) {
      LOG(WARNING) << "Live preview frame skipped: " << error.what();
    }
    if (produced) Emit(stage);
  }
}

void LivePreview::Emit(int stage) {
  cross3dgs_sfm_preview frame;
  frame.rgba = canvas_.data();
  frame.width = target_.width;
  frame.height = target_.height;
  frame.stage = stage;
  frame.caption = caption_.c_str();
  target_.fn(&frame, target_.user_data);
}

bool LivePreview::OpenDatabase() {
  if (database_ != nullptr) return true;
  // Open only once COLMAP has switched the file to WAL mode. Between
  // creating the file and "PRAGMA journal_mode=WAL" the database is in
  // rollback mode, where a reader's shared lock makes that pragma fail with
  // "database is locked".
  // Header bytes 18 and 19 (write/read format version) are 2 in WAL mode.
  {
    std::ifstream header(database_path_, std::ios::binary);
    unsigned char bytes[20] = {};
    if (!header.read(reinterpret_cast<char*>(bytes), sizeof(bytes)) ||
        bytes[18] != 2 || bytes[19] != 2) {
      return false;
    }
  }
  // Read-only connection next to COLMAP's writer: the database is in WAL
  // mode, so readers never block it and never checkpoint.
  if (sqlite3_open_v2(database_path_.string().c_str(), &database_,
                      SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX,
                      nullptr) != SQLITE_OK) {
    sqlite3_close(database_);
    database_ = nullptr;
    return false;
  }
  sqlite3_busy_timeout(database_, 50);
  return true;
}

namespace {

// Runs `sql` and calls row(stmt) per result row; false if it could not run
// (e.g. the table does not exist yet).
template <typename RowFn>
bool Query(sqlite3* database, const char* sql, RowFn row) {
  sqlite3_stmt* statement = nullptr;
  if (sqlite3_prepare_v2(database, sql, -1, &statement, nullptr) != SQLITE_OK) {
    sqlite3_finalize(statement);
    return false;
  }
  int rc;
  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) row(statement);
  sqlite3_finalize(statement);
  return rc == SQLITE_DONE;
}

// Keypoint rows of one image: x, y, then scale or affine shape. Empty if the
// row is missing or malformed.
std::vector<float> ReadKeypoints(sqlite3* database, int64_t image_id, int* cols) {
  std::vector<float> keypoints;
  const std::string sql = "SELECT rows, cols, data FROM keypoints WHERE image_id = " +
                          std::to_string(image_id);
  Query(database, sql.c_str(), [&](sqlite3_stmt* s) {
    const int rows = sqlite3_column_int(s, 0);
    *cols = sqlite3_column_int(s, 1);
    const int bytes = sqlite3_column_bytes(s, 2);
    if (rows > 0 && *cols >= 2 &&
        bytes == rows * *cols * static_cast<int>(sizeof(float))) {
      keypoints.resize(static_cast<size_t>(rows) * *cols);
      std::memcpy(keypoints.data(), sqlite3_column_blob(s, 2), bytes);
    }
  });
  return keypoints;
}

// Inlier matches of one pair as index pairs into the keypoints of the lower
// and the higher image id, the order COLMAP stores them in.
std::vector<uint32_t> ReadInlierMatches(sqlite3* database, int64_t pair_id) {
  std::vector<uint32_t> matches;
  const std::string sql = "SELECT rows, cols, data FROM two_view_geometries "
                          "WHERE pair_id = " + std::to_string(pair_id);
  Query(database, sql.c_str(), [&](sqlite3_stmt* s) {
    const int rows = sqlite3_column_int(s, 0);
    const int bytes = sqlite3_column_bytes(s, 2);
    if (rows > 0 && sqlite3_column_int(s, 1) == 2 &&
        bytes == rows * 2 * static_cast<int>(sizeof(uint32_t))) {
      matches.resize(static_cast<size_t>(rows) * 2);
      std::memcpy(matches.data(), sqlite3_column_blob(s, 2), bytes);
    }
  });
  return matches;
}

std::string Stem(const std::string& name) {
  return std::filesystem::path(name).stem().string();
}

}  // namespace

std::shared_ptr<const LivePreview::Thumbnail> LivePreview::LoadThumbnail(
    const std::string& name, int max_width, int max_height) {
  const auto cached = thumbnails_.find(name);
  if (cached != thumbnails_.end()) return cached->second;
  colmap::Bitmap bitmap;
  if (!bitmap.Read(image_path_ / name, /*as_rgb=*/true)) return nullptr;
  auto thumbnail = std::make_shared<Thumbnail>();
  thumbnail->scale = std::min(static_cast<float>(max_width) / bitmap.Width(),
                              static_cast<float>(max_height) / bitmap.Height());
  thumbnail->width =
      std::max(1, static_cast<int>(std::lround(bitmap.Width() * thumbnail->scale)));
  thumbnail->height =
      std::max(1, static_cast<int>(std::lround(bitmap.Height() * thumbnail->scale)));
  bitmap.Rescale(thumbnail->width, thumbnail->height);
  thumbnail->rgb = bitmap.RowMajorData();
  if (thumbnails_.size() >= kMaxThumbnails) thumbnails_.clear();
  thumbnails_.emplace(name, thumbnail);
  return thumbnail;
}

bool LivePreview::RenderFeatures() {
  if (!OpenDatabase()) return false;
  int64_t image_id = -1;
  std::string name;
  int rows = 0, cols = 0;
  std::vector<float> keypoints;
  if (!Query(database_,
             "SELECT keypoints.image_id, images.name, keypoints.rows, "
             "keypoints.cols, keypoints.data FROM keypoints JOIN images ON "
             "images.image_id = keypoints.image_id "
             "ORDER BY keypoints.image_id DESC LIMIT 1",
             [&](sqlite3_stmt* s) {
               image_id = sqlite3_column_int64(s, 0);
               name = reinterpret_cast<const char*>(sqlite3_column_text(s, 1));
               rows = sqlite3_column_int(s, 2);
               cols = sqlite3_column_int(s, 3);
               const int bytes = sqlite3_column_bytes(s, 4);
               if (rows > 0 && cols > 0 &&
                   bytes == rows * cols * static_cast<int>(sizeof(float))) {
                 keypoints.resize(static_cast<size_t>(rows) * cols);
                 std::memcpy(keypoints.data(), sqlite3_column_blob(s, 4), bytes);
               }
             }) ||
      image_id < 0 || image_id == last_keypoint_image_) {
    return false;
  }
  int64_t done = 0;
  Query(database_, "SELECT COUNT(*) FROM keypoints",
        [&](sqlite3_stmt* s) { done = sqlite3_column_int64(s, 0); });

  colmap::Bitmap bitmap;
  if (!bitmap.Read(image_path_ / name, /*as_rgb=*/true)) return false;
  last_keypoint_image_ = image_id;
  const int source_width = bitmap.Width(), source_height = bitmap.Height();
  Canvas canvas(&canvas_, target_.width, target_.height);
  canvas.Clear();
  const float scale = std::min(static_cast<float>(target_.width) / source_width,
                               static_cast<float>(target_.height) / source_height);
  const int fit_width = std::max(1, static_cast<int>(std::lround(source_width * scale)));
  const int fit_height = std::max(1, static_cast<int>(std::lround(source_height * scale)));
  bitmap.Rescale(fit_width, fit_height);
  const int x0 = (target_.width - fit_width) / 2, y0 = (target_.height - fit_height) / 2;
  const std::vector<uint8_t>& rgb = bitmap.RowMajorData();
  for (int y = 0; y < fit_height; ++y) {
    for (int x = 0; x < fit_width; ++x) {
      const uint8_t* p = rgb.data() + (static_cast<size_t>(y) * fit_width + x) * 3;
      // Dimmed so the keypoints stand out.
      canvas.Set(x0 + x, y0 + y, {p[0] * 0.45f, p[1] * 0.45f, p[2] * 0.45f});
    }
  }
  // Keypoint rows are x, y and either scale/orientation (4 columns) or an
  // affine shape a11 a12 a21 a22 (6 columns); the circle shows the scale.
  for (int i = 0; i < rows; ++i) {
    const float* k = keypoints.data() + static_cast<size_t>(i) * cols;
    float feature_scale = 1.0f;
    if (cols >= 6) {
      feature_scale = std::sqrt(std::abs(k[2] * k[5] - k[3] * k[4]));
    } else if (cols >= 3) {
      feature_scale = k[2];
    }
    const float t = std::clamp(std::log2(std::max(feature_scale, 1.0f)) / 5.0f, 0.0f, 1.0f);
    const Rgb color = {80 + 175 * t, 220 - 10 * t, 255 - 195 * t};
    canvas.Circle(x0 + k[0] * scale, y0 + k[1] * scale,
                  std::clamp(feature_scale * scale * 2.0f, 1.5f, 14.0f), color, 0.75f);
  }
  const size_t total = std::max<size_t>(num_input_images_, static_cast<size_t>(done));
  ProgressBar(canvas, total > 0 ? static_cast<double>(done) / total : 0.0);
  // Two short lines (about 60 characters fit a phone's portrait width):
  // progress, then how to read the picture.
  caption_ = "SIFT " + std::to_string(done) + "/" + std::to_string(total) + " \xC2\xB7 " +
             name + " \xC2\xB7 " + WithThousands(static_cast<size_t>(rows)) +
             " keypoints\n"
             "Circle color/size = scale (cyan fine \xE2\x86\x92 yellow coarse)";
  return true;
}

bool LivePreview::RenderMatching() {
  if (!OpenDatabase()) return false;
  int64_t pair_count = -1;
  if (!Query(database_, "SELECT COUNT(*) FROM two_view_geometries",
             [&](sqlite3_stmt* s) { pair_count = sqlite3_column_int64(s, 0); }) ||
      (pair_count == last_pair_count_ && pending_pairs_.empty())) {
    return false;
  }
  // Images in capture (name) order define the matrix axes.
  std::unordered_map<int64_t, int> index_of;
  std::unordered_map<int64_t, std::string> name_of;
  Query(database_, "SELECT image_id, name FROM images ORDER BY name", [&](sqlite3_stmt* s) {
    const int64_t id = sqlite3_column_int64(s, 0);
    index_of.emplace(id, static_cast<int>(index_of.size()));
    name_of.emplace(id, reinterpret_cast<const char*>(sqlite3_column_text(s, 1)));
  });
  const int n = static_cast<int>(index_of.size());
  if (n < 2) return false;
  std::vector<int> inliers(static_cast<size_t>(n) * n, -1);  // -1: not verified yet
  size_t verified = 0;
  int max_inliers = 1;
  std::vector<int> geometry_inliers;
  std::vector<MatchedPair> fresh_pairs;
  constexpr int64_t kMaxImageId = 2147483647;
  Query(database_, "SELECT pair_id, rows FROM two_view_geometries", [&](sqlite3_stmt* s) {
    const int64_t pair_id = sqlite3_column_int64(s, 0);
    const int64_t id2 = pair_id % kMaxImageId, id1 = (pair_id - id2) / kMaxImageId;
    const auto a = index_of.find(id1), b = index_of.find(id2);
    if (a == index_of.end() || b == index_of.end()) return;
    const int count = sqlite3_column_int(s, 1);
    inliers[static_cast<size_t>(a->second) * n + b->second] = count;
    inliers[static_cast<size_t>(b->second) * n + a->second] = count;
    ++verified;
    if (count > 0) {
      geometry_inliers.push_back(count);
      max_inliers = std::max(max_inliers, count);
      if (seen_pairs_.insert(pair_id).second) fresh_pairs.push_back({pair_id, count});
    }
  });
  last_pair_count_ = pair_count;
  if (!fresh_pairs.empty()) {
    std::sort(fresh_pairs.begin(), fresh_pairs.end(),
              [](const MatchedPair& a, const MatchedPair& b) { return a.inliers < b.inliers; });
    pending_pairs_ = std::move(fresh_pairs);
  }
  if (!pending_pairs_.empty()) {
    shown_pair_ = pending_pairs_.back();
    pending_pairs_.pop_back();
  }

  Canvas canvas(&canvas_, target_.width, target_.height);
  canvas.Clear();
  // The matrix sits on the left; the matched pair takes the rest of the width.
  const int margin = std::max(8, target_.width / 60);
  const int content_height = target_.height * 92 / 100;
  const int side = std::min(content_height, target_.width * 52 / 100);
  const int x0 = margin, y0 = (target_.height - side) / 2;
  const float log_max = std::log1p(static_cast<float>(max_inliers));
  for (int py = 0; py < side; ++py) {
    const int i = py * n / side;
    for (int px = 0; px < side; ++px) {
      const int j = px * n / side;
      Rgb color;
      if (i == j) {
        color = {70, 72, 80};
      } else {
        const int count = inliers[static_cast<size_t>(i) * n + j];
        if (count < 0) {
          color = {28, 30, 36};      // not verified yet
        } else if (count == 0) {
          color = {60, 22, 26};      // verified, no geometry
        } else {
          color = Inferno(0.15f + 0.85f * std::log1p(static_cast<float>(count)) / log_max);
        }
      }
      canvas.Set(x0 + px, y0 + py, color);
    }
  }
  // Color bar right of the matrix: inlier count on the same log scale,
  // 1 inlier at the bottom to the current maximum (in the caption) on top.
  const int bar_x0 = x0 + side + std::max(8, side / 30);
  const int bar_x1 = bar_x0 + std::max(6, side / 40);
  for (int py = 0; py < side; ++py) {
    const float t = 1.0f - static_cast<float>(py) / std::max(1, side - 1);
    canvas.FillRect(bar_x0, y0 + py, bar_x1, y0 + py + 1, Inferno(0.15f + 0.85f * t));
  }

  std::string pair_line;
  if (shown_pair_.pair_id >= 0) {
    const int64_t stored2 = shown_pair_.pair_id % kMaxImageId;
    const int64_t stored1 = (shown_pair_.pair_id - stored2) / kMaxImageId;
    // Pair ids order images by id, which need not be capture order; the
    // earlier frame goes on top and first in the caption.
    const auto order1 = index_of.find(stored1), order2 = index_of.find(stored2);
    const bool swapped = order1 != index_of.end() && order2 != index_of.end() &&
                         order1->second > order2->second;
    const int64_t id1 = swapped ? stored2 : stored1;
    const int64_t id2 = swapped ? stored1 : stored2;
    const auto name1 = name_of.find(id1), name2 = name_of.find(id2);
    const int panel_x0 = bar_x1 + margin;
    const int panel_width = target_.width - panel_x0 - margin;
    const int gap = std::max(6, target_.height / 60);
    const int slot_height = (content_height - gap) / 2;
    if (name1 != name_of.end() && name2 != name_of.end() && panel_width > 0 &&
        slot_height > 0) {
      const auto top = LoadThumbnail(name1->second, panel_width, slot_height);
      const auto bottom = LoadThumbnail(name2->second, panel_width, slot_height);
      if (top != nullptr && bottom != nullptr) {
        // The two images meet at the middle so the lines between them stay short.
        const int top_x = panel_x0 + (panel_width - top->width) / 2;
        const int top_y = (target_.height - gap) / 2 - top->height;
        const int bottom_x = panel_x0 + (panel_width - bottom->width) / 2;
        const int bottom_y = (target_.height + gap) / 2;
        const auto blit = [&](const Thumbnail& image, int left, int upper) {
          for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
              const uint8_t* p =
                  image.rgb.data() + (static_cast<size_t>(y) * image.width + x) * 3;
              // Dimmed as in the SIFT view, so the matches stand out.
              canvas.Set(left + x, upper + y, {p[0] * 0.45f, p[1] * 0.45f, p[2] * 0.45f});
            }
          }
        };
        blit(*top, top_x, top_y);
        blit(*bottom, bottom_x, bottom_y);

        int cols1 = 0, cols2 = 0;
        const std::vector<float> keypoints1 = ReadKeypoints(database_, id1, &cols1);
        const std::vector<float> keypoints2 = ReadKeypoints(database_, id2, &cols2);
        const std::vector<uint32_t> matches =
            ReadInlierMatches(database_, shown_pair_.pair_id);
        const size_t match_count = matches.size() / 2;
        const size_t step = (match_count + kMaxMatchLines - 1) / kMaxMatchLines;
        for (size_t m = 0; step > 0 && m < match_count; m += step) {
          const size_t k1 = static_cast<size_t>(matches[m * 2 + (swapped ? 1 : 0)]) * cols1;
          const size_t k2 = static_cast<size_t>(matches[m * 2 + (swapped ? 0 : 1)]) * cols2;
          if (k1 + 1 >= keypoints1.size() || k2 + 1 >= keypoints2.size()) continue;
          const float ax = top_x + keypoints1[k1] * top->scale;
          const float ay = top_y + keypoints1[k1 + 1] * top->scale;
          const float bx = bottom_x + keypoints2[k2] * bottom->scale;
          const float by = bottom_y + keypoints2[k2 + 1] * bottom->scale;
          const Rgb color = Inferno(0.4f + 0.6f * static_cast<float>(m) / match_count);
          canvas.Line(ax, ay, bx, by, color, 0.7f);
          canvas.Circle(ax, ay, 2.5f, color, 1.0f);
          canvas.Circle(bx, by, 2.5f, color, 1.0f);
        }

        // Mark the pair in the matrix, on both sides of the diagonal.
        const auto mark = [&](int row, int col) {
          const float left = static_cast<float>(x0 + col * side / n) - 1.0f;
          const float right = static_cast<float>(x0 + (col + 1) * side / n) + 1.0f;
          const float upper = static_cast<float>(y0 + row * side / n) - 1.0f;
          const float lower = static_cast<float>(y0 + (row + 1) * side / n) + 1.0f;
          const Rgb white = {255, 255, 255};
          canvas.Line(left, upper, right, upper, white, 1.0f);
          canvas.Line(right, upper, right, lower, white, 1.0f);
          canvas.Line(right, lower, left, lower, white, 1.0f);
          canvas.Line(left, lower, left, upper, white, 1.0f);
        };
        const int row = index_of.at(id1), col = index_of.at(id2);
        mark(row, col);
        mark(col, row);
        pair_line = Stem(name1->second) + " \xE2\x86\x94 " + Stem(name2->second) +
                    " \xC2\xB7 " + WithThousands(static_cast<size_t>(shown_pair_.inliers)) +
                    " inliers \xC2\xB7 dark red = none";
      }
    }
  }

  const size_t expected = expected_pairs_.load();
  ProgressBar(canvas, expected > 0 ? static_cast<double>(verified) / expected : 0.0);
  std::ostringstream caption;
  caption << "Matching " << WithThousands(verified);
  if (expected > 0) caption << "/" << WithThousands(expected);
  caption << " pairs";
  if (!geometry_inliers.empty()) {
    auto middle = geometry_inliers.begin() + geometry_inliers.size() / 2;
    std::nth_element(geometry_inliers.begin(), middle, geometry_inliers.end());
    caption << " \xC2\xB7 inliers median " << *middle << " / max "
            << WithThousands(static_cast<size_t>(max_inliers));
  }
  caption << "\n"
          << (pair_line.empty()
                  ? "Color = match quality (inliers/pair, log; dark red = none)"
                  : pair_line);
  caption_ = caption.str();
  return true;
}

bool LivePreview::RenderMapping(bool final_frame) {
  std::shared_ptr<const Model> snapshot;
  {
    std::lock_guard<std::mutex> lock(model_mutex_);
    snapshot = model_;
  }
  if (!snapshot || (snapshot->points.empty() && snapshot->cameras.empty())) return false;
  const Model& model = *snapshot;
  // Robust center and extent: per-axis median of the points; radius covering
  // most points and the cameras.
  Eigen::Vector3f center = Eigen::Vector3f::Zero();
  const std::vector<Eigen::Vector3f>& source = model.points;
  if (!source.empty()) {
    std::vector<float> axis(source.size());
    for (int d = 0; d < 3; ++d) {
      for (size_t i = 0; i < source.size(); ++i) axis[i] = source[i][d];
      auto middle = axis.begin() + axis.size() / 2;
      std::nth_element(axis.begin(), middle, axis.end());
      center[d] = *middle;
    }
  } else {
    for (const auto& camera : model.cameras) center += camera.center;
    center /= static_cast<float>(model.cameras.size());
  }
  auto percentile = [](std::vector<float> values, float q) {
    if (values.empty()) return 0.0f;
    auto at = values.begin() + static_cast<long>((values.size() - 1) * q);
    std::nth_element(values.begin(), at, values.end());
    return *at;
  };
  std::vector<float> point_distances, camera_distances;
  for (const auto& p : model.points) point_distances.push_back((p - center).norm());
  for (const auto& c : model.cameras) camera_distances.push_back((c.center - center).norm());
  const float radius = std::max(
      {percentile(point_distances, 0.8f), percentile(camera_distances, 0.9f), 1e-3f}) * 1.1f;

  Eigen::Vector3f down = model.down;
  if (down.norm() < 0.5f) down = Eigen::Vector3f(0, 1, 0);
  Eigen::Vector3f e1 = down.unitOrthogonal();
  Eigen::Vector3f e2 = down.cross(e1);
  // Turntable: about 23 degrees per second, 30 degrees above the horizon.
  const float seconds = std::chrono::duration<float>(Clock::now() - started_).count();
  const float azimuth = seconds * 0.4f;
  const float elevation = 30.0f * kPi / 180.0f;
  const float half_fov = 27.5f * kPi / 180.0f;
  const float distance = radius / std::sin(half_fov);
  const Eigen::Vector3f horizontal = std::cos(azimuth) * e1 + std::sin(azimuth) * e2;
  const Eigen::Vector3f eye =
      center + distance * (std::cos(elevation) * horizontal - std::sin(elevation) * down);
  const Eigen::Vector3f forward = (center - eye).normalized();
  const Eigen::Vector3f right = down.cross(forward).normalized();
  const Eigen::Vector3f image_down = forward.cross(right);
  const float focal = 0.5f * target_.height / std::tan(half_fov);
  const float cx = 0.5f * target_.width, cy = 0.5f * target_.height;
  auto project = [&](const Eigen::Vector3f& x, float* u, float* v, float* z) {
    const Eigen::Vector3f d = x - eye;
    *z = d.dot(forward);
    if (*z < 1e-3f * distance) return false;
    *u = focal * d.dot(right) / *z + cx;
    *v = focal * d.dot(image_down) / *z + cy;
    return true;
  };

  Canvas canvas(&canvas_, target_.width, target_.height);
  canvas.Clear();
  std::vector<float> depth(static_cast<size_t>(target_.width) * target_.height,
                           std::numeric_limits<float>::infinity());
  const int splat = model.points.size() < 5000 ? 3 : 2;
  for (size_t i = 0; i < model.points.size(); ++i) {
    float u, v, z;
    if (!project(model.points[i], &u, &v, &z)) continue;
    const int x0 = static_cast<int>(u) - splat / 2, y0 = static_cast<int>(v) - splat / 2;
    const auto& c = model.colors[i];
    for (int y = y0; y < y0 + splat; ++y) {
      for (int x = x0; x < x0 + splat; ++x) {
        if (x < 0 || y < 0 || x >= target_.width || y >= target_.height) continue;
        float& d = depth[static_cast<size_t>(y) * target_.width + x];
        if (z >= d) continue;
        d = z;
        canvas.Set(x, y, {std::min(255.0f, c[0] * 1.1f), std::min(255.0f, c[1] * 1.1f),
                          std::min(255.0f, c[2] * 1.1f)});
      }
    }
  }
  // Capture trajectory, then camera frusta (newest highlighted).
  for (size_t i = 1; i < model.cameras.size(); ++i) {
    float u0, v0, z0, u1, v1, z1;
    if (project(model.cameras[i - 1].center, &u0, &v0, &z0) &&
        project(model.cameras[i].center, &u1, &v1, &z1)) {
      canvas.Line(u0, v0, u1, v1, {110, 115, 135}, 0.6f);
    }
  }
  const float size = radius * 0.06f;
  for (const auto& camera : model.cameras) {
    const bool newest = model.newest.count(camera.name) > 0 && !final_frame;
    const Rgb color = newest ? Rgb{255, 140, 40} : Rgb{205, 210, 225};
    const Eigen::Vector3f& c = camera.center;
    const Eigen::Vector3f r = camera.camera_to_world.col(0) * (0.66f * size);
    const Eigen::Vector3f d = camera.camera_to_world.col(1) * (0.5f * size);
    const Eigen::Vector3f f = camera.camera_to_world.col(2) * size;
    const Eigen::Vector3f corners[4] = {c + f - r - d, c + f + r - d, c + f + r + d,
                                        c + f - r + d};
    float uc, vc, zc;
    if (!project(c, &uc, &vc, &zc)) continue;
    float u[4], v[4], z;
    bool visible = true;
    for (int k = 0; k < 4; ++k) visible = visible && project(corners[k], &u[k], &v[k], &z);
    if (!visible) continue;
    const float alpha = newest ? 1.0f : 0.7f;
    for (int k = 0; k < 4; ++k) {
      canvas.Line(uc, vc, u[k], v[k], color, alpha);
      canvas.Line(u[k], v[k], u[(k + 1) % 4], v[(k + 1) % 4], color, alpha);
    }
  }
  const size_t total = std::max(num_input_images_, model.cameras.size());
  ProgressBar(canvas, total > 0 ? static_cast<double>(model.cameras.size()) / total : 0.0);
  caption_ = std::string(final_frame ? "Sparse model " : "Incremental SfM ") +
             std::to_string(model.cameras.size()) + "/" + std::to_string(total) +
             " images \xC2\xB7 " + WithThousands(model.points.size()) + " points\n" +
             (final_frame ? "Frusta = cameras \xC2\xB7 up = gravity"
                          : "Frusta = cameras (orange = newest) \xC2\xB7 up = gravity");
  return true;
}

}  // namespace cross3dgs_sfm
