// CPU COLMAP sparse SfM shared by the Android and iOS capture-to-view apps.
//
// Both platforms run this one recorded SfM configuration: faiss-index SIFT
// matching, single shared camera, 8192 features at max image size 2400,
// exhaustive pairing up to 100 images and sequential (overlap 15) above,
// relaxed incremental init, and position priors when frame_priors.txt is
// present. The stages are composed manually because
// AutomaticReconstructionController's SIFT_BRUTEFORCE matcher allocates
// multi-GB distance matrices. Logging goes through a C log callback. An
// optional camera-model override serves image-folder inputs that carry no
// intrinsics prior, and on iOS only the extraction/matching thread counts are
// memory-bounded (see MemoryBoundedThreads).

#include "cross3dgs_sfm.h"
#include "sfm_preview.h"

#if defined(__ANDROID__) || defined(__linux__)
#include <sched.h>
#endif

#if defined(__ANDROID__)
#include <cblas.h>
#endif

// os_proc_available_memory() exists on iOS only (not macOS), so the memory
// bound keys on TARGET_OS_IOS rather than __APPLE__.
#if defined(__APPLE__)
#include <TargetConditionals.h>
#if TARGET_OS_IOS
#include <os/proc.h>
#define CROSS3DGS_SFM_MEMORY_BOUNDED_THREADS 1
#endif
#endif

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

#include "colmap/controllers/feature_extraction.h"
#include "colmap/controllers/feature_matching.h"
#include "colmap/controllers/global_pipeline.h"
#include "colmap/controllers/image_reader.h"
#include "colmap/controllers/incremental_pipeline.h"
#include "colmap/controllers/pairing.h"
#include "colmap/estimators/alignment.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/feature/extractor.h"
#include "colmap/feature/matcher.h"
#include "colmap/feature/sift.h"
#include "colmap/geometry/pose_prior.h"
#include "colmap/optim/ransac.h"
#include "colmap/scene/database.h"
#include "colmap/scene/reconstruction_manager.h"
#include "colmap/util/logging.h"

#include <glog/logging.h>

namespace {

namespace fs = std::filesystem;

std::mutex g_log_mutex;
cross3dgs_sfm_log_fn g_log = nullptr;
void* g_log_user_data = nullptr;

// cross3dgs_sfm_set_threads overrides (0 = default).
std::atomic<int> g_extraction_threads{0};
std::atomic<int> g_matching_threads{0};
std::atomic<int> g_mapping_threads{0};
std::atomic<unsigned long long> g_cpu_mask{0};
std::atomic<int> g_seed{-1};
std::atomic<int> g_actual_extraction{0}, g_actual_matching{0}, g_actual_mapping{0};

// Restricts the calling thread (and so every thread it starts) to
// g_cpu_mask for its lifetime; restores the previous affinity after.
class ScopedCpuMask {
 public:
  ScopedCpuMask() {
#if defined(__ANDROID__) || defined(__linux__)
    const unsigned long long mask = g_cpu_mask.load();
    if (mask == 0) return;
    if (sched_getaffinity(0, sizeof(previous_), &previous_) != 0) return;
    cpu_set_t wanted;
    CPU_ZERO(&wanted);
    for (int cpu = 0; cpu < 64; ++cpu) {
      if (mask & (1ull << cpu)) CPU_SET(cpu, &wanted);
    }
    applied_ = sched_setaffinity(0, sizeof(wanted), &wanted) == 0;
    LOG(INFO) << "CPU mask 0x" << std::hex << mask << std::dec
              << (applied_ ? " applied" : " rejected");
#endif
  }
  ~ScopedCpuMask() {
#if defined(__ANDROID__) || defined(__linux__)
    if (applied_) sched_setaffinity(0, sizeof(previous_), &previous_);
#endif
  }

 private:
#if defined(__ANDROID__) || defined(__linux__)
  cpu_set_t previous_{};
#endif
  bool applied_ = false;
};

// Applies a thread-count override; never above a positive default (the iOS
// memory bound).
void OverrideThreads(int requested, int* threads) {
  if (requested > 0) {
    *threads = *threads > 0 ? std::min(*threads, requested) : requested;
  }
  // COLMAP's default (-1) is one thread per core of the machine; with a CPU
  // mask that would stack several workers on each allowed core.
  const int mask_cpus = __builtin_popcountll(g_cpu_mask.load());
  if (mask_cpus > 0 && *threads <= 0) *threads = mask_cpus;
}

std::mutex g_preview_mutex;
cross3dgs_sfm::PreviewTarget g_preview;

std::mutex g_cancel_mutex;
colmap::Thread* g_active_thread = nullptr;
std::atomic<bool> g_cancel_requested{false};

class CallbackSink : public google::LogSink {
 public:
  void send(google::LogSeverity /*severity*/,
            const char* /*full_filename*/,
            const char* /*base_filename*/,
            int /*line*/,
            const google::LogMessageTime& /*time*/,
            const char* message,
            size_t message_len) override {
    const std::string text(message, message_len);
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (g_log != nullptr) g_log(text.c_str(), g_log_user_data);
  }
};

void EnsureLoggingInitialized() {
  static std::once_flag once;
  std::call_once(once, []() {
    google::InitGoogleLogging("cross3dgs_sfm");
    // No glog files; stderr is the Xcode console on iOS (discarded on
    // Android) and the sink feeds the app log (and logcat via JNI).
    FLAGS_logtostderr = true;
    google::AddLogSink(new CallbackSink());
  });
}

// ---- Frame priors (from the AR capture) ----
//
// File format (written by frame extraction next to the extracted frames):
//   # intrinsics fx fy cx cy width height     (PINHOLE, frame resolution)
//   <image_name> tx ty tz gx gy gz sigma
// Positions are metric camera centers in the capture world frame; gravity is
// the "down" direction in the COLMAP camera frame; sigma is the position
// std-dev in meters.

struct FramePrior {
  Eigen::Vector3d position;
  Eigen::Vector3d gravity;
  double sigma = 0.05;
  bool has_gravity = false;
};

struct PriorIntrinsics {
  double fx = 0, fy = 0, cx = 0, cy = 0;
  bool valid = false;
};

bool ParsePriorsFile(const std::string& path,
                     std::unordered_map<std::string, FramePrior>* priors,
                     PriorIntrinsics* intrinsics) {
  std::ifstream file(path);
  if (!file.is_open()) return false;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    if (line[0] == '#') {
      std::istringstream ss(line.substr(1));
      std::string key;
      ss >> key;
      if (key == "intrinsics") {
        int w = 0, h = 0;
        if (ss >> intrinsics->fx >> intrinsics->fy >> intrinsics->cx >>
            intrinsics->cy >> w >> h) {
          intrinsics->valid = true;
        }
      }
      continue;
    }
    std::istringstream ss(line);
    std::string name;
    FramePrior prior;
    double gx, gy, gz;
    if (!(ss >> name >> prior.position.x() >> prior.position.y() >>
          prior.position.z() >> gx >> gy >> gz >> prior.sigma)) {
      continue;
    }
    prior.gravity = Eigen::Vector3d(gx, gy, gz);
    prior.has_gravity = prior.gravity.norm() > 0.5;
    (*priors)[name] = prior;
  }
  return !priors->empty();
}

size_t WritePriorsToDatabase(
    const fs::path& database_path,
    const std::unordered_map<std::string, FramePrior>& priors) {
  auto database = colmap::Database::Open(database_path);
  size_t written = 0;
  for (const colmap::Image& image : database->ReadAllImages()) {
    const auto it = priors.find(image.Name());
    if (it == priors.end()) continue;
    colmap::PosePrior prior;
    prior.corr_data_id = image.DataId();
    prior.position = it->second.position;
    prior.position_covariance =
        Eigen::Matrix3d::Identity() * it->second.sigma * it->second.sigma;
    prior.coordinate_system = colmap::PosePrior::CoordinateSystem::CARTESIAN;
    if (it->second.has_gravity) {
      prior.gravity = it->second.gravity;
    }
    database->WritePosePrior(prior);
    ++written;
  }
  return written;
}

#ifdef CROSS3DGS_SFM_MEMORY_BOUNDED_THREADS
// iOS kills a process that crosses its memory limit (jetsam), and COLMAP's
// CPU SIFT upsamples the first octave, so one 1920x1440 capture frame needs
// a scale space of roughly 0.7 GB. One extractor per core (6 on an A14)
// therefore exceeds the limit on 6 GB phones. Per-image features do not
// depend on the thread count, so it is bounded by the memory the process may
// still allocate instead of by the core count. (Matching and geometric
// verification vary run to run on every platform regardless of the thread
// count.) Android keeps COLMAP's
// default (one thread per core).
int MemoryBoundedThreads(size_t bytes_per_thread, const char* stage) {
  constexpr size_t kReserveBytes = size_t{1} << 30;  // UI, database, queues
  const size_t available = os_proc_available_memory();
  const int cores = static_cast<int>(
      std::max(1u, std::thread::hardware_concurrency()));
  const size_t usable = available > kReserveBytes ? available - kReserveBytes : 0;
  const int threads = std::clamp(
      static_cast<int>(usable / bytes_per_thread), 1, cores);
  LOG(INFO) << stage << " threads: " << threads << " (available "
            << (available >> 20) << " MiB, " << (bytes_per_thread >> 20)
            << " MiB per thread, " << cores << " cores)";
  return threads;
}
#endif

// Runs a stage thread with cancellation bookkeeping. Returns false if
// cancellation was requested during the stage.
bool RunStageThread(std::unique_ptr<colmap::Thread> thread) {
  {
    std::lock_guard<std::mutex> lock(g_cancel_mutex);
    if (g_cancel_requested) return false;
    g_active_thread = thread.get();
  }
  thread->Start();
  thread->Wait();
  {
    std::lock_guard<std::mutex> lock(g_cancel_mutex);
    g_active_thread = nullptr;
  }
  return !g_cancel_requested;
}

int Run(const fs::path& image_path,
        const fs::path& workspace,
        const std::string& priors_path,
        const std::string& camera_model,
        bool use_global,
        bool single_camera) {
  const fs::path database_path = workspace / "database.db";
  const fs::path sparse_path = workspace / "sparse";
  fs::create_directories(sparse_path);

  std::unordered_map<std::string, FramePrior> priors;
  PriorIntrinsics prior_intrinsics;
  if (!priors_path.empty() &&
      ParsePriorsFile(priors_path, &priors, &prior_intrinsics)) {
    LOG(INFO) << "Loaded " << priors.size() << " frame priors"
              << (prior_intrinsics.valid ? " (with intrinsics)" : "");
  }

  // Live preview (optional): read-only observer of the stages below.
  std::unique_ptr<cross3dgs_sfm::LivePreview> preview;
  {
    cross3dgs_sfm::PreviewTarget target;
    {
      std::lock_guard<std::mutex> lock(g_preview_mutex);
      target = g_preview;
    }
    if (target.fn != nullptr) {
      std::unordered_map<std::string, Eigen::Vector3d> gravity_by_image;
      for (const auto& [name, prior] : priors) {
        if (prior.has_gravity) gravity_by_image.emplace(name, prior.gravity);
      }
      preview = std::make_unique<cross3dgs_sfm::LivePreview>(
          target, image_path, database_path, std::move(gravity_by_image));
    }
  }
  // Largest model of the manager (the one the mapper is growing).
  const auto largest = [](const colmap::ReconstructionManager& manager)
      -> std::shared_ptr<const colmap::Reconstruction> {
    std::shared_ptr<const colmap::Reconstruction> best;
    for (size_t i = 0; i < manager.Size(); ++i) {
      if (!best || manager.Get(i)->NumRegImages() > best->NumRegImages()) {
        best = manager.Get(i);
      }
    }
    return best;
  };

  // Stage 1: feature extraction.
  colmap::ImageReaderOptions reader_options;
  reader_options.image_path = image_path;
  reader_options.single_camera = single_camera;
  if (prior_intrinsics.valid) {
    // Known intrinsics from the AR capture; marks has_prior_focal_length.
    reader_options.camera_model = "PINHOLE";
    std::ostringstream params;
    params << prior_intrinsics.fx << "," << prior_intrinsics.fy << ","
           << prior_intrinsics.cx << "," << prior_intrinsics.cy;
    reader_options.camera_params = params.str();
  } else if (!camera_model.empty()) {
    reader_options.camera_model = camera_model;
  }

  colmap::FeatureExtractionOptions extraction_options;
  extraction_options.use_gpu = false;
  extraction_options.max_image_size = 2400;
  extraction_options.sift->max_num_features = 8192;
#ifdef CROSS3DGS_SFM_MEMORY_BOUNDED_THREADS
  extraction_options.num_threads =
      MemoryBoundedThreads(size_t{900} << 20, "Feature extraction");
#endif
  OverrideThreads(g_extraction_threads.load(), &extraction_options.num_threads);
  g_actual_extraction = extraction_options.num_threads;

  LOG(INFO) << "=== Feature extraction ===";
  if (!RunStageThread(colmap::CreateFeatureExtractorController(
          database_path, reader_options, extraction_options))) {
    return CROSS3DGS_SFM_CANCELED;
  }

  size_t num_priors_written = 0;
  bool have_gravity = false;
  if (!priors.empty()) {
    num_priors_written = WritePriorsToDatabase(database_path, priors);
    for (const auto& [name, prior] : priors) {
      if (prior.has_gravity) {
        have_gravity = true;
        break;
      }
    }
    LOG(INFO) << "Wrote " << num_priors_written << " pose priors to database";
  }

  // Stage 2: matching. Exhaustive for small sets, sequential beyond 100.
  size_t num_images = 0;
  {
    auto database = colmap::Database::Open(database_path);
    num_images = database->NumImages();
  }
  colmap::FeatureMatchingOptions matching_options;
  matching_options.use_gpu = false;
#ifdef CROSS3DGS_SFM_MEMORY_BOUNDED_THREADS
  matching_options.num_threads =
      MemoryBoundedThreads(size_t{400} << 20, "Feature matching");
#endif
  OverrideThreads(g_matching_threads.load(), &matching_options.num_threads);
  g_actual_matching = matching_options.num_threads;
  colmap::TwoViewGeometryOptions geometry_options;
  geometry_options.ransac_options.random_seed = g_seed.load();

  LOG(INFO) << "=== Feature matching (" << num_images << " images) ===";
  if (preview) {
    preview->SetExpectedPairs(num_images <= 100 ? num_images * (num_images - 1) / 2
                                                : 0);
    preview->SetStage(CROSS3DGS_SFM_STAGE_MATCHING);
  }
  if (num_images <= 100) {
    colmap::ExhaustivePairingOptions pairing_options;
    // Small blocks keep cancellation responsive (Thread::Stop is honored
    // only between blocks).
    pairing_options.block_size = 20;
    if (!RunStageThread(colmap::CreateExhaustiveFeatureMatcher(
            pairing_options, matching_options, geometry_options,
            database_path))) {
      return CROSS3DGS_SFM_CANCELED;
    }
  } else {
    colmap::SequentialPairingOptions pairing_options;
    pairing_options.overlap = 15;
    pairing_options.loop_detection = false;
    if (!RunStageThread(colmap::CreateSequentialFeatureMatcher(
            pairing_options, matching_options, geometry_options,
            database_path))) {
      return CROSS3DGS_SFM_CANCELED;
    }
  }

  // Stage 3: mapping.
  auto reconstruction_manager =
      std::make_shared<colmap::ReconstructionManager>();
  if (preview) preview->SetStage(CROSS3DGS_SFM_STAGE_MAPPING);
  if (use_global) {
    LOG(INFO) << "=== Global mapping ===";
    colmap::GlobalPipelineOptions options;
    options.image_path = image_path;
    if (have_gravity) {
      options.mapper.rotation_averaging.use_gravity = true;
      LOG(INFO) << "Rotation averaging with gravity priors";
    }
    auto database = colmap::Database::Open(database_path);
    colmap::GlobalPipeline pipeline(std::move(options), database,
                                    reconstruction_manager);
    pipeline.SetCheckIfStoppedFunc(
        []() { return g_cancel_requested.load(); });
    pipeline.Run();
    if (num_priors_written > 0 && reconstruction_manager->Size() > 0) {
      const std::vector<colmap::PosePrior> db_priors =
          database->ReadAllPosePriors();
      colmap::RANSACOptions ransac_options;
      ransac_options.max_error = 0.25;
      for (size_t i = 0; i < reconstruction_manager->Size(); ++i) {
        colmap::Sim3d metric_from_model;
        if (colmap::AlignReconstructionToPosePriors(
                *reconstruction_manager->Get(i), db_priors, ransac_options,
                &metric_from_model)) {
          reconstruction_manager->Get(i)->Transform(metric_from_model);
          LOG(INFO) << "Aligned global model " << i
                    << " to pose priors (metric)";
        } else {
          LOG(WARNING) << "Global model " << i
                       << " could not be aligned to pose priors";
        }
      }
    }
  } else {
    LOG(INFO) << "=== Incremental mapping ===";
    auto options = std::make_shared<colmap::IncrementalPipelineOptions>();
    options->image_path = image_path;
    // Default (100) rejects every init pair on sparse capture subsets.
    options->mapper.init_min_num_inliers = 50;
    OverrideThreads(g_mapping_threads.load(), &options->num_threads);
    g_actual_mapping = options->num_threads;
    options->random_seed = g_seed.load();
    LOG(INFO) << "Threads (-1 = one per core): extraction "
              << extraction_options.num_threads << ", matching "
              << matching_options.num_threads << ", mapping "
              << options->num_threads;
    if (num_priors_written > 0) {
      options->use_prior_position = true;
      options->use_robust_loss_on_prior_position = true;
      LOG(INFO) << "Incremental mapping with position priors";
    }
    colmap::IncrementalPipeline pipeline(
        options, colmap::Database::Open(database_path),
        reconstruction_manager);
    pipeline.SetCheckIfStoppedFunc(
        []() { return g_cancel_requested.load(); });
    if (preview) {
      // Runs on this (mapping) thread after each registration; copies only.
      const auto snapshot = [&]() {
        if (const auto model = largest(*reconstruction_manager)) {
          preview->OnReconstruction(*model);
        }
      };
      pipeline.AddCallback(
          colmap::IncrementalPipeline::INITIAL_IMAGE_PAIR_REG_CALLBACK, snapshot);
      pipeline.AddCallback(colmap::IncrementalPipeline::NEXT_IMAGE_REG_CALLBACK,
                           snapshot);
    }
    pipeline.Run();
  }

  if (g_cancel_requested) return CROSS3DGS_SFM_CANCELED;
  if (reconstruction_manager->Size() == 0) {
    LOG(ERROR) << "No sparse model produced";
    return CROSS3DGS_SFM_NO_MODEL;
  }
  reconstruction_manager->Write(sparse_path);
  if (preview) {
    // The destructor (on return) emits this final model as the last frame.
    preview->OnReconstruction(*largest(*reconstruction_manager));
    preview->SetStage(CROSS3DGS_SFM_STAGE_DONE);
  }
  LOG(INFO) << "=== Done: " << reconstruction_manager->Size()
            << " model(s) written to " << sparse_path << " ===";
  return CROSS3DGS_SFM_OK;
}

}  // namespace

extern "C" int cross3dgs_sfm_run(const char* image_path,
                                 const char* workspace_path,
                                 const char* priors_path,
                                 const char* camera_model,
                                 int use_global,
                                 int single_camera,
                                 cross3dgs_sfm_log_fn log,
                                 void* log_user_data) {
  EnsureLoggingInitialized();
  {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log = log;
    g_log_user_data = log_user_data;
  }
  g_cancel_requested = false;
  int result = CROSS3DGS_SFM_FAILED;
  try {
    const ScopedCpuMask cpu_mask;
#if defined(__ANDROID__)
    LOG(INFO) << "OpenBLAS: " << openblas_get_config()
              << "; core " << openblas_get_corename()
              << "; threads " << openblas_get_num_threads();
#endif
    result = Run(image_path != nullptr ? image_path : "",
                 workspace_path != nullptr ? workspace_path : "",
                 priors_path != nullptr ? priors_path : "",
                 camera_model != nullptr ? camera_model : "",
                 use_global != 0, single_camera != 0);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Reconstruction failed: " << e.what();
    result = CROSS3DGS_SFM_FAILED;
  }
  {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log = nullptr;
    g_log_user_data = nullptr;
  }
  return result;
}

extern "C" void cross3dgs_sfm_cancel(void) {
  g_cancel_requested = true;
  std::lock_guard<std::mutex> lock(g_cancel_mutex);
  if (g_active_thread != nullptr) {
    g_active_thread->Stop();
  }
}

extern "C" void cross3dgs_sfm_set_preview(cross3dgs_sfm_preview_fn fn,
                                          void* user_data,
                                          int width,
                                          int height) {
  std::lock_guard<std::mutex> lock(g_preview_mutex);
  g_preview.fn = fn;
  g_preview.user_data = user_data;
  g_preview.width = std::clamp(width, 64, 1920);
  g_preview.height = std::clamp(height, 64, 1920);
}

extern "C" void cross3dgs_sfm_set_threads(int extraction, int matching,
                                          int mapping) {
  g_extraction_threads = std::max(extraction, 0);
  g_matching_threads = std::max(matching, 0);
  g_mapping_threads = std::max(mapping, 0);
}

extern "C" void cross3dgs_sfm_set_cpu_mask(unsigned long long mask) {
  g_cpu_mask = mask;
}

extern "C" void cross3dgs_sfm_set_seed(int seed) { g_seed = std::max(-1, seed); }
extern "C" void cross3dgs_sfm_get_effective_threads(int* e, int* m, int* b) {
  if (e) *e = g_actual_extraction.load();
  if (m) *m = g_actual_matching.load();
  if (b) *b = g_actual_mapping.load();
}
