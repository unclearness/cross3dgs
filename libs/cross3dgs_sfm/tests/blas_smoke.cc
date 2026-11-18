// Correctness smoke for the single-threaded BLAS used concurrently by FAISS.
// Integer-valued inputs make every expected float result exactly representable.
// No timing: this is not a performance experiment or a quality comparison.
#include <cblas.h>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

namespace {
constexpr int kWorkers = 8;
constexpr int kCalls = 16;
constexpr int kM = 127;
constexpr int kN = 131;
constexpr int kK = 129;

template <typename T>
bool CheckGemm(int worker, int call, bool transpose) {
  const int lda = transpose ? kK : kM;
  std::vector<T> a(kM * kK), b(kK * kN), c(kM * kN);
  for (int i = 0; i < static_cast<int>(a.size()); ++i)
    a[i] = T((i + worker + call) % 7 - 3);
  for (int i = 0; i < static_cast<int>(b.size()); ++i)
    b[i] = T((i + 2 * worker + call) % 5 - 2);
  // Exercise beta accumulation as well as packing-buffer reuse.
  for (int i = 0; i < static_cast<int>(c.size()); ++i) c[i] = T(i % 3);
  const auto op = transpose ? CblasTrans : CblasNoTrans;
  if constexpr (sizeof(T) == sizeof(float)) {
    cblas_sgemm(CblasColMajor, op, CblasNoTrans, kM, kN, kK,
                1, a.data(), lda, b.data(), kK, 1, c.data(), kM);
  } else {
    cblas_dgemm(CblasColMajor, op, CblasNoTrans, kM, kN, kK,
                1, a.data(), lda, b.data(), kK, 1, c.data(), kM);
  }
  for (int j = 0; j < kN; ++j) {
    for (int i = 0; i < kM; ++i) {
      int expected = (i + j * kM) % 3;
      for (int k = 0; k < kK; ++k) {
        expected += static_cast<int>(a[transpose ? k + i * lda : i + k * lda]) *
                    static_cast<int>(b[k + j * kK]);
      }
      if (c[i + j * kM] != T(expected)) return false;  // NaN also fails.
    }
  }
  return true;
}
}  // namespace

int main() {
  std::printf("OpenBLAS: %s; core %s; threads %d\n", openblas_get_config(),
              openblas_get_corename(), openblas_get_num_threads());
  if (openblas_get_parallel() != OPENBLAS_SEQUENTIAL ||
      openblas_get_num_threads() != 1) {
    std::fprintf(stderr, "FAIL: expected single-threaded BLAS\n");
    return 1;
  }
  std::mutex mutex;
  std::condition_variable ready;
  int waiting = 0;
  bool start = false;
  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  for (int worker = 0; worker < kWorkers; ++worker) {
    workers.emplace_back([&, worker] {
      {
        std::unique_lock<std::mutex> lock(mutex);
        ++waiting;
        ready.notify_all();
        ready.wait(lock, [&] { return start; });
      }
      for (int call = 0; call < kCalls; ++call) {
        for (bool transpose : {false, true}) {
          if (!CheckGemm<float>(worker, call, transpose)) ++failures;
          if (!CheckGemm<double>(worker, call, transpose)) ++failures;
        }
      }
    });
  }
  {
    std::unique_lock<std::mutex> lock(mutex);
    ready.wait(lock, [&] { return waiting == kWorkers; });
    start = true;
  }
  ready.notify_all();
  for (auto& worker : workers) worker.join();
  std::printf("%s: %d concurrent workers, %d GEMMs, %d mismatches\n",
              failures == 0 ? "PASS" : "FAIL", kWorkers,
              kWorkers * kCalls * 4, failures.load());
  return failures == 0 ? 0 : 1;
}
