// Small helpers shared by the playground tests: device buffers, random data,
// comparison against a reference, and timing with CUDA events.
#pragma once
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "asr_kernels.h"

#define CHECK(call)                                                                     \
  do {                                                                                  \
    cudaError_t err_ = static_cast<cudaError_t>(call);                                  \
    if (err_ != cudaSuccess) {                                                          \
      std::fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", __FILE__, __LINE__, \
                   static_cast<int>(err_), cudaGetErrorName(err_), #call);              \
      std::exit(EXIT_FAILURE);                                                          \
    }                                                                                   \
  } while (0)

// `./test_taskN small` runs only the small cases and does not time them. Use it under
// compute-sanitizer, which makes every kernel launch many times slower.
inline bool g_small_only = false;

inline void parse_args(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "small") {
      g_small_only = true;
    } else {
      std::fprintf(stderr, "Unknown argument '%s'. The only option is 'small'.\n", argv[i]);
      std::exit(EXIT_FAILURE);
    }
  }
  if (g_small_only) std::printf("Small cases only, without timing.\n");
}

// Print the GPU and return its approximate peak memory bandwidth in GB/s.
inline double print_gpu() {
  int dev = 0, clock_khz = 0, bus_bits = 0;
  CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, dev));
  CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, dev));
  CHECK(cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev));
  double peak = 2.0 * clock_khz * 1e3 * (bus_bits / 8.0) / 1e9;
  std::printf("GPU: %s (sm_%d%d), %d SMs, peak memory bandwidth about %.0f GB/s\n\n",
              prop.name, prop.major, prop.minor, prop.multiProcessorCount, peak);
  return peak;
}

// A float array on the GPU.
struct DeviceArray {
  float* ptr = nullptr;
  size_t n = 0;
  explicit DeviceArray(size_t count) : n(count) {
    CHECK(cudaMalloc(&ptr, std::max<size_t>(n, 1) * sizeof(float)));
  }
  explicit DeviceArray(const std::vector<float>& host) : DeviceArray(host.size()) {
    upload(host);
  }
  ~DeviceArray() { cudaFree(ptr); }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  void upload(const std::vector<float>& host) {
    CHECK(cudaMemcpy(ptr, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  }
  std::vector<float> download() const {
    std::vector<float> host(n);
    CHECK(cudaMemcpy(host.data(), ptr, n * sizeof(float), cudaMemcpyDeviceToHost));
    return host;
  }
};

inline std::vector<float> random_vector(size_t n, unsigned seed, float lo = -1.0f,
                                        float hi = 1.0f) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(gen);
  return v;
}

// Compare got[i] with ref[i]: element i is wrong if |got - ref| > atol + rtol * |ref|
// (NaN and inf count as wrong).
struct Comparison {
  double max_err = 0.0;
  size_t wrong = 0;
  size_t first_wrong = 0;
  float first_got = 0.0f;
  double first_ref = 0.0;
  size_t checked = 0;
};

inline void compare_into(Comparison& c, float got, double ref, size_t index, double atol,
                         double rtol) {
  double err = std::fabs(static_cast<double>(got) - ref);
  if (!std::isfinite(got)) err = INFINITY;
  c.max_err = std::max(c.max_err, err);
  if (!(err <= atol + rtol * std::fabs(ref))) {
    if (c.wrong == 0) {
      c.first_wrong = index;
      c.first_got = got;
      c.first_ref = ref;
    }
    ++c.wrong;
  }
  ++c.checked;
}

inline Comparison compare(const std::vector<float>& got, const std::vector<double>& ref,
                          double atol, double rtol) {
  Comparison c;
  for (size_t i = 0; i < got.size(); ++i) compare_into(c, got[i], ref[i], i, atol, rtol);
  return c;
}

// Average time of one call of `launch` in milliseconds, measured with CUDA events.
template <class Launch>
float time_ms(Launch launch, int warmup = 3, int iters = 20) {
  if (g_small_only) return 0.0f;
  for (int i = 0; i < warmup; ++i) CHECK(launch());
  CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) CHECK(launch());
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, start, stop));
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  return ms / iters;
}

// One line of the result table. Returns true if the case passed.
inline bool report(const std::string& name, const Comparison& c, float ms,
                   const std::string& speed) {
  bool ok = c.wrong == 0;
  if (ok && g_small_only) {
    std::printf("%-36s PASS  max_err=%.2e\n", name.c_str(), c.max_err);
  } else if (ok) {
    std::printf("%-36s PASS  max_err=%-9.2e %8.3f ms  %s\n", name.c_str(), c.max_err, ms,
                speed.c_str());
  } else {  // the speed of a wrong kernel means nothing, so it is not shown
    std::printf("%-36s FAIL  max_err=%.2e\n", name.c_str(), c.max_err);
    std::printf("    %zu of %zu checked values are wrong. First: index %zu, got %g, expected %g\n",
                c.wrong, c.checked, c.first_wrong, c.first_got, c.first_ref);
  }
  return ok;
}

inline std::string fmt(const char* format, double a, double b = 0.0) {
  char buf[128];
  std::snprintf(buf, sizeof(buf), format, a, b);
  return buf;
}

inline int finish(const char* task, bool all_ok) {
  std::printf("\n%s: %s\n", task, all_ok ? "ALL PASS" : "SOME CASES FAILED");
  return all_ok ? 0 : 1;
}
