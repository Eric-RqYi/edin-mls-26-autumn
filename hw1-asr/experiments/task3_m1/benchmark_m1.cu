// Additional experiment coverage; the coursework tests remain unchanged.
// Compare shared libraries on identical inputs in one process/GPU context.
#include <dlfcn.h>

#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "test_util.h"

using Linear = int (*)(const float*, const float*, const float*, float*, int, int,
                      int, int, cudaStream_t);

struct Variant {
  std::string name;
  void* library;
  Linear launch;
};

struct Case {
  const char* name;
  int M, N, K;
  bool bias;
  int act;
};

struct Options {
  std::string csv;
  int warmup = 5, iterations = 20, repeats = 7, seed = 21;
  bool small = false, correctness_only = false;
  std::vector<Variant> variants;
};

static Options parse_options(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("Missing value for " + arg);
      return argv[i];
    };
    if (arg == "--csv") o.csv = value();
    else if (arg == "--warmup") o.warmup = std::stoi(value());
    else if (arg == "--iterations") o.iterations = std::stoi(value());
    else if (arg == "--repeats") o.repeats = std::stoi(value());
    else if (arg == "--seed") o.seed = std::stoi(value());
    else if (arg == "--small") o.small = true;
    else if (arg == "--correctness-only") o.correctness_only = true;
    else {
      size_t sep = arg.find('=');
      if (sep == std::string::npos || sep == 0 || sep + 1 == arg.size())
        throw std::runtime_error("Expected name=/absolute/library.so, got " + arg);
      std::string name = arg.substr(0, sep);
      if (name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") !=
          std::string::npos)
        throw std::runtime_error("Variant names must use letters, digits, '_' or '-'");
      for (const auto& v : o.variants)
        if (v.name == name) throw std::runtime_error("Duplicate variant " + name);
      void* lib = dlopen(arg.substr(sep + 1).c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!lib) throw std::runtime_error(dlerror());
      dlerror();
      auto launch = reinterpret_cast<Linear>(dlsym(lib, "launch_linear"));
      const char* error = dlerror();
      if (error) throw std::runtime_error(error);
      o.variants.push_back({name, lib, launch});
    }
  }
  if (o.variants.empty()) throw std::runtime_error("At least one variant is required");
  if (o.warmup < 0 || o.iterations < 1 || o.repeats < 1 || o.seed < 0)
    throw std::runtime_error("Invalid warmup/iterations/repeats/seed");
  if (!o.correctness_only && o.csv.empty())
    throw std::runtime_error("--csv is required when timing");
  return o;
}

// Events must be on the same (non-default) stream as the launches. Allocations,
// CPU reference, H2D/D2H copies, and library loading are outside the timed region.
template <class Launch>
static float measure(Launch launch, const Options& o, cudaStream_t stream) {
  for (int i = 0; i < o.warmup; ++i) CHECK(launch());
  CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < o.iterations; ++i) CHECK(launch());
  CHECK(cudaEventRecord(stop, stream));
  CHECK(cudaEventSynchronize(stop));
  float ms;
  CHECK(cudaEventElapsedTime(&ms, start, stop));
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  return ms / o.iterations;
}

static void run_case(const Case& t, int case_index, const Options& o,
                     cudaStream_t stream, std::ofstream& csv) {
  const size_t nx = static_cast<size_t>(t.M) * t.K;
  const size_t nw = static_cast<size_t>(t.N) * t.K;
  const size_t ny = static_cast<size_t>(t.M) * t.N;
  constexpr size_t guard = 32;
  constexpr float sentinel = 12345.0f;
  auto x = random_vector(nx, o.seed + case_index * 3);
  auto w = random_vector(nw, o.seed + case_index * 3 + 1, -0.1f, 0.1f);
  auto b = random_vector(t.N, o.seed + case_index * 3 + 2);
  DeviceArray dx(x), dw(w), db(b), dy(ny + guard);
  const float* bias = t.bias ? db.ptr : nullptr;

  // An independent double-precision CPU reference also checks the entire lm_head.
  std::vector<double> ref(ny);
  for (int m = 0; m < t.M; ++m)
    for (int n = 0; n < t.N; ++n) {
      double acc = 0;
      for (int k = 0; k < t.K; ++k)
        acc += double(x[static_cast<size_t>(m) * t.K + k]) *
               w[static_cast<size_t>(n) * t.K + k];
      if (t.bias) acc += b[n];
      if (t.act == 1) acc = 0.5 * acc * (1.0 + std::erf(acc / std::sqrt(2.0)));
      ref[static_cast<size_t>(m) * t.N + n] = acc;
    }

  std::vector<double> errors;
  for (const auto& v : o.variants) {
    dy.upload(std::vector<float>(ny + guard, sentinel));
    CHECK(v.launch(dx.ptr, dw.ptr, bias, dy.ptr, t.M, t.N, t.K, t.act, stream));
    CHECK(cudaStreamSynchronize(stream));
    auto got = dy.download();
    for (size_t i = ny; i < ny + guard; ++i)
      if (got[i] != sentinel) throw std::runtime_error(v.name + " overwrote output guard");
    got.resize(ny);
    Comparison c = compare(got, ref, 1e-3, 1e-3);  // unchanged coursework tolerances
    std::printf("%-24s %-10s %s max_err=%.3e\n", t.name, v.name.c_str(),
                c.wrong ? "FAIL" : "PASS", c.max_err);
    if (c.wrong) {
      std::fprintf(stderr, "%zu wrong; first at %zu: got %.9g, expected %.17g\n",
                   c.wrong, c.first_wrong, c.first_got, c.first_ref);
      throw std::runtime_error("Correctness failed; do not interpret performance");
    }
    errors.push_back(c.max_err);
  }
  if (o.correctness_only) return;

  std::mt19937 order_rng(o.seed + case_index);
  std::vector<size_t> order(o.variants.size());
  std::iota(order.begin(), order.end(), 0);
  for (int repeat = 1; repeat <= o.repeats; ++repeat) {
    // Shuffle each round to reduce systematic ordering/temperature bias.
    std::shuffle(order.begin(), order.end(), order_rng);
    for (size_t position = 0; position < order.size(); ++position) {
      const size_t i = order[position];
      const auto& v = o.variants[i];
      float ms = measure([&] {
        return v.launch(dx.ptr, dw.ptr, bias, dy.ptr, t.M, t.N, t.K, t.act, stream);
      }, o, stream);
      if (!std::isfinite(ms) || ms <= 0) throw std::runtime_error("Invalid CUDA event time");
      double weight_bytes = double(nw) * sizeof(float);
      double min_io_bytes = double(nx + nw + ny + (t.bias ? t.N : 0)) * sizeof(float);
      // Logical bandwidth proxies, NOT measured DRAM traffic (caches may serve reads).
      csv << t.name << ',' << v.name << ',' << t.M << ',' << t.N << ',' << t.K << ','
          << t.bias << ',' << t.act << ',' << repeat << ',' << position << ',' << ms << ','
          << weight_bytes << ',' << min_io_bytes << ',' << weight_bytes / (ms * 1e6) << ','
          << 2.0 * t.M * t.N * t.K / (ms * 1e6) << ',' << errors[i] << ",0,"
          << o.seed << ',' << o.warmup << ',' << o.iterations << '\n';
    }
  }
  csv.flush();
  if (!csv) throw std::runtime_error("Failed to write CSV");
}

int main(int argc, char** argv) {
  try {
    Options o = parse_options(argc, argv);
    print_gpu();
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::ofstream csv;
    if (!o.correctness_only) {
      csv.open(o.csv);
      if (!csv) throw std::runtime_error("Cannot open CSV " + o.csv);
      csv.precision(12);
      csv << "case,variant,M,N,K,bias,act,repeat,order,ms,weight_bytes,min_io_bytes,"
             "weight_gbs,gflops,max_abs_error,wrong,seed,warmup,iterations\n";
    }
    const Case cases[] = {
        {"m1_scalar", 1, 1, 1, false, 0},
        {"m1_short", 1, 5, 7, false, 0},
        {"m1_bias_k31", 1, 7, 31, true, 0},
        {"m1_gelu_k32", 1, 31, 32, false, 1},
        {"m1_bias_gelu_tail", 1, 33, 33, true, 1},
        {"m1_k129", 1, 77, 129, true, 1},
        {"m1_long_odd", 1, 257, 2049, false, 0},
        {"m1_large_k", 1, 129, 6144, false, 0},
        {"lm_head", 1, 59264, 2048, false, 0},
        {"fallback_m2", 2, 33, 31, true, 1},
        {"fallback_m3", 3, 7, 129, false, 0},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      const auto& t = cases[i];
      if (o.small && double(t.M) * t.N * t.K > 1e5) continue;
      run_case(t, static_cast<int>(i), o, stream, csv);
    }
    CHECK(cudaStreamDestroy(stream));
    // Keep libraries resident until process exit; CUDA registered their kernels.
    std::puts("Task 3 M=1 experiment: ALL PASS");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Task 3 M=1 experiment failed: %s\n", e.what());
    return 1;
  }
}
