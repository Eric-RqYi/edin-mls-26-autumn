// Experiment-only references/timing; coursework tests are kept unchanged.
#include <cublas_v2.h>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "test_util.h"

using Linear = int (*)(const float*, const float*, const float*, float*, int, int,
                      int, int, cudaStream_t);
struct Variant { std::string name; void* handle; Linear launch; };
struct Case { std::string name; int M, N, K, bias, act; };
struct Options {
  std::string cases, csv;
  int warmup = 5, iterations = 20, repeats = 7, seed = 21;
  bool small = false, correctness_only = false;
  std::vector<Variant> variants;
};

static void blas_check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS reference failed: " + std::to_string(status));
}

static Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("Missing value for " + arg);
      return argv[i];
    };
    if (arg == "--cases") o.cases = value();
    else if (arg == "--csv") o.csv = value();
    else if (arg == "--warmup") o.warmup = std::stoi(value());
    else if (arg == "--iterations") o.iterations = std::stoi(value());
    else if (arg == "--repeats") o.repeats = std::stoi(value());
    else if (arg == "--seed") o.seed = std::stoi(value());
    else if (arg == "--small") o.small = true;
    else if (arg == "--correctness-only") o.correctness_only = true;
    else {
      size_t split = arg.find('=');
      if (split == 0 || split == std::string::npos || split + 1 == arg.size())
        throw std::runtime_error("Expected name=/path/library.so: " + arg);
      std::string name = arg.substr(0, split);
      if (name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_") != std::string::npos)
        throw std::runtime_error("Invalid variant name");
      for (const auto& v : o.variants)
        if (v.name == name) throw std::runtime_error("Duplicate variant");
      void* handle = dlopen(arg.substr(split + 1).c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!handle) throw std::runtime_error(dlerror());
      dlerror();
      auto fn = reinterpret_cast<Linear>(dlsym(handle, "launch_linear"));
      if (const char* error = dlerror()) throw std::runtime_error(error);
      o.variants.push_back({name, handle, fn});
    }
  }
  if (o.cases.empty() || o.variants.empty() || (!o.correctness_only && o.csv.empty()) ||
      o.warmup < 0 || o.iterations < 1 || o.repeats < 1 || o.seed < 0)
    throw std::runtime_error("Invalid benchmark configuration");
  return o;
}

static std::vector<Case> read_cases(const std::string& path) {
  std::ifstream input(path);
  std::string line;
  if (!std::getline(input, line) || line != "case,M,N,K,bias,act")
    throw std::runtime_error("Invalid case manifest header");
  std::vector<Case> cases;
  while (std::getline(input, line)) {
    std::istringstream fields(line);
    std::vector<std::string> v;
    std::string value;
    while (std::getline(fields, value, ',')) v.push_back(value);
    if (v.size() != 6) throw std::runtime_error("Invalid case row");
    Case c{v[0], std::stoi(v[1]), std::stoi(v[2]), std::stoi(v[3]),
           std::stoi(v[4]), std::stoi(v[5])};
    if (c.name.empty() || c.M < 1 || c.N < 1 || c.K < 1 ||
        (c.bias != 0 && c.bias != 1) || (c.act != 0 && c.act != 1))
      throw std::runtime_error("Invalid case dimensions/epilogue");
    for (const auto& previous : cases)
      if (previous.name == c.name) throw std::runtime_error("Duplicate case");
    cases.push_back(c);
  }
  if (cases.empty()) throw std::runtime_error("Empty case manifest");
  return cases;
}

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

static void run_case(const Case& t, int index, const Options& o, cublasHandle_t blas,
                     cudaStream_t stream, std::ofstream& csv) {
  const size_t nx = static_cast<size_t>(t.M) * t.K;
  const size_t nw = static_cast<size_t>(t.N) * t.K;
  const size_t ny = static_cast<size_t>(t.M) * t.N;
  auto x = random_vector(nx, o.seed + index * 3);
  auto w = random_vector(nw, o.seed + index * 3 + 1, -0.1f, 0.1f);
  auto b = random_vector(t.N, o.seed + index * 3 + 2);
  constexpr size_t guard = 32;
  constexpr float sentinel = 12345.0f;
  DeviceArray dx(x), dw(w), db(b), dy(ny + guard), dref(ny);
  // Uploads use the legacy default stream, whereas launches below deliberately
  // use a nonblocking stream. Finish transfers before either reference or test.
  CHECK(cudaDeviceSynchronize());
  const float* bias = t.bias ? db.ptr : nullptr;
  const bool cpu = double(t.M) * t.N * t.K <= 2e7;
  std::vector<double> reference(ny);
  if (cpu) {
    for (int m = 0; m < t.M; ++m)
      for (int n = 0; n < t.N; ++n) {
        double sum = 0;
        for (int k = 0; k < t.K; ++k)
          sum += double(x[static_cast<size_t>(m) * t.K + k]) *
                 w[static_cast<size_t>(n) * t.K + k];
        reference[static_cast<size_t>(m) * t.N + n] = sum;
      }
  } else {
    const float one = 1.0f, zero = 0.0f;
    // Row-major y=x*w^T -> column-major y^T=w*x^T, plain FP32 reference.
    blas_check(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, t.N, t.M, t.K,
                          &one, dw.ptr, t.K, dx.ptr, t.K, &zero, dref.ptr, t.N));
    CHECK(cudaStreamSynchronize(stream));
    auto values = dref.download();
    reference.assign(values.begin(), values.end());
  }
  for (size_t i = 0; i < ny; ++i) {
    if (t.bias) reference[i] += b[i % t.N];
    if (t.act == 1)
      reference[i] = 0.5 * reference[i] * (1.0 + std::erf(reference[i] / std::sqrt(2.0)));
  }
  std::vector<double> errors;
  for (const auto& variant : o.variants) {
    dy.upload(std::vector<float>(ny + guard, sentinel));
    CHECK(cudaDeviceSynchronize());  // guard upload is outside every timed region
    CHECK(variant.launch(dx.ptr, dw.ptr, bias, dy.ptr, t.M, t.N, t.K, t.act, stream));
    CHECK(cudaStreamSynchronize(stream));
    auto got = dy.download();
    for (size_t i = ny; i < ny + guard; ++i)
      if (got[i] != sentinel) throw std::runtime_error("Output guard overwritten");
    got.resize(ny);
    auto c = compare(got, reference, 1e-3, 1e-3);
    std::printf("%-20s %-8s %s max_err=%.3e ref=%s\n", t.name.c_str(),
                variant.name.c_str(), c.wrong ? "FAIL" : "PASS", c.max_err,
                cpu ? "CPU_double" : "cuBLAS_FP32");
    if (c.wrong) {
      std::fprintf(stderr, "First mismatch at %zu: got %.9g, expected %.17g\n",
                   c.first_wrong, c.first_got, c.first_ref);
      throw std::runtime_error("Correctness failed; no performance claim is valid");
    }
    errors.push_back(c.max_err);
  }
  if (o.correctness_only) return;
  std::vector<size_t> order(o.variants.size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(o.seed + index);
  for (int repeat = 1; repeat <= o.repeats; ++repeat) {
    std::shuffle(order.begin(), order.end(), rng);
    for (size_t position = 0; position < order.size(); ++position) {
      size_t i = order[position];
      const auto& variant = o.variants[i];
      float ms = measure([&] {
        return variant.launch(dx.ptr, dw.ptr, bias, dy.ptr, t.M, t.N, t.K, t.act, stream);
      }, o, stream);
      if (!std::isfinite(ms) || ms <= 0) throw std::runtime_error("Invalid CUDA time");
      csv << t.name << ',' << variant.name << ',' << t.M << ',' << t.N << ',' << t.K << ','
          << t.bias << ',' << t.act << ',' << repeat << ',' << position << ',' << ms << ','
          << 2.0 * t.M * t.N * t.K / (ms * 1e6) << ',' << errors[i] << ",0,"
          << (cpu ? "CPU_double" : "cuBLAS_FP32") << ',' << o.seed << ','
          << o.warmup << ',' << o.iterations << '\n';
    }
  }
  csv.flush();
  if (!csv) throw std::runtime_error("CSV write failed");
}

int main(int argc, char** argv) {
  try {
    auto o = parse(argc, argv);
    auto cases = read_cases(o.cases);
    print_gpu();
    cudaStream_t stream;
    CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cublasHandle_t blas;
    blas_check(cublasCreate(&blas));
    blas_check(cublasSetStream(blas, stream));
    blas_check(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
    std::ofstream csv;
    if (!o.correctness_only) {
      csv.open(o.csv);
      if (!csv) throw std::runtime_error("Cannot open CSV");
      csv.precision(12);
      csv << "case,variant,M,N,K,bias,act,repeat,order,ms,gflops,max_abs_error,wrong,"
             "reference,seed,warmup,iterations\n";
    }
    for (size_t i = 0; i < cases.size(); ++i) {
      const auto& t = cases[i];
      if (o.small && double(t.M) * t.N * t.K > 1e5) continue;
      run_case(t, static_cast<int>(i), o, blas, stream, csv);
    }
    blas_check(cublasDestroy(blas));
    CHECK(cudaStreamDestroy(stream));
    std::puts("Task 3 GEMM experiment: ALL PASS");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "Task 3 GEMM experiment failed: %s\n", error.what());
    return 1;
  }
}
