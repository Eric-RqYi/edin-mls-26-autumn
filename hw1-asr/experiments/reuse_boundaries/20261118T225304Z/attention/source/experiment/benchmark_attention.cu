// Experiment references only; the task kernels never call cuBLAS.
#include <cublas_v2.h>
#include <dlfcn.h>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include "test_util.h"

using Attention = int (*)(const float*, const float*, const float*, float*, int, int, int,
                         int, int, int, float, int, cudaStream_t);
struct Variant { std::string name; void* handle; Attention launch; };
struct Case { std::string name; int B, H, Hkv, Sq, Sk, D, causal; float range; int pattern; };
struct Options {
  std::string cases, csv;
  int warmup = 5, iterations = 20, repeats = 7, seed = 21;
  bool small = false, correctness_only = false;
  std::vector<Variant> variants;
};
static void blas_check(cublasStatus_t s) {
  if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS reference error " + std::to_string(s));
}
static Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("Missing argument for " + a);
      return argv[i];
    };
    if (a == "--cases") o.cases = value();
    else if (a == "--csv") o.csv = value();
    else if (a == "--warmup") o.warmup = std::stoi(value());
    else if (a == "--iterations") o.iterations = std::stoi(value());
    else if (a == "--repeats") o.repeats = std::stoi(value());
    else if (a == "--seed") o.seed = std::stoi(value());
    else if (a == "--small") o.small = true;
    else if (a == "--correctness-only") o.correctness_only = true;
    else {
      auto split = a.find('=');
      if (split == std::string::npos || split == 0 || split + 1 == a.size())
        throw std::runtime_error("Expected name=/path/library.so");
      std::string name = a.substr(0, split);
      if (name.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_") != std::string::npos)
        throw std::runtime_error("Invalid variant name");
      for (const auto& v : o.variants) if (v.name == name) throw std::runtime_error("Duplicate variant");
      void* handle = dlopen(a.substr(split + 1).c_str(), RTLD_LOCAL | RTLD_NOW);
      if (!handle) throw std::runtime_error(dlerror());
      dlerror();
      auto fn = reinterpret_cast<Attention>(dlsym(handle, "launch_attention"));
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
  std::string line, field;
  if (!std::getline(input, line) || line != "case,batch,num_heads,num_kv_heads,seq_q,seq_k,head_dim,causal,q_range,pattern")
    throw std::runtime_error("Invalid attention case header");
  std::vector<Case> cases;
  while (std::getline(input, line)) {
    std::istringstream stream(line);
    std::vector<std::string> v;
    while (std::getline(stream, field, ',')) v.push_back(field);
    if (v.size() != 10) throw std::runtime_error("Invalid attention case row");
    Case t{v[0], std::stoi(v[1]), std::stoi(v[2]), std::stoi(v[3]), std::stoi(v[4]),
           std::stoi(v[5]), std::stoi(v[6]), std::stoi(v[7]), std::stof(v[8]), std::stoi(v[9])};
    if (t.name.empty() || t.B < 1 || t.H < 1 || t.Hkv < 1 || t.H % t.Hkv || t.Sq < 1 || t.Sk < 1 ||
        t.D < 32 || t.D > 128 || t.D % 32 || t.range <= 0 || !std::isfinite(t.range) ||
        (t.causal != 0 && t.causal != 1) || t.pattern < 0 || t.pattern > 3)
      throw std::runtime_error("Invalid attention dimensions/pattern");
    for (const auto& previous : cases) if (previous.name == t.name) throw std::runtime_error("Duplicate case");
    cases.push_back(t);
  }
  if (cases.empty()) throw std::runtime_error("Empty case file");
  return cases;
}
static std::vector<double> cpu_reference(const Case& t, const std::vector<float>& q,
                                        const std::vector<float>& k, const std::vector<float>& v, float scale) {
  std::vector<double> out(q.size(), 0.0), scores(t.Sk);
  const size_t qs = size_t(t.H) * t.D, ks = size_t(t.Hkv) * t.D;
  for (int b = 0; b < t.B; ++b) for (int h = 0; h < t.H; ++h) for (int i = 0; i < t.Sq; ++i) {
    int last = t.causal ? std::min(t.Sk - 1, i + t.Sk - t.Sq) : t.Sk - 1;
    if (last < 0) continue;  // explicit zero convention for fully masked queries
    size_t qb = (size_t(b) * t.Sq + i) * qs + h * t.D;
    size_t kb = size_t(b) * t.Sk * ks + (h / (t.H / t.Hkv)) * t.D;
    double maximum = -INFINITY;
    for (int j = 0; j <= last; ++j) {
      double dot = 0;
      for (int d = 0; d < t.D; ++d) dot += double(q[qb + d]) * k[kb + j * ks + d];
      scores[j] = dot * scale;
      maximum = std::max(maximum, scores[j]);
    }
    double sum = 0;
    for (int j = 0; j <= last; ++j) sum += (scores[j] = std::exp(scores[j] - maximum));
    for (int d = 0; d < t.D; ++d) {
      double acc = 0;
      for (int j = 0; j <= last; ++j) acc += scores[j] * v[kb + j * ks + d];
      out[qb + d] = acc / sum;
    }
  }
  return out;
}
static std::vector<double> blas_reference(const Case& t, const DeviceArray& q, const DeviceArray& k,
                                         const DeviceArray& v, float scale, cublasHandle_t blas, cudaStream_t stream) {
  // Check every output of large cases. Materialize one head's scores at a time;
  // CPU double softmax is independent of the task's streaming implementation.
  DeviceArray score(size_t(t.Sq) * t.Sk), probabilities(size_t(t.Sq) * t.Sk), result(q.n);
  const int qs = t.H * t.D, ks = t.Hkv * t.D;
  const float one = 1, zero = 0;
  for (int b = 0; b < t.B; ++b) for (int h = 0; h < t.H; ++h) {
    size_t qb = size_t(b) * t.Sq * qs + h * t.D;
    size_t kb = size_t(b) * t.Sk * ks + (h / (t.H / t.Hkv)) * t.D;
    blas_check(cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, t.Sk, t.Sq, t.D,
                          &scale, k.ptr + kb, ks, q.ptr + qb, qs, &zero, score.ptr, t.Sk));
    CHECK(cudaStreamSynchronize(stream));
    auto values = score.download();
    std::vector<float> p(values.size(), 0);
    for (int i = 0; i < t.Sq; ++i) {
      int last = t.causal ? std::min(t.Sk - 1, i + t.Sk - t.Sq) : t.Sk - 1;
      if (last < 0) continue;
      double maximum = -INFINITY, sum = 0;
      for (int j = 0; j <= last; ++j) maximum = std::max(maximum, double(values[size_t(i) * t.Sk + j]));
      for (int j = 0; j <= last; ++j) sum += std::exp(double(values[size_t(i) * t.Sk + j]) - maximum);
      for (int j = 0; j <= last; ++j) p[size_t(i) * t.Sk + j] =
          float(std::exp(double(values[size_t(i) * t.Sk + j]) - maximum) / sum);
    }
    probabilities.upload(p);
    CHECK(cudaDeviceSynchronize());  // default-stream H2D -> nonblocking reference stream
    blas_check(cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, t.D, t.Sq, t.Sk,
                          &one, v.ptr + kb, ks, probabilities.ptr, t.Sk, &zero, result.ptr + qb, qs));
    CHECK(cudaStreamSynchronize(stream));
  }
  auto values = result.download();
  return std::vector<double>(values.begin(), values.end());
}
template <class Launch> static float measure(Launch launch, const Options& o, cudaStream_t stream) {
  for (int i = 0; i < o.warmup; ++i) CHECK(launch());
  CHECK(cudaStreamSynchronize(stream));
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < o.iterations; ++i) CHECK(launch());
  CHECK(cudaEventRecord(stop, stream)); CHECK(cudaEventSynchronize(stop));
  float ms; CHECK(cudaEventElapsedTime(&ms, start, stop));
  CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
  return ms / o.iterations;
}
static void run_case(const Case& t, int index, const Options& o, cublasHandle_t blas,
                     cudaStream_t stream, std::ofstream& csv) {
  size_t nq = size_t(t.B) * t.Sq * t.H * t.D, nk = size_t(t.B) * t.Sk * t.Hkv * t.D;
  auto q = random_vector(nq, o.seed + index * 3, -t.range, t.range);
  auto k = random_vector(nk, o.seed + index * 3 + 1, -t.range, t.range);
  auto v = random_vector(nk, o.seed + index * 3 + 2);
  if (t.pattern) {
    std::fill(q.begin(), q.end(), 0);
    if (t.pattern != 1) {
      std::fill(k.begin(), k.end(), 0);
      for (size_t i = 0; i < nq; i += t.D) q[i] = std::sqrt(float(t.D));
      for (int b = 0; b < t.B; ++b) for (int j = 0; j < t.Sk; ++j) for (int h = 0; h < t.Hkv; ++h)
        k[((size_t(b) * t.Sk + j) * t.Hkv + h) * t.D] = t.pattern == 2 ? j * .5f : -50.0f - j * .1f;
    }
  }
  constexpr size_t guard = 32;
  constexpr float sentinel = 12345;
  DeviceArray dq(q), dk(k), dv(v), out(nq + guard);
  CHECK(cudaDeviceSynchronize());
  float scale = 1.0f / std::sqrt(float(t.D));
  bool cpu = double(t.B) * t.H * t.Sq * t.Sk * t.D <= 2e7;
  auto reference = cpu ? cpu_reference(t, q, k, v, scale) : blas_reference(t, dq, dk, dv, scale, blas, stream);
  std::vector<double> errors;
  for (const auto& variant : o.variants) {
    out.upload(std::vector<float>(nq + guard, sentinel)); CHECK(cudaDeviceSynchronize());
    CHECK(variant.launch(dq.ptr, dk.ptr, dv.ptr, out.ptr, t.B, t.H, t.Hkv, t.Sq, t.Sk, t.D, scale, t.causal, stream));
    CHECK(cudaStreamSynchronize(stream));
    auto got = out.download();
    for (size_t i = nq; i < nq + guard; ++i) if (got[i] != sentinel) throw std::runtime_error("Output guard overwritten");
    got.resize(nq);
    auto c = compare(got, reference, 1e-4, 1e-3);
    std::printf("%-18s %-14s %s max_err=%.3e ref=%s\n", t.name.c_str(), variant.name.c_str(),
                c.wrong ? "FAIL" : "PASS", c.max_err, cpu ? "CPU_double" : "cuBLAS_CPU_softmax");
    if (c.wrong) throw std::runtime_error("Attention correctness failed at index " + std::to_string(c.first_wrong));
    errors.push_back(c.max_err);
  }
  if (o.correctness_only) return;
  double pairs = 0;
  for (int i = 0; i < t.Sq; ++i) pairs += t.causal ? std::max(0, std::min(t.Sk, i + t.Sk - t.Sq + 1)) : t.Sk;
  std::vector<size_t> order(o.variants.size()); std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(o.seed + index);
  for (int repeat = 1; repeat <= o.repeats; ++repeat) {
    std::shuffle(order.begin(), order.end(), rng);
    for (size_t position = 0; position < order.size(); ++position) {
      size_t i = order[position]; const auto& variant = o.variants[i];
      float ms = measure([&] { return variant.launch(dq.ptr, dk.ptr, dv.ptr, out.ptr,
          t.B, t.H, t.Hkv, t.Sq, t.Sk, t.D, scale, t.causal, stream); }, o, stream);
      if (!std::isfinite(ms) || ms <= 0) throw std::runtime_error("Invalid CUDA time");
      csv << t.name << ',' << variant.name << ',' << t.B << ',' << t.H << ',' << t.Hkv << ',' << t.Sq << ','
          << t.Sk << ',' << t.D << ',' << t.causal << ',' << t.range << ',' << t.pattern << ','
          << repeat << ',' << position << ',' << ms << ',' << 4.0 * t.B * t.H * pairs * t.D / (ms * 1e6)
          << ',' << errors[i] << ",0," << (cpu ? "CPU_double" : "cuBLAS_CPU_softmax") << ','
          << o.seed << ',' << o.warmup << ',' << o.iterations << '\n';
    }
  }
  csv.flush(); if (!csv) throw std::runtime_error("CSV write failed");
}
int main(int argc, char** argv) {
  try {
    auto o = parse(argc, argv); auto cases = read_cases(o.cases); print_gpu();
    cudaStream_t stream; CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cublasHandle_t blas; blas_check(cublasCreate(&blas)); blas_check(cublasSetStream(blas, stream));
    blas_check(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
    for (const auto& variant : o.variants)
      if (variant.launch(nullptr, nullptr, nullptr, nullptr, 1, 1, 0, 1, 1, 32, 1, 0, stream) != cudaErrorInvalidValue)
        throw std::runtime_error("Invalid num_kv_heads was not rejected");
    std::ofstream csv;
    if (!o.correctness_only) {
      csv.open(o.csv); if (!csv) throw std::runtime_error("Cannot open CSV"); csv.precision(12);
      csv << "case,variant,batch,num_heads,num_kv_heads,seq_q,seq_k,head_dim,causal,q_range,pattern,"
             "repeat,order,ms,gflops,max_abs_error,wrong,reference,seed,warmup,iterations\n";
    }
    for (size_t i = 0; i < cases.size(); ++i) {
      const auto& t = cases[i];
      if (o.small && double(t.B) * t.H * t.Sq * t.Sk * t.D > 1e6) continue;
      run_case(t, int(i), o, blas, stream, csv);
    }
    blas_check(cublasDestroy(blas)); CHECK(cudaStreamDestroy(stream));
    std::puts("Attention experiment: ALL PASS"); return 0;
  } catch (const std::exception& e) { std::fprintf(stderr, "Attention experiment failed: %s\n", e.what()); return 1; }
}
