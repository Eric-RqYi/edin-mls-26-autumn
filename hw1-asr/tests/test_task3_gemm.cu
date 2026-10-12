// Task 3 test: the tiled linear layer on the model's shapes.
// Small cases are checked against a CPU loop, large ones against cuBLAS.
// For each case the test also times a naive kernel (no shared memory) and cuBLAS,
// so you can see where your kernel stands.
// Build and run with:  make test3
#include <cublas_v2.h>

#include "test_util.h"

#define CHECK_CUBLAS(call)                                                     \
  do {                                                                         \
    cublasStatus_t st_ = (call);                                               \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                        \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", static_cast<int>(st_), \
                   __FILE__, __LINE__);                                        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// The naive version: one thread per output, every operand read from global memory.
__global__ void linear_naive_kernel(const float* x, const float* w, float* y, int M, int N,
                                    int K) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
      acc += x[static_cast<size_t>(row) * K + k] * w[static_cast<size_t>(col) * K + k];
    y[static_cast<size_t>(row) * N + col] = acc;
  }
}

static int launch_naive(const float* x, const float* w, float* y, int M, int N, int K) {
  dim3 block(16, 16), grid((N + 15) / 16, (M + 15) / 16);
  linear_naive_kernel<<<grid, block>>>(x, w, y, M, N, K);
  return static_cast<int>(cudaGetLastError());
}

// cuBLAS is column-major. Row-major y = x * w^T is column-major y^T = w * x^T,
// which is gemm(op(w) = transpose, op(x) = none).
static int launch_cublas(cublasHandle_t h, const float* x, const float* w, float* y, int M,
                         int N, int K) {
  const float one = 1.0f, zero = 0.0f;
  return cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, w, K, x, K, &zero, y, N) ==
                 CUBLAS_STATUS_SUCCESS
             ? 0
             : static_cast<int>(cudaErrorUnknown);
}

static double gelu_ref(double v) { return 0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0))); }

struct GemmCase {
  const char* name;
  int M, N, K;
  bool bias;
  int act;
};

static bool run_case(const GemmCase& t, cublasHandle_t handle) {
  size_t nx = static_cast<size_t>(t.M) * t.K, nw = static_cast<size_t>(t.N) * t.K;
  size_t ny = static_cast<size_t>(t.M) * t.N;
  auto x = random_vector(nx, 21);
  auto w = random_vector(nw, 22, -0.1f, 0.1f);
  auto b = random_vector(t.N, 23);
  DeviceArray dx(x), dw(w), db(b), dy(ny), dref(ny);
  const float* bias = t.bias ? db.ptr : nullptr;

  CHECK(launch_linear(dx.ptr, dw.ptr, bias, dy.ptr, t.M, t.N, t.K, t.act, 0));
  CHECK(cudaDeviceSynchronize());
  auto got = dy.download();

  // Reference: x * w^T, from a CPU loop (small cases) or from cuBLAS (large cases).
  std::vector<double> ref(ny);
  bool small = static_cast<double>(t.M) * t.N * t.K <= 2e7;
  if (small) {
    for (int m = 0; m < t.M; ++m)
      for (int n = 0; n < t.N; ++n) {
        double acc = 0;
        for (int k = 0; k < t.K; ++k)
          acc += double(x[static_cast<size_t>(m) * t.K + k]) * w[static_cast<size_t>(n) * t.K + k];
        ref[static_cast<size_t>(m) * t.N + n] = acc;
      }
  } else {
    CHECK(launch_cublas(handle, dx.ptr, dw.ptr, dref.ptr, t.M, t.N, t.K));
    auto r = dref.download();
    ref.assign(r.begin(), r.end());
  }
  for (size_t i = 0; i < ny; ++i) {  // epilogue
    if (t.bias) ref[i] += b[i % t.N];
    if (t.act == 1) ref[i] = gelu_ref(ref[i]);
  }
  Comparison c = compare(got, ref, 1e-3, 1e-3);

  double flops = 2.0 * t.M * t.N * t.K;
  float ms = time_ms([&] { return launch_linear(dx.ptr, dw.ptr, bias, dy.ptr, t.M, t.N, t.K, t.act, 0); });
  float ms_naive = time_ms([&] { return launch_naive(dx.ptr, dw.ptr, dref.ptr, t.M, t.N, t.K); }, 1, 5);
  float ms_cublas = time_ms([&] { return launch_cublas(handle, dx.ptr, dw.ptr, dref.ptr, t.M, t.N, t.K); });
  bool ok = report(t.name, c, ms, fmt("%7.0f GFLOP/s", flops / (ms * 1e-3) / 1e9));
  if (ok && !g_small_only)
    std::printf("    for comparison: naive %6.0f GFLOP/s, cuBLAS %6.0f GFLOP/s (%s reference)\n",
                flops / (ms_naive * 1e-3) / 1e9, flops / (ms_cublas * 1e-3) / 1e9,
                small ? "CPU" : "cuBLAS");
  return ok;
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  print_gpu();
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));  // plain FP32, no TF32

  const GemmCase cases[] = {
      // name                                  M      N     K   bias  act (1 = GELU)
      {"tiny 3 x 5, K = 7", 3, 5, 7, false, 0},
      {"odd sizes 37 x 100, K = 33, bias", 37, 100, 33, true, 0},
      {"odd sizes, bias + GELU", 45, 77, 129, true, 1},
      {"decoder q_proj (64x2048x2048)", 64, 2048, 2048, false, 0},
      {"decoder gate_proj (64x6144x2048)", 64, 6144, 2048, false, 0},
      {"decoder down_proj (64x2048x6144)", 64, 2048, 6144, false, 0},
      {"lm_head, last token (1x59264x2048)", 1, 59264, 2048, false, 0},
      {"encoder q_proj (1500x1280x1280)", 1500, 1280, 1280, true, 0},
      {"encoder fc1 + GELU (1500x5120x1280)", 1500, 5120, 1280, true, 1},
      {"conv2 via im2col (1500x1280x3840)", 1500, 1280, 3840, true, 1},
  };
  bool ok = true;
  for (const auto& t : cases) {
    if (g_small_only && double(t.M) * t.N * t.K > 2e7) continue;
    ok = run_case(t, handle) && ok;
  }
  CHECK_CUBLAS(cublasDestroy(handle));
  return finish("Task 3 (tiled GEMM)", ok);
}
