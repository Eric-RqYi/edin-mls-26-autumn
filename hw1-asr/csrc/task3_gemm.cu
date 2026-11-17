// Task 3: the linear layer as a tiled GEMM. Read docs/task3-gemm.md first.
//
//     y[M, N] = act( x[M, K] * w[N, K]^T + bias[N] )
//
// This is the tiled GEMM from cuda-tutorial/5-tiled-gemm, with three changes:
//   1. w is stored the way PyTorch stores Linear weights: one row per OUTPUT
//      feature, [N, K]. So column n of the result uses row n of w.
//   2. bias is added, and GELU can be applied, before the result is stored
//      (the "epilogue"), so no extra kernel has to read y again.
//   3. every Linear layer of the ASR model and both convolutions run through it.
//
// Block (TILE x TILE threads) computes a TILE x TILE piece of y. Thread (ty, tx)
// computes y[row][col] with row = blockIdx.y*TILE + ty, col = blockIdx.x*TILE + tx.
// The block walks along K one tile at a time, staging a tile of x and a tile of w
// in shared memory so every loaded value is used TILE times.
#include "asr_kernels.h"
#include "common.cuh"

// Keep the previously validated tiled16 + M=1/w4 configuration by default.
// The next-round experiment builds each alternative into a separate library.
#ifndef ASR_TASK3_TILE
#define ASR_TASK3_TILE 16
#endif
#ifndef ASR_TASK3_GEMM_IMPL
#define ASR_TASK3_GEMM_IMPL 0  // 0: one output/thread, 1: register blocking
#endif
#ifndef ASR_TASK3_RM
#define ASR_TASK3_RM 2
#endif
#ifndef ASR_TASK3_RN
#define ASR_TASK3_RN 2
#endif
constexpr int TILE = ASR_TASK3_TILE;
static_assert(TILE == 8 || TILE == 16 || TILE == 32, "TILE must be 8, 16 or 32");
static_assert(ASR_TASK3_GEMM_IMPL == 0 || ASR_TASK3_GEMM_IMPL == 1,
              "GEMM_IMPL must be 0 (tiled) or 1 (register blocked)");
static_assert(ASR_TASK3_RM >= 1 && ASR_TASK3_RM <= 4 &&
              ASR_TASK3_RN >= 1 && ASR_TASK3_RN <= 4,
              "RM and RN must be between 1 and 4");

// Build the original tiled baseline with -DASR_TASK3_M1_ENABLED=0. Separate
// build directories are required when changing these flags (see experiments/).
#ifndef ASR_TASK3_M1_ENABLED
#define ASR_TASK3_M1_ENABLED 1
#endif
#ifndef ASR_TASK3_M1_WARPS
#define ASR_TASK3_M1_WARPS 4
#endif
static_assert(ASR_TASK3_M1_ENABLED == 0 || ASR_TASK3_M1_ENABLED == 1,
              "ASR_TASK3_M1_ENABLED must be 0 or 1");
static_assert(ASR_TASK3_M1_WARPS >= 1 && ASR_TASK3_M1_WARPS <= 32,
              "ASR_TASK3_M1_WARPS must be between 1 and 32");

#if ASR_TASK3_M1_ENABLED
// M = 1 is a matrix-vector product, not a square output tile. Each warp
// computes one output feature: adjacent lanes read adjacent K elements of a
// contiguous weight row. The input vector is reused through the GPU caches.
// There is no shared-memory staging or block-wide barrier on this path.
__global__ void linear_m1_kernel(const float* x, const float* w, const float* bias,
                                  float* y, int N, int K, int act) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp = threadIdx.x / WARP_SIZE;
  const int col = blockIdx.x * ASR_TASK3_M1_WARPS + warp;
  // This condition is uniform within a warp, so every surviving warp still
  // has all 32 lanes participating in the FULL_MASK shuffle below.
  if (col >= N) return;

  const float* wr = w + static_cast<size_t>(col) * K;
  float acc = 0.0f;
  for (int k = lane; k < K; k += WARP_SIZE) acc += x[k] * wr[k];
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    acc += __shfl_down_sync(FULL_MASK, acc, offset);

  if (lane == 0) {
    if (bias != nullptr) acc += bias[col];
    if (act == ACT_GELU) acc = gelu(acc);
    y[col] = acc;
  }
}
#endif

__global__ void linear_tiled_kernel(const float* x, const float* w, const float* bias,
                                    float* y, int M, int N, int K, int act) {
  __shared__ float xs[TILE][TILE];      // xs[m][k]: rows of x for this block
  __shared__ float ws[TILE][TILE + 1];  // ws[n][k]: rows of w for this block (+1: see Try this)

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;  // the output row this thread computes
  const int col = blockIdx.x * TILE + tx;  // the output column this thread computes

  float acc = 0.0f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    // ======================== BLANK 3.1 =========================
    // Load this thread's element of the x tile into xs. Use 0 outside x.
    // YOUR CODE HERE
    xs[ty][tx] = (row < M && k0 + tx < K)
    ? x[static_cast<size_t>(row) * K + k0 + tx]
    : 0.0f;
    // ============================================================

    // ======================== BLANK 3.2 =========================
    // Load this thread's element of the w tile, so that ws[n][k] holds
    // w[blockIdx.x * TILE + n][k0 + k]. Use 0 outside w. Threads with consecutive
    // tx must read consecutive addresses of w.
    // YOUR CODE HERE
    const int w_row = blockIdx.x * TILE + ty;
    ws[ty][tx] = (w_row < N && k0 + tx < K)
    ? w[static_cast<size_t>(w_row) * K + k0 + tx]
    : 0.0f;
    // ============================================================

    // ======================== BLANK 3.3 =========================
    // Add this K step's contribution to acc.
    // YOUR CODE HERE
    __syncthreads();
    for (int k = 0; k < TILE; k++){
      acc += xs[ty][k] * ws[tx][k];
    }
    __syncthreads();
    // ============================================================
  }

  // ======================== BLANK 3.4 =========================
  // Epilogue: for outputs inside y, add bias[col] (bias may be nullptr), apply
  // gelu() when act == ACT_GELU, and store.
  // YOUR CODE HERE
  if (row < M && col < N) {
      if (bias != nullptr) acc += bias[col];
      if (act == ACT_GELU) acc = gelu(acc);
      y[row * N + col] = acc;
  }
  // ============================================================
}

#if ASR_TASK3_GEMM_IMPL == 1
// A 16x16 thread block computes (16*RM)x(16*RN) outputs. Threads keep their
// RM*RN accumulators in registers and reuse RM inputs / RN weights per K step.
// Cooperative global reads remain contiguous along K, including edge tiles.
constexpr int RB_THREADS = 16;
constexpr int RB_K = 16;
constexpr int RB_M = RB_THREADS * ASR_TASK3_RM;
constexpr int RB_N = RB_THREADS * ASR_TASK3_RN;

__global__ void linear_register_kernel(const float* x, const float* w, const float* bias,
                                       float* y, int M, int N, int K, int act) {
  __shared__ float xs[RB_M][RB_K];
  __shared__ float ws[RB_N][RB_K + 1];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * RB_THREADS + tx;
  const int row0 = blockIdx.y * RB_M;
  const int col0 = blockIdx.x * RB_N;
  float acc[ASR_TASK3_RM][ASR_TASK3_RN] = {};

  for (int k0 = 0; k0 < K; k0 += RB_K) {
    for (int index = tid; index < RB_M * RB_K; index += RB_THREADS * RB_THREADS) {
      const int r = index / RB_K;
      const int k = index % RB_K;
      xs[r][k] = (row0 + r < M && k0 + k < K)
                     ? x[static_cast<size_t>(row0 + r) * K + k0 + k] : 0.0f;
    }
    for (int index = tid; index < RB_N * RB_K; index += RB_THREADS * RB_THREADS) {
      const int n = index / RB_K;
      const int k = index % RB_K;
      ws[n][k] = (col0 + n < N && k0 + k < K)
                     ? w[static_cast<size_t>(col0 + n) * K + k0 + k] : 0.0f;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < RB_K; ++k) {
      float av[ASR_TASK3_RM], bv[ASR_TASK3_RN];
      #pragma unroll
      for (int i = 0; i < ASR_TASK3_RM; ++i) av[i] = xs[ty + i * RB_THREADS][k];
      #pragma unroll
      for (int j = 0; j < ASR_TASK3_RN; ++j) bv[j] = ws[tx + j * RB_THREADS][k];
      #pragma unroll
      for (int i = 0; i < ASR_TASK3_RM; ++i) {
        #pragma unroll
        for (int j = 0; j < ASR_TASK3_RN; ++j) acc[i][j] += av[i] * bv[j];
      }
    }
    // Even threads with no valid output must finish before tiles are reused.
    __syncthreads();
  }
  #pragma unroll
  for (int i = 0; i < ASR_TASK3_RM; ++i) {
    const int row = row0 + ty + i * RB_THREADS;
    #pragma unroll
    for (int j = 0; j < ASR_TASK3_RN; ++j) {
      const int col = col0 + tx + j * RB_THREADS;
      if (row < M && col < N) {
        float value = acc[i][j];
        if (bias != nullptr) value += bias[col];
        if (act == ACT_GELU) value = gelu(value);
        y[static_cast<size_t>(row) * N + col] = value;
      }
    }
  }
}
#endif

ASR_API int launch_linear(const float* x, const float* w, const float* bias, float* y,
                          int M, int N, int K, int act, cudaStream_t stream) {
#if ASR_TASK3_M1_ENABLED
  if (M == 1) {
    linear_m1_kernel<<<ceil_div(N, ASR_TASK3_M1_WARPS),
                       ASR_TASK3_M1_WARPS * WARP_SIZE, 0, stream>>>(
        x, w, bias, y, N, K, act);
    RETURN_LAUNCH_ERROR();
  }
#endif
#if ASR_TASK3_GEMM_IMPL == 1
  dim3 block(RB_THREADS, RB_THREADS);
  dim3 grid(ceil_div(N, RB_N), ceil_div(M, RB_M));
  linear_register_kernel<<<grid, block, 0, stream>>>(x, w, bias, y, M, N, K, act);
#else
  dim3 block(TILE, TILE);
  dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
  linear_tiled_kernel<<<grid, block, 0, stream>>>(x, w, bias, y, M, N, K, act);
#endif
  RETURN_LAUNCH_ERROR();
}
