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

constexpr int TILE = 16;  // try 8 or 32 (32 x 32 = 1024 threads, the maximum per block)

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
    // Load one element of the x tile: xs[ty][tx] = x[row][k0 + tx].
    // Store 0 instead if that element is outside x (row >= M or k0 + tx >= K).
    // YOUR CODE HERE
    // ============================================================

    // ======================== BLANK 3.2 =========================
    // Load one element of the w tile, so that afterwards
    //     ws[i][j] = w[blockIdx.x * TILE + i][k0 + j]    for all i, j < TILE
    // (rows of w that this block's output columns need). Pick which thread loads
    // which element so that threads with consecutive tx read consecutive addresses.
    // Store 0 outside w (row >= N or column >= K).
    // YOUR CODE HERE
    // ============================================================

    // ======================== BLANK 3.3 =========================
    // Wait for the whole tile, add this tile's contribution to acc
    // (xs[ty][q] times the matching element of ws, for q = 0 .. TILE-1),
    // then wait again before the next iteration overwrites the tile.
    // YOUR CODE HERE
    // ============================================================
  }

  // ======================== BLANK 3.4 =========================
  // Epilogue: if (row, col) is inside y, add bias[col] (bias may be nullptr),
  // apply gelu() when act == ACT_GELU, and store the result.
  // YOUR CODE HERE
  // ============================================================
}

ASR_API int launch_linear(const float* x, const float* w, const float* bias, float* y,
                          int M, int N, int K, int act, cudaStream_t stream) {
  dim3 block(TILE, TILE);
  dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
  linear_tiled_kernel<<<grid, block, 0, stream>>>(x, w, bias, y, M, N, K, act);
  RETURN_LAUNCH_ERROR();
}
