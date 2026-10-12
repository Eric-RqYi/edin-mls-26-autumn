// The C interface of libasr_kernels.so. Python (glm_asr_cuda/ops.py) and the
// playground tests call these functions. You do not need to change this file.
//
// Conventions:
//   - every pointer is a GPU (device) pointer to contiguous row-major data
//   - every float tensor is float32
//   - each function launches its kernel(s) on `stream` and returns the CUDA
//     error code of the launch (0 = OK); it does not wait for the GPU
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- Task 1: rotary position embedding, in place ----
// q: [batch, seq, num_q_heads, head_dim], k: [batch, seq, num_k_heads, head_dim]
// inv_freq: [rotary_dim / 2]
int launch_rope(float* q, float* k, const float* inv_freq, int batch, int seq_len,
                int num_q_heads, int num_k_heads, int head_dim, int rotary_dim,
                cudaStream_t stream);

// ---- Task 2: row-wise reductions. x, y: [rows, cols] ----
int launch_rmsnorm(const float* x, const float* weight, float* y, int rows, int cols,
                   float eps, cudaStream_t stream);
int launch_layernorm(const float* x, const float* weight, const float* bias, float* y,
                     int rows, int cols, float eps, cudaStream_t stream);
int launch_softmax(const float* x, float* y, int rows, int cols, cudaStream_t stream);

// ---- Task 3: linear layer  y = act(x @ w^T + bias) ----
// x: [M, K], w: [N, K], bias: [N] or NULL, y: [M, N], act: 0 = none, 1 = GELU
int launch_linear(const float* x, const float* w, const float* bias, float* y, int M,
                  int N, int K, int act, cudaStream_t stream);

// ---- Task 4: attention ----
// q, out: [batch, seq_q, num_heads, head_dim]
// k, v:   [batch, seq_k, num_kv_heads, head_dim]
int launch_attention(const float* q, const float* k, const float* v, float* out,
                     int batch, int num_heads, int num_kv_heads, int seq_q, int seq_k,
                     int head_dim, float scale, int causal, cudaStream_t stream);

// ---- Provided kernels (already complete, in provided.cu) ----
int launch_silu_mul(const float* gate, const float* up, float* y, int n,
                    cudaStream_t stream);
int launch_embedding(const int64_t* ids, const float* table, float* out, int num_ids,
                     int dim, cudaStream_t stream);
int launch_im2col_1d(const float* x, float* cols, int batch, int length, int channels,
                     int kernel_size, int stride, int padding, int out_length,
                     cudaStream_t stream);
const char* asr_error_string(int code);

#ifdef __cplusplus
}
#endif
