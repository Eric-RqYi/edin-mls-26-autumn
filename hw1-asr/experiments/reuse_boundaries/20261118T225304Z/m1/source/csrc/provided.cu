// Kernels that are already complete. You do not need to change this file, but it
// is worth reading: each one is a small, finished example of the patterns the
// tasks use.
#include <cstdint>

#include "asr_kernels.h"
#include "common.cuh"

constexpr int PROVIDED_THREADS = 256;

// SwiGLU gate of the text decoder MLP: y = silu(gate) * up, silu(g) = g / (1 + exp(-g)).
// Element-wise: one thread per element, like vector_add.
__global__ void silu_mul_kernel(const float* gate, const float* up, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float g = gate[i];
    y[i] = g / (1.0f + expf(-g)) * up[i];
  }
}

// Token embedding: out[t, :] = table[ids[t], :]. One thread per output element.
__global__ void embedding_kernel(const int64_t* ids, const float* table, float* out,
                                 int num_ids, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_ids * dim) {
    int t = i / dim;
    int d = i % dim;
    out[i] = table[ids[t] * dim + d];
  }
}

// im2col for a 1D convolution, so the convolution becomes one GEMM (task 3).
// x:    [batch, length, channels]           (time-major, one row per time step)
// cols: [batch * out_length, kernel_size * channels]
// Row (b, t) of cols holds the input window of output step t:
//     cols[(b, t), kk * channels + c] = x[b, t * stride + kk - padding, c]   (0 outside x)
// The convolution is then  y = cols * w^T + bias  with w: [out_channels, kernel_size * channels].
__global__ void im2col_1d_kernel(const float* x, float* cols, int batch, int length,
                                 int channels, int kernel_size, int stride, int padding,
                                 int out_length) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int row_len = kernel_size * channels;
  if (i >= batch * out_length * row_len) return;
  int row = i / row_len;
  int within = i % row_len;
  int kk = within / channels;
  int c = within % channels;
  int b = row / out_length;
  int t = row % out_length;
  int src_t = t * stride + kk - padding;
  bool inside = src_t >= 0 && src_t < length;
  cols[i] = inside ? x[(static_cast<size_t>(b) * length + src_t) * channels + c] : 0.0f;
}

ASR_API int launch_silu_mul(const float* gate, const float* up, float* y, int n,
                            cudaStream_t stream) {
  silu_mul_kernel<<<ceil_div(n, PROVIDED_THREADS), PROVIDED_THREADS, 0, stream>>>(gate, up, y, n);
  RETURN_LAUNCH_ERROR();
}

ASR_API int launch_embedding(const int64_t* ids, const float* table, float* out, int num_ids,
                             int dim, cudaStream_t stream) {
  int n = num_ids * dim;
  embedding_kernel<<<ceil_div(n, PROVIDED_THREADS), PROVIDED_THREADS, 0, stream>>>(
      ids, table, out, num_ids, dim);
  RETURN_LAUNCH_ERROR();
}

ASR_API int launch_im2col_1d(const float* x, float* cols, int batch, int length, int channels,
                             int kernel_size, int stride, int padding, int out_length,
                             cudaStream_t stream) {
  int n = batch * out_length * kernel_size * channels;
  im2col_1d_kernel<<<ceil_div(n, PROVIDED_THREADS), PROVIDED_THREADS, 0, stream>>>(
      x, cols, batch, length, channels, kernel_size, stride, padding, out_length);
  RETURN_LAUNCH_ERROR();
}

ASR_API const char* asr_error_string(int code) {
  return cudaGetErrorString(static_cast<cudaError_t>(code));
}
