// Shared helpers for all kernels. You do not need to change this file.
#pragma once
#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr unsigned FULL_MASK = 0xffffffffu;  // "all 32 lanes take part", for __shfl_*_sync

// Number of blocks of size b needed to cover a items.
__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Activation codes for the GEMM epilogue (task 3).
constexpr int ACT_NONE = 0;
constexpr int ACT_GELU = 1;

// Exact GELU, as used by the GLM-ASR audio encoder and projector.
__device__ __forceinline__ float gelu(float x) {
  return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}

// Every launcher is a plain C function, so Python can call it through ctypes.
// It returns the CUDA error code of the launch (0 means no error).
#define ASR_API extern "C" __attribute__((visibility("default")))
#define RETURN_LAUNCH_ERROR() return static_cast<int>(cudaGetLastError())
