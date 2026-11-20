#include <cuda_runtime.h>
__global__ void probe(float* value) { if (threadIdx.x == 0) value[0] = 1.0f; }
