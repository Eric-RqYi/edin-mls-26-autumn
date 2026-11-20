#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "cuda_probe.fatbin.c"
extern void __device_stub__Z5probePf(float *);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
void __device_stub__Z5probePf(float *__par0){__cudaLaunchPrologue(1);__cudaSetupArgSimple(__par0, 0UL);__cudaLaunch(((char *)((void ( *)(float *))probe)));}
# 2 "/home/s2955206/edin-mls-26-autumn/hw1-asr/experiments/reuse_boundaries/results/20261118T225304Z/attention/preflight/cuda_probe.cu"
void probe( float *__cuda_0)
# 2 "/home/s2955206/edin-mls-26-autumn/hw1-asr/experiments/reuse_boundaries/results/20261118T225304Z/attention/preflight/cuda_probe.cu"
{__device_stub__Z5probePf( __cuda_0); }
# 1 "/home/s2955206/edin-mls-26-autumn/hw1-asr/experiments/reuse_boundaries/results/20261118T225304Z/attention/preflight/keep/cuda_probe.cudafe1.stub.c"
static void __nv_cudaEntityRegisterCallback( void **__T0) {  __nv_dummy_param_ref(__T0); __nv_save_fatbinhandle_for_managed_rt(__T0); __cudaRegisterEntry(__T0, ((void ( *)(float *))probe), _Z5probePf, (-1)); }
static void __sti____cudaRegisterAll(void) {  __cudaRegisterBinary(__nv_cudaEntityRegisterCallback);  }

#pragma GCC diagnostic pop
