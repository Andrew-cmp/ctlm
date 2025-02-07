
#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __maxnreg__(1) main_kernel0(float* __restrict__ A, float* __restrict__ B) {
  __shared__ float A_shared[130];
  for (int ax0_0 = 0; ax0_0 < 2; ++ax0_0) {
    if (((ax0_0 * 64) + (((int)threadIdx.x) >> 1)) < 65) {
      A_shared[((ax0_0 * 128) + ((int)threadIdx.x))] = A[(((((int)blockIdx.x) * 128) + (ax0_0 * 128)) + ((int)threadIdx.x))];
    }
  }
  __syncthreads();
  B[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = ((A_shared[((int)threadIdx.x)] + A_shared[(((int)threadIdx.x) + 1)]) + A_shared[(((int)threadIdx.x) + 2)]);

}

