
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
extern "C" __global__ void __launch_bounds__(1024) main_kernel0(float* __restrict__ B, float* __restrict__ A, float* __restrict__ C) {
  float B_local[1024];
  float A_local[1024];
  float C_local[1];
  for (int ax1 = 0; ax1 < 1024; ++ax1) {
    B_local[ax1] = B[(((ax1 * 1024) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x))];
  }
  for (int ax2 = 0; ax2 < 1024; ++ax2) {
    A_local[ax2] = A[(((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 1024)) + ax2)];
  }
  for (int k = 0; k < 1024; ++k) {
    if (k == 0) {
      C_local[0] = 0.000000e+00f;
    }
    C_local[0] = (C_local[0] + (A_local[k] * B_local[k]));
  }
  C[((((((int)blockIdx.y) * 32768) + (((int)threadIdx.y) * 1024)) + (((int)blockIdx.x) * 32)) + ((int)threadIdx.x))] = C_local[0];
}

