
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
extern "C" __global__ void __maxnreg__(50) main_kernel0(float* __restrict__ p0, float* __restrict__ p1, float* __restrict__ T_batch_matmul_NN) {
  float T_batch_matmul_NN_local[2048];

  __shared__ float p0_shared[8];
  __shared__ float p1_shared[256];
  for (int k_0 = 0; k_0 < 512; ++k_0) {
    __syncthreads();
    for (int ax0_ax1_ax2_fused = 0; ax0_ax1_ax2_fused < 8; ++ax0_ax1_ax2_fused) {
      p0_shared[ax0_ax1_ax2_fused] = p0[((((((int)blockIdx.x) >> 1) * 4096) + (ax0_ax1_ax2_fused * 512)) + k_0)];
    }
    for (int ax0_ax1_ax2_fused_1 = 0; ax0_ax1_ax2_fused_1 < 256; ++ax0_ax1_ax2_fused_1) {
      p1_shared[ax0_ax1_ax2_fused_1] = p1[(((((((int)blockIdx.x) >> 4) * 262144) + (k_0 * 512)) + ((((int)blockIdx.x) & 1) * 256)) + ax0_ax1_ax2_fused_1)];
    }
    __syncthreads();
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      for (int j_3 = 0; j_3 < 2; ++j_3) {
        for (int i_4 = 0; i_4 < 4; ++i_4) {
          for (int j_4 = 0; j_4 < 32; ++j_4) {
            if (k_0 == 0) {
              T_batch_matmul_NN_local[((((i_3 * 256) + (i_4 * 64)) + (j_3 * 32)) + j_4)] = 0.000000e+00f;
              T_batch_matmul_NN_local[(((((i_3 * 256) + (i_4 * 64)) + (j_3 * 32)) + j_4) + 512)] = 0.000000e+00f;
              T_batch_matmul_NN_local[(((((i_3 * 256) + (i_4 * 64)) + (j_3 * 32)) + j_4) + 1024)] = 0.000000e+00f;
              T_batch_matmul_NN_local[(((((i_3 * 256) + (i_4 * 64)) + (j_3 * 32)) + j_4) + 1536)] = 0.000000e+00f;
            }
            T_batch_matmul_NN_local[((((i_3 * 256) + (i_4 * 64)) + (j_3 * 32)) + j_4)] = (T_batch_matmul_NN_local[((((i_3 * 256) + (i_4 * 64)) + (j_3 * 32)) + j_4)] + (p0_shared[((i_3 * 4) + i_4)] * p1_shared[((j_3 * 32) + j_4)]));
            T_batch_matmul_NN_local[(((((i_3 * 256) + (i_4 * 64)) + (j_3 * 32)) + j_4) + 512)] = (T_batch_matmul_NN_local[(((((i_3 * 256) + (i_4 * 64)) + (j_3 * 32)) + j_4) + 512)] + (p0_shared[((i_3 * 4) + i_4)] * p1_shared[(((j_3 * 32) + j_4) + 64)]));
            T_batch_matmul_NN_local[(((((i_3 * 256) + (i_4 * 64)) + (j_3 * 32)) + j_4) + 1024)] = (T_batch_matmul_NN_local[(((((i_3 * 256) + (i_4 * 64)) + (j_3 * 32)) + j_4) + 1024)] + (p0_shared[((i_3 * 4) + i_4)] * p1_shared[(((j_3 * 32) + j_4) + 128)]));
            T_batch_matmul_NN_local[(((((i_3 * 256) + (i_4 * 64)) + (j_3 * 32)) + j_4) + 1536)] = (T_batch_matmul_NN_local[(((((i_3 * 256) + (i_4 * 64)) + (j_3 * 32)) + j_4) + 1536)] + (p0_shared[((i_3 * 4) + i_4)] * p1_shared[(((j_3 * 32) + j_4) + 192)]));
          }
        }
      }
    }
  }
  for (int ax1 = 0; ax1 < 8; ++ax1) {
    for (int ax2 = 0; ax2 < 64; ++ax2) {
      T_batch_matmul_NN[(((((((int)blockIdx.x) >> 1) * 4096) + (ax1 * 512)) + ((((int)blockIdx.x) & 1) * 256)) + ax2)] = T_batch_matmul_NN_local[((ax1 * 64) + ax2)];
      T_batch_matmul_NN[((((((((int)blockIdx.x) >> 1) * 4096) + (ax1 * 512)) + ((((int)blockIdx.x) & 1) * 256)) + ax2) + 64)] = T_batch_matmul_NN_local[(((ax1 * 64) + ax2) + 512)];
      T_batch_matmul_NN[((((((((int)blockIdx.x) >> 1) * 4096) + (ax1 * 512)) + ((((int)blockIdx.x) & 1) * 256)) + ax2) + 128)] = T_batch_matmul_NN_local[(((ax1 * 64) + ax2) + 1024)];
      T_batch_matmul_NN[((((((((int)blockIdx.x) >> 1) * 4096) + (ax1 * 512)) + ((((int)blockIdx.x) & 1) * 256)) + ax2) + 192)] = T_batch_matmul_NN_local[(((ax1 * 64) + ax2) + 1536)];
    }
  }
}

