#include <cuda_runtime.h>
#include <stdio.h>
extern "C" __global__ void __launch_bounds__(12) main_kernel0(signed char* __restrict__ c, signed char* __restrict__ a) {
    volatile int i =c[blockIdx.x];
    c[(((((int)blockIdx.x) * 12) + (((int)threadIdx.x) * 4)) + ((int)threadIdx.y))] = i+((signed char)((5.000000e-01f) < a[(((((int)blockIdx.x) * 12) + (((int)threadIdx.x) * 4)) + ((int)threadIdx.y))]));
   
  }
  
  int main(){
    int size = 24 * sizeof(signed char);
    signed char *h_c = (signed char*)malloc(size);
    signed char *h_a = (signed char*)malloc(size);
    for(int i = 0; i < 24; i++){
      h_a[i] = i;
      h_c[i] = 0;
    }

    signed char *d_c, *d_a;
    cudaMalloc((void**)&d_c, size);
    cudaMalloc((void**)&d_a, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(2);
    dim3 dimGrid(3,4);
    main_kernel0<<<dimGrid, dimBlock>>>(d_c, d_a);
    cudaError_t cuda_error= cudaGetLastError();
    if(cuda_error != cudaSuccess){
      printf("CUDA error: %s\n", cudaGetErrorString(cuda_error));
    }
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 24; i++){
      printf("%d ", h_c[i]);
    }

    // 清理资源
    free(h_c);
    free(h_a);
    cudaFree(d_c);
    cudaFree(d_a);
  }
