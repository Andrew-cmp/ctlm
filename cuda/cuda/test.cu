#include <cuda_runtime.h>
#include <stdio.h>
#define CHECK(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}
__global__ void testkernel(){
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;
    int iz=threadIdx.z+blockIdx.z*blockDim.z;
    printf("thread_id(%d,%d,%d) block_id(%d,%d,%d) coordinate(%d,%d,%d)\n",threadIdx.x,threadIdx.y,threadIdx.z,
            blockIdx.x,blockIdx.y,blockIdx.z,ix,iy,iz);
}

int main(){



    int device = 0;
    cudaSetDevice(device);
    dim3 grid(1,1,1);
    dim3 block(256,2,2);
    testkernel<<<grid,block>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Possibly: exit(-1) if program cannot continue....
    } 
    else{
        printf("1");
    }
    

}