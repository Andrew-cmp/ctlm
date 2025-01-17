#include <cuda_runtime.h>
#include <stdio.h>

#define GETATTR(func, device) \
    int func##device;\
    cudaDeviceGetAttribute(&func##device, func, device);\
    printf(#func":%d \n",func##device);

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device name: %s\n" , prop.name) ;
    printf("Compute capability: %d\n"  , prop.minor) ;
    

    GETATTR(cudaDevAttrMultiProcessorCount,0);
    cudaError_t err = cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    if (err != cudaSuccess) {
       printf("Error setting shared memory config: %s",cudaGetErrorString(err)) ;
    }   cudaFuncCache pCacheConfig;
    cudaError_t err3 = cudaDeviceGetCacheConfig(&pCacheConfig);
    if (err != cudaSuccess) {
        printf("Error setting shared memory config: %s",cudaGetErrorString(err3)) ;
    }
    printf("%d\n",pCacheConfig);

    printf("sharedMemPerMultiprocessor:%d\n", prop.sharedMemPerMultiprocessor );
    GETATTR(cudaDevAttrMaxSharedMemoryPerBlock,0);
    GETATTR(cudaDevAttrWarpSize,0);
    GETATTR(cudaDevAttrComputeCapabilityMajor,0);
    GETATTR(cudaDevAttrMaxBlockDimX,0);
    GETATTR(cudaDevAttrMaxBlockDimY,0);
    GETATTR(cudaDevAttrMaxBlockDimZ,0);
    GETATTR(cudaDevAttrMaxGridDimX,0);
    GETATTR(cudaDevAttrMaxGridDimY,0);
    GETATTR(cudaDevAttrMaxGridDimZ,0);
    GETATTR(cudaDevAttrTotalConstantMemory,0);
    GETATTR(cudaDevAttrMaxPitch,0);
}