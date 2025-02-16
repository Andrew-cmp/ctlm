#include <cuda_runtime.h>
#include <stdio.h>

#define GETATTR(func, device) \
    int func##device;\
    cudaDeviceGetAttribute(&func##device, func, device);\
    printf(#func":%d \n",func##device);

int main() {

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("系统检测到 %d 个CUDA设备\n", deviceCount);

    #pragma unroll
    for (int dev = 0; dev < deviceCount; dev++){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        printf("Device name: %s\n" , prop.name) ;
        printf("Compute capability: %d\n"  , prop.minor) ;
        GETATTR(cudaDevAttrMultiProcessorCount,dev);
        cudaError_t err = cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
        if (err != cudaSuccess) {
           printf("Error setting shared memory config: %s",cudaGetErrorString(err)) ;
        }   cudaFuncCache pCacheConfig;
        cudaError_t err3 = cudaDeviceGetCacheConfig(&pCacheConfig);
        if (err != cudaSuccess) {
            printf("Error setting shared memory config: %s",cudaGetErrorString(err3)) ;
        }
        printf("sharedMemPerMultiprocessor:%d\n", prop.sharedMemPerMultiprocessor );
        GETATTR(cudaDevAttrMaxSharedMemoryPerBlock,dev);
        GETATTR(cudaDevAttrWarpSize,dev);
        GETATTR(cudaDevAttrComputeCapabilityMajor,dev);
        GETATTR(cudaDevAttrMaxBlockDimX,dev);
        GETATTR(cudaDevAttrMaxBlockDimY,dev);
        GETATTR(cudaDevAttrMaxBlockDimZ,dev);
        GETATTR(cudaDevAttrMaxGridDimX,dev);
        GETATTR(cudaDevAttrMaxGridDimY,dev);
        GETATTR(cudaDevAttrMaxGridDimZ,dev);
        GETATTR(cudaDevAttrTotalConstantMemory,dev);
        GETATTR(cudaDevAttrMaxPitch,dev);
        GETATTR(cudaDevAttrClockRate,dev);
        GETATTR(cudaDevAttrGlobalMemoryBusWidth,dev);

    }




}
