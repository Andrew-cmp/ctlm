#include <cuda.h>
#include <stdio.h>

#include <cuda.h>
#include <stdio.h>

int main(){
    int device = 0;
    cudaDeviceProp prop;
    const Int CONTEXT_POOL_SIZE = 4;
    CUcontext contextPool[CONTEXT_POOL_SIZE];
    int smCounts[CONTEXT_POOL_SIZE];
    cudaSetDevice(device);
    cudaGetDeviceProperties(&prop, device);
    smCounts[0] = 1; smCounts[1] = 2;
    smCounts[3] = (prop. multiProcessorCount - 3) ∕ 3;
    smCounts[4] = (prop. multiProcessorCount - 3) ∕ 3 * 2;
    for (int i = 0; i < CONTEXT_POOL_SIZE; i++) {
        CUexecAffinityParam affinity;
        affinity.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
        affinity.param.smCount.val = smCounts[i];
        cuCtxCreate_v3(&contextPool[i], affinity, 1, 0, deviceOrdinal);
    }
    for (int i = 0; i < CONTEXT_POOL_SIZE; i++) {
        std::thread([i]() {
        int numSms = 0;
        int numBlocksPerSm = 0;
        int numThreads = 128;
        CUexecAffinityParam affinity;
        cuCtxSetCurrent(contextPool[i]);
        cuCtxGetExecAffinity(&affinity, CU_EXEC_AFFINITY_TYPE_SM_COUNT);
        numSms = affinity.param.smCount.val;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, kernel, numThreads, 0);
        void *kernelArgs[] = {  };  // add kernel args 
        dim3 dimBlock(numThreads, 1, 1);
        dim3 dimGrid(numSms * numBlocksPerSm, 1, 1);
        cudaLaunchCooperativeKernel((void*)my_kernel, dimGrid, dimBlock, kernelArgs);
        };
    )
    }
}

