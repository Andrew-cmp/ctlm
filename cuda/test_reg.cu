#include <cuda_runtime.h>
#include <iostream>
__global__ void registerBandwidthTest(float* input,float *output, int iterations) {
    float r0, r1, r2 , r3 , r4; // 使用多个寄存器
    r0 = input[0];
    r1 = input[1];
    r2 = input[2];
    r3 = input[3];
    r4 = input[4];
    for (int i = 0; i < iterations; i++) {
        r1 = r1 * r2 + r3;
        r2 = r2 * r3 + r4;
        r3 = r3 * r4 + r1;
        r4 = r4 * r1 + r2;
    }
    output[threadIdx.x] = r1 + r2 + r3 + r4;
}
__global__ void registerLatencyTest(float *output, int iterations) {
    float r = 1.0; 
    for (int i = 0; i < iterations; i++) {
        r = r + 1.0;  // 重复同一个操作确保没有数据依赖
    }
    output[threadIdx.x] = r;
}

int main() {
    float *h_input, *h_output;
    h_input = (float*)malloc(sizeof(float)*256);
    h_output = (float*)malloc(sizeof(float)*256);
    for(int i = 0 ;i < 256;i++){
        h_input[i] = 0;
    }
    float *d_output;
    float *d_input;
    //cudaError_t err = cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);

    cudaMalloc(&d_output, sizeof(float) * 256);
    cudaMalloc(&d_input, sizeof(float) * 256);

    cudaMemcpy(d_input, h_input, 256, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    registerBandwidthTest<<<1, 1>>>(d_input,d_output, 10000);  // 调整迭代次数
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Elapsed time: " << milliseconds << " ms\n";

    cudaFree(d_output);
    cudaFree(d_input);
    free(h_input);
    free(h_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
