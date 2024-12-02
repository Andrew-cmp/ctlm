#include <iostream>
#include <cuda_runtime.h>

#define N 64   // A的行数
#define M 512  // A的列数，也是B的行数
#define K 512  // B的列数
__global__ void __maxnreg__(50) gemm_kernel(float* A, float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
int main() {
    // 主机端矩阵（初始化为一些数据）
    float *h_A = new float[N * M]; // 64x512
    float *h_B = new float[M * K]; // 512x512
    float *h_C = new float[N * K]; // 64x512

    // 初始化矩阵数据（可以根据需要填充数据）
    for (int i = 0; i < N * M; i++) {
        h_A[i] = static_cast<float>(i % 100);  // 仅为示例，实际应用中需要根据需要填充数据
    }

    for (int i = 0; i < M * K; i++) {
        h_B[i] = static_cast<float>((i + 1) % 100);  // 仅为示例
    }

    // 设备端矩阵指针
    float *d_A, *d_B, *d_C;
    
    // 分配设备内存
    cudaMalloc((void**)&d_A, N * M * sizeof(float));
    cudaMalloc((void**)&d_B, M * K * sizeof(float));
    cudaMalloc((void**)&d_C, N * K * sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, M * K * sizeof(float), cudaMemcpyHostToDevice);

    // 设置 CUDA 核函数的网格和块的维度
    dim3 threadsPerBlock(16, 16); // 每个块中有 16x16 个线程
    dim3 numBlocks((K + 15) / 16, (N + 15) / 16); // 计算块的数量

    cudaFuncSetCacheConfig(gemm_kernel, cudaFuncCachePreferEqual);
    // 调用 CUDA 核函数进行矩阵乘法
    gemm_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

    // 检查核函数调用是否有错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // 将结果从设备复制回主机
    cudaMemcpy(h_C, d_C, N * K * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果（只打印部分数据以避免输出过多）
    std::cout << "Result (first 5 elements of C):\n";
    for (int i = 0; i < 5; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 释放主机内存
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}