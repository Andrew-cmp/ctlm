#include <cuda_runtime.h>
#include <iostream>
#include<cuda.h>

const char* cuGetErrorString(CUresult err) {
    const char* errorStr;
    cuGetErrorString(err, &errorStr);  // 获取错误字符串
    return errorStr;
}
void gemm_cuda(float* A, float* B, float* C, int M, int N, int K) {
    // CUDA 初始化
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::cerr << "CUDA set device failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // 计算 grid 和 block 尺寸
    dim3 block(16, 16);  // 线程块大小 16x16
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // 将数据从主机传输到设备
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 加载内核函数
    CUfunction gemm_func;
    CUmodule module;
    CUresult res = cuModuleLoad(&module, "gemm.ptx"); // 加载 PTX 文件
    if (res != CUDA_SUCCESS) {
        std::cerr << "Failed to load module" << std::endl;
        return;
    }

    // 获取内核函数
    res = cuModuleGetFunction(&gemm_func, module, "_Z11gemm_kernelPfS_S_");
    if (res != CUDA_SUCCESS) {
        std::cerr << "Failed to get function" << std::endl;
        std::cerr <<cuGetErrorString(res)<< std::endl;
        return;
    }

    // 设置内核参数
    void* kernel_params[] = {
        (void*)&d_A,
        (void*)&d_B,
        (void*)&d_C,
        (void*)&M,
        (void*)&N,
        (void*)&K
    };

    // 启动内核
    res = cuLaunchKernel(gemm_func,
                         grid.x, grid.y, 1,        // Grid size
                         block.x, block.y, 1,      // Block size
                         0,                         // Shared memory size
                         0,                         // Stream
                         kernel_params, nullptr);  // Parameters

    if (res != CUDA_SUCCESS) {
        std::cerr << "Failed to launch kernel" << std::endl;
        return;
    }

    // 等待内核完成
    cudaDeviceSynchronize();

    // 将结果从设备复制回主机
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 清理资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int M = 1024, N = 1024, K = 1024;
    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[M * N];

    // 初始化 A 和 B
    for (int i = 0; i < M * K; ++i) A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) B[i] = 1.0f;

    gemm_cuda(A, B, C, M, N, K);

    // 输出 C 的部分结果
    std::cout << "C[0][0]: " << C[0] << std::endl;

    // 清理主机内存
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
