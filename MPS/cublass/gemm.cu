#include <cuda_runtime.h>
#include <stdio.h>
#define COM_TYPE float32_t
// Kernel for GEMM (General Matrix Multiply): C = alpha * A * B + beta * C
__global__ void gemm_fp32_v1(uint32_t M, uint32_t N, uint32_t K,  //
    float alpha, float beta,      //
    const float *__restrict__ A,  //
    const float *__restrict__ B,  //
    float *__restrict__ C) {
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= M || x >= N) {
    return;
    }
    float sum = 0.0f;
    for (uint32_t k = 0; k < K; k++) {
    sum += A[y * K + k] * B[k * N + x];
    }
    C[y * N + x] = alpha * sum + C[y * N + x] * beta;
}

// Host code to launch the GEMM kernel
void gemm(COM_TYPE *h_A, COM_TYPE *h_B, COM_TYPE *h_C, int M, int N, int K, COM_TYPE alpha, COM_TYPE beta) {
    COM_TYPE *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(COM_TYPE);
    size_t size_B = K * N * sizeof(COM_TYPE);
    size_t size_C = M * N * sizeof(COM_TYPE);

    // Allocate device memory
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Copy matrices A, B, and C to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // Define the block and grid dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the GEMM kernel
    gemmKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K, alpha, beta);

    // Copy the result matrix C back to the host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


int main() {
    // Matrix dimensions
    int M = 4096; // Number of rows of A and C
    int N = 4096; // Number of columns of B and C
    int K = 4096; // Number of columns of A and rows of B

    // Scalars
    COM_TYPE alpha = 1.0f;
    COM_TYPE beta = 0.0f;
    
    // Allocate host memory for matrices A, B, and C
    COM_TYPE *h_A = (COM_TYPE*)malloc(M * K * sizeof(COM_TYPE));
    COM_TYPE *h_B = (COM_TYPE*)malloc(K * N * sizeof(COM_TYPE));
    COM_TYPE *h_C = (COM_TYPE*)malloc(M * N * sizeof(COM_TYPE));



    // Initialize matrices A and B with some values
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f;
    for (int i = 0; i < M * N; ++i) h_C[i] = 0.0f;
    // Perform GEMM operation
    gemm(h_A, h_B, h_C, M, N, K, alpha, beta);
    // Print a small portion of the result matrix C
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            printf("%f ", h_C[i * N + j]);
        }
        printf("\n");
    }
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}


