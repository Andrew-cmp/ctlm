
// #include <cuda_runtime.h>
// #include <stdio.h>
// __global__ void simple_kernel() {
//     extern __share__[]
//     while (true) {}
// }

// int main() {
    
//     simple_kernel<<<1, 1>>>();
//     cudaDeviceSynchronize();
// }

#include <cuda_runtime.h>
#include <stdio.h>
#define TILE_SIZE 16
#define GETATTR(func, device) \
    int func##device;\
    cudaDeviceGetAttribute(&func##device, func, device);\
    printf(#func":%d \n",func##device);

// Kernel for GEMM (General Matrix Multiply): C = alpha * A * B + beta * C
__global__ void gemmKernel(float *A) {
    __shared__ float AAAA[10];
    AAAA[0] += 1;
    while (true) {}
}

// Host code to launch the GEMM kernel
void gemm(float *h_A, float *h_B, float *h_C, int M, int N, int K, float alpha, float beta) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Copy matrices A, B, and C to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

    // Define the block and grid dimensions
    //dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    //dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the GEMM kernel
    gemmKernel<<<1, 1,100*sizeof(float)>>>(d_A);

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
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Allocate host memory for matrices A, B, and C
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));



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


