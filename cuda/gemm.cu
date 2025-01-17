#include <cuda_runtime.h>
#include <stdio.h>
#define TILE_SIZE 16
#define GETATTR(func, device) \
    int func##device;\
    cudaDeviceGetAttribute(&func##device, func, device);\
    printf(#func":%d \n",func##device);

// Kernel for GEMM (General Matrix Multiply): C = alpha * A * B + beta * C
__global__ void gemmKernel(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta) {
    // Define shared memory for tiles of A and B
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    // Calculate row and column index of the element
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float C_value = 0.0f;

    // Loop over the tiles of A and B that are required to compute C[row][col]
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Collaborative loading of A and B tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Perform the computation for the tile
        for (int i = 0; i < TILE_SIZE; ++i) {
            C_value += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to the output matrix C
    if (row < M && col < N) {
        C[row * N + col] = alpha * C_value + beta * C[row * N + col];
    }
}

// Host code to launch the GEMM kernel
void gemm(float *h_A, float *h_B, float *h_C, int M, int N, int K, float alpha, float beta) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    cudaError_t err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    if (err != cudaSuccess) {
       printf("Error setting shared memory config: %s",cudaGetErrorString(err)) ;
    }
    cudaFuncCache pCacheConfig;
    cudaError_t err3 = cudaDeviceGetCacheConfig(&pCacheConfig);
    if (err != cudaSuccess) {
        printf("Error setting shared memory config: %s",cudaGetErrorString(err3)) ;
     }
    printf("%d\n",pCacheConfig);
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
    cudaFuncAttributes attr;
    cudaError_t err2 = cudaFuncGetAttributes(&attr, "gemmKernel");
    printf("Shared memory size for myKernel: %d\n", attr.sharedSizeBytes);
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
    for(int i = 0; i<1;i++){
        // Perform GEMM operation
        gemm(h_A, h_B, h_C, M, N, K, alpha, beta);
        // Print a small portion of the result matrix C
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                printf("%f ", h_C[i * N + j]);
            }
            printf("\n");
        }
    }
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}


