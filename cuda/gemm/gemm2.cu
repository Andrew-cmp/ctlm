#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#define ELE_TYPE float
#define TILE_M 2
#define TILE_N 2
template<uint32_t M,uint32_t N,uint32_t K>
__global__ void gemm_kernel(ELE_TYPE* A, ELE_TYPE* B,ELE_TYPE* C){
    // row指的是行上的坐标，而不是指的是第几行
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    // col指的是列上的坐标，而不是第几列
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    __shared__ ELE_TYPE Sa[M/TILE_M][K];
    __shared__ ELE_TYPE Sb[K][N/TILE_N];
    ELE_TYPE sum = 0;
    //先从global mem移动到share mem
    __syncthreads();
    for(int i = threadIdx.x;i < K;i+=TILE_M){
        Sa[threadIdx.y][i] = A[y*K+i];
    } 
    for(int i = threadIdx.y;i < K;i += TILE_N){
        Sb[i][threadIdx.x] = B[i*N+x];
    }
    __syncthreads();
    for(int s = 0;s < K;s++ ){
        sum += Sa[threadIdx.y][s] * Sb[s][threadIdx.x];
    }
    C[y*N+x] = sum;

    // __syncthreads();
    // for(int i = threadIdx.y;i < K;i+=TILE_M){
    //     Sa[threadIdx.x][i] = A[x*K+i];
    // } 
    // for(int i = threadIdx.x;i < K;i += TILE_N){
    //     Sb[i][threadIdx.y] = B[i*N+y];
    // }
    // __syncthreads();
    // for(int s = 0;s < K;s++ ){
    //     sum += Sa[threadIdx.x][s] * Sb[s][threadIdx.y];
    // }
    // C[x*N+y] = sum;
    
}

int main(){

    ///这个不加还不能当cuda模板参数
    ///要求必须是编译时期已知且运行时不会变的constant
    constexpr uint32_t N = 16;
    constexpr uint32_t M = 32;
    constexpr uint32_t K = 8;
    int size_A = M*K*sizeof(ELE_TYPE);
    int size_B = K*N*sizeof(ELE_TYPE);
    int size_C = M*N*sizeof(ELE_TYPE);
    ELE_TYPE * h_a =(ELE_TYPE*)malloc(size_A);
    ELE_TYPE * h_b =(ELE_TYPE*)malloc(size_B);
    ELE_TYPE * h_c =(ELE_TYPE*)malloc(size_C);
    for (int i = 0; i < M * K; ++i) h_a[i] = 2.0f;
    for (int i = 0; i < K * N; ++i) h_b[i] = 2.0f;
    ELE_TYPE *d_a, *d_b, *d_c;
    cudaMalloc(&d_a,M*K*sizeof(ELE_TYPE));
    cudaMalloc(&d_b,K*N*sizeof(ELE_TYPE));
    cudaMalloc(&d_c,M*N*sizeof(ELE_TYPE));
    
    cudaMemcpy(d_a,h_a,size_A,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size_B,cudaMemcpyHostToDevice);
    
    dim3 blockDim(4,4);
    dim3 gridDim((N+threadperblock.x-1)/threadperblock.x,
                      (M+threadperblock.y-1)/threadperblock.y );
    //草，大模型给的代码，下面的GridDim和blockDim位置对调了
    //gemm_kernel<M,N,K><<<blockDim,gridDim>>>(d_a,d_b,d_c);
    gemm_kernel<M,N,K><<<gridDim,blockDim>>>(d_a,d_b,d_c);

    cudaMemcpy(h_c,d_c,size_C,cudaMemcpyDeviceToHost);

    for(int i = 0;i < M;++i){
        for(int j = 0;j < N;j++){
            printf("%0.1f ",*(h_c+i*N+j));
        }
        printf("\n");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
}