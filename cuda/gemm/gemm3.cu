#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#define ELE_TYPE float
#define BLOCK_DIM 8///线程块的维度大小，因为每个线程对应处理一个元素，所以也是对应要处理的C的维度大小，也是share——men的大小
template<uint32_t M,uint32_t N,uint32_t K>
__global__ void gemm_kernel(ELE_TYPE* A, ELE_TYPE* B,ELE_TYPE* C){
    // row指的是行上的坐标，而不是指的是第几行
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    // col指的是列上的坐标，而不是第几列
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    printf("blockIdx.x:%d blockDim.x:%d threadIdx.x:%d x:%d\n",blockIdx.x,blockDim.x , threadIdx.x,x);
    __shared__ ELE_TYPE Sa[BLOCK_DIM][BLOCK_DIM];
    __shared__ ELE_TYPE Sb[BLOCK_DIM][BLOCK_DIM];
    ELE_TYPE sum = 0;
    int num_k = (K + BLOCK_DIM -1)/BLOCK_DIM; /// 等同于ceil(k/BLOCK_DIM)
    for(int i = 0;i < num_k;i++){
        ///如果K或M不是BLOCK_DIM的整数倍，那么相当于对A填充0
        if(y < M && i*BLOCK_DIM+threadIdx.x < K){
            Sa[threadIdx.y][threadIdx.x] = A[y*K+i*BLOCK_DIM+threadIdx.x];
        }
        else {
            Sa[threadIdx.y][threadIdx.x] = 6;
        }
        if((BLOCK_DIM*i+threadIdx.y)<K && x < N){
            Sb[threadIdx.y][threadIdx.x] = B[(BLOCK_DIM*i+threadIdx.y)*N+x];
        }
        else {
            Sb[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        for(int j = 0; j < BLOCK_DIM;j++){
            sum += Sa[threadIdx.y][j] * Sb[j][threadIdx.x];
        }
        __syncthreads();
    }
    if(x < N && y < M){
        C[y*N + x] = sum;
    }
    
}

int main(){

    ///这个不加还不能当cuda模板参数
    ///要求必须是编译时期已知且运行时不会变的constant
    constexpr uint32_t N = 16;
    constexpr uint32_t M = 32;
    constexpr uint32_t K = 16;
    int size_A = M*K*sizeof(ELE_TYPE);
    int size_B = K*N*sizeof(ELE_TYPE);
    int size_C = M*N*sizeof(ELE_TYPE);
    ELE_TYPE * h_a =(ELE_TYPE*)malloc(size_A);
    ELE_TYPE * h_b =(ELE_TYPE*)malloc(size_B);
    ELE_TYPE * h_c =(ELE_TYPE*)malloc(size_C);
    for (int i = 0; i < M * K; ++i) h_a[i] = 1.0f;
    for (int i = 0; i < K * N; ++i) h_b[i] = 1.0f;
    ELE_TYPE *d_a, *d_b, *d_c;
    cudaMalloc(&d_a,M*K*sizeof(ELE_TYPE));
    cudaMalloc(&d_b,K*N*sizeof(ELE_TYPE));
    cudaMalloc(&d_c,M*N*sizeof(ELE_TYPE));
    
    cudaMemcpy(d_a,h_a,size_A,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size_B,cudaMemcpyHostToDevice);
    dim3 blockDim(8,8);
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