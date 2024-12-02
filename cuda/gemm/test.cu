#include <cuda_runtime.h>
#include <stdio.h>
#define ELE_TYPE float

__global__ void gemm_kernel(ELE_TYPE* A, ELE_TYPE* B,ELE_TYPE* C,int M, int N, int K){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    printf("blockIdx.x:%d blockDim.x:%d threadIdx.x:%d x:%d\n",blockIdx.x,blockDim.x , threadIdx.x,x);
    
    printf("blockIdx.y:%d blockDim.y:%d threadIdx.y:%d y:%d\n",blockIdx.y,blockDim.y , threadIdx.y,y);
    
}

int main(){

    int N = 6;
    int M = 6;
    int K = 6;
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
    dim3 threadperblock(2,2);
    dim3 blockpergrid((N+threadperblock.x-1)/threadperblock.x,
                      (M+threadperblock.y-1)/threadperblock.y );

    gemm_kernel<<<threadperblock,blockpergrid>>>(d_a,d_b,d_c,M,N,K);

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