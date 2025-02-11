
#include <cuda_runtime.h>
#include <stdio.h>
#define N 1024
#define BLOCK_SIZE 256
///https://zhuanlan.zhihu.com/p/688610091
// block reduce 针对block内的所有数据进行规约，block间的数据交给host端
//最简单的树形规约，规约轮次为logBLOCK_SIZE
__global__ void reduce_v1(float *g_idata, float * g_odata){
    __share__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __synchreads();
    for(unsigned int s = 1; s<blockDim.x;s *= 2){
        if(tid%(s<<1) == 0){
            sdata[tid] += sdata[tid+s];
        }
    }
    __synchreads();
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce_v2(float *g_idata, float * g_odata){
    __share__ float sdata[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockDim.x*blockIdx.x + threadIdx.x;
    sdata[tid] = g_idata[i];
    __synchreads();
    for(unsigned int s = 1; s<blockDim.x;s *= 2){
        ///thread 负责的第一个元素的索引：
        //其计算方式综合图和 stride得到
        int index = 2*s*tid;
        for(index < blockDim.x){
            sdata[index] += sdata[index+s]
        }
    }
    __synchreads();
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


int main(){


    float A[N], B[N], C[N];
    for(int i = 0; i < N;i++){
        A[i] = i;
        B[i] = i;
        C[i] = 0;
    }
    int * d_A,* d_B,* d_C;
    int size_A = sizeof(float)*N;
    int size_B = sizeof(float)*N;
    int size_C = sizeof(float)*N;
    cudaMalloc(&d_A,sizeof(float)*N);
    cudaMalloc(&d_B,sizeof(float)*N);
    cudaMalloc(&d_C,sizeof(float)*N);

    cudaMemcpy(d_A,A,size_A,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,size_B,cudaMemcpyHostToDevice);


    dim block(BLOCK_SIZE);
    dim grid(N/BLOCK_SIZE);

}
