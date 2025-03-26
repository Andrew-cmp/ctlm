template<unsigned block_size>
__device__ __forceinline__  void warpReduce_v5(float sum){
    if (blockSize >= 32)
        sum += __shfl_down_sync(0xffffffff, sum, 16);  // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16)
        sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)
        sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)
        sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)
        sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
    return sum;
}
//算出一个block的和，此block的和存在g_odata[blockidx.x]中
__global__ void BlockSumKernel(float * g_idata, float * g_odata){
    unsigned int tid = threadIdx.x;
    unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
    float sum = 0;
    sum = g_idata[gid];
    int lid = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    int num_warps = block_size/WARP_SIZE;
    sum = warpReduce_v5<block_size>(sum);

    __shared__ float shared[WARP_SIZE];
    if(lid == 0){
        shared[wid] = sum;
    }
    __syncthreads();
    sum = (tid<num_warps)? shared[lid] : 0;
    sum = warpReduce_v5<block_size>(sum);
    if(tid == 0) g_odata[blockIdx.x] = sum;
}
__global__ void ScanPartKernel(const int* g_idata, int *g_odata, int *g_buffer, size_t part_num){
    extern __shared__ int32_t sdata[];
    
    for(size_t i =  blockIdx.x;i < part_num;i += gridDim.x){
        
    }
    
}
/// @brief 适用于大量数据的前缀和如N>1024*128，如N=1024*128*2。part_num=256，block_num=128,说明需要处理两次。这就是为什么SanPartKernel中会出现for循环。
/// @param input 
/// @param buffer 
/// @param output 
/// @param n 
void san_the_fan(const int*input, int *buffer,int * output, size_t n){
    size_t block_size = 1024;
    size_t part_num = (n+block_size-1)/block_size;
    size_t block_num = std::min(part_num,128);

    ScanPartKernel<<<block_num,block_size>>>(input,output,buffer,part_num);
}