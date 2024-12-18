import tvm
from tvm.script import tir as T
from tvm.script import ir as I

@I.ir_module
class Module:
    @T.prim_func
    def main(p0: T.Buffer((T.int64(2), T.int64(64), T.int64(512)), "float32"), p1: T.Buffer((T.int64(2), T.int64(512), T.int64(512)), "float32"), T_batch_matmul_NN: T.Buffer((T.int64(2), T.int64(64), T.int64(512)), "float32")):
        T.func_attr({"global_symbol": "main", "layout_free_buffers": [1], "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"meta_schedule.unroll_explicit": 64})
            T_batch_matmul_NN_local = T.alloc_buffer((T.int64(2), T.int64(64), T.int64(512)), scope="local")
            p0_shared = T.alloc_buffer((T.int64(2), T.int64(64), T.int64(1024)), scope="shared")
            p1_shared = T.alloc_buffer((T.int64(2), T.int64(512), T.int64(1024)), scope="shared")
            for b_0_i_0_j_0_fused in T.thread_binding(T.int64(32), thread="blockIdx.x"):
                for b_1_i_1_j_1_fused in T.thread_binding(T.int64(4), thread="vthread.x"):
                    for b_2_i_2_j_2_fused in T.thread_binding(T.int64(1), thread="threadIdx.x"):
                        for k_0 in range(T.int64(512)):
                            for ax0_ax1_ax2_fused in range(T.int64(8)):
                                with T.block("p0_shared"):
                                    v0 = T.axis.spatial(T.int64(2), b_0_i_0_j_0_fused // T.int64(16))
                                    v1 = T.axis.spatial(T.int64(64), b_0_i_0_j_0_fused % T.int64(16) // T.int64(2) * T.int64(8) + ax0_ax1_ax2_fused)
                                    v2 = T.axis.spatial(T.int64(512), k_0)
                                    T.reads(p0[v0, v1, v2])
                                    T.writes(p0_shared[v0, v1, v2])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 4})
                                    p0_shared[v0, v1, v2] = p0[v0, v1, v2]
                            for ax0_ax1_ax2_fused in range(T.int64(256)):
                                with T.block("p1_shared"):
                                    v0 = T.axis.spatial(T.int64(2), b_0_i_0_j_0_fused // T.int64(16))
                                    v1 = T.axis.spatial(T.int64(512), k_0)
                                    v2 = T.axis.spatial(T.int64(512), b_0_i_0_j_0_fused % T.int64(2) * T.int64(256) + ax0_ax1_ax2_fused)
                                    T.reads(p1[v0, v1, v2])
                                    T.writes(p1_shared[v0, v1, v2])
                                    T.block_attr({"meta_schedule.cooperative_fetch": 1})
                                    p1_shared[v0, v1, v2] = p1[v0, v1, v2]
                            for k_1, b_3, i_3, j_3, k_2, b_4, i_4, j_4 in T.grid(T.int64(1), T.int64(1), T.int64(2), T.int64(2), T.int64(1), T.int64(1), T.int64(4), T.int64(32)):
                                with T.block("T_batch_matmul_NN"):
                                    v_b = T.axis.spatial(T.int64(2), b_4 + b_0_i_0_j_0_fused // T.int64(16) + b_3)
                                    v_i = T.axis.spatial(T.int64(64), b_0_i_0_j_0_fused % T.int64(16) // T.int64(2) * T.int64(8) + i_3 * T.int64(4) + i_4)
                                    v_j = T.axis.spatial(T.int64(512), b_0_i_0_j_0_fused % T.int64(2) * T.int64(256) + b_1_i_1_j_1_fused * T.int64(64) + j_3 * T.int64(32) + j_4)
                                    v_k = T.axis.reduce(T.int64(512), k_1 + k_2 + k_0)
                                    T.reads(p0_shared[v_b, v_i, v_k], p1_shared[v_b, v_k, v_j])
                                    T.writes(T_batch_matmul_NN_local[v_b, v_i, v_j])
                                    T.block_attr({"meta_schedule.thread_extent_high_inclusive": 1024, "meta_schedule.thread_extent_low_inclusive": 32, "meta_schedule.tiling_structure": "SSSRRSRS"})
                                    with T.init():
                                        T_batch_matmul_NN_local[v_b, v_i, v_j] = T.float32(0)
                                    T_batch_matmul_NN_local[v_b, v_i, v_j] = T_batch_matmul_NN_local[v_b, v_i, v_j] + p0_shared[v_b, v_i, v_k] * p1_shared[v_b, v_k, v_j]
                        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(8), T.int64(64)):
                            with T.block("T_batch_matmul_NN_local"):
                                v0 = T.axis.spatial(T.int64(2), b_0_i_0_j_0_fused // T.int64(16) + ax0)
                                v1 = T.axis.spatial(T.int64(64), b_0_i_0_j_0_fused % T.int64(16) // T.int64(2) * T.int64(8) + ax1)
                                v2 = T.axis.spatial(T.int64(512), b_0_i_0_j_0_fused % T.int64(2) * T.int64(256) + b_1_i_1_j_1_fused * T.int64(64) + ax2)
                                T.reads(T_batch_matmul_NN_local[v0, v1, v2])
                                T.writes(T_batch_matmul_NN[v0, v1, v2])
                                T_batch_matmul_NN[v0, v1, v2] = T_batch_matmul_NN_local[v0, v1, v2]
                                

# sch = tvm.tir.Schedule(Module)

from tvm.contrib import nvcc
@tvm.register_func("tvm_callback_cuda_compile", override=True)
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code)
    print("this is override tvm_callback_cuda_compile")
    with open("test_shared_dyn.ptx",'wb')as f:
        f.write(ptx)
    return ptx
@tvm._ffi.register_func("tvm_codegen_maxreg",override=True)
def tvm_codegen_maxreg():
    i = -1
    return i
lib = tvm.build(Module, target='nvidia/nvidia-a100')

with open("GenCUDA.cpp",'w') as f:
    f.write(str(lib.imported_modules[0].get_source()))

print("ToTest")
import numpy as np

B = 2
M = 64
N = 512
K = 512
dtype = 'float32'
num_flop = 2 * B * M * N * K
# dev = tvm.cuda()
dev = tvm.cuda()
A_np = np.random.uniform(size=(B, M, K)).astype(dtype)
B_np = np.random.randint(-5, 6, size=(B, K, N)).astype(dtype)
# S_np = np.random.uniform(size=(1,)).astype("float16")
C_np = (A_np.astype("float32") @ B_np.astype("float32")).astype(dtype)

A_nd = tvm.nd.array(A_np, dev)
B_nd = tvm.nd.array(B_np, dev)
# S_nd = tvm.nd.array(S_np, dev)
C_nd = tvm.nd.array(np.zeros((B, M, N), dtype=dtype), dev)
evaluator = lib.time_evaluator("main", dev, number=10)
print(evaluator(A_nd, B_nd, C_nd).mean)
print("MetaSchedule: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))
delta = np.sum(np.abs(C_nd.numpy().astype("float64") - C_np.astype("float64")))/(M*N)
print("delta: ", delta)
print("to run, see htop ...")