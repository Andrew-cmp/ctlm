import argparse
import logging
from distutils.util import strtobool
from typing import Optional,Tuple
import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.meta_schedule.testing.te_workload import create_te_workload
from tvm.support import describe
from tvm import te
from tvm import topi
import numpy as np
def write_code(str,path):
    with open(path,'w') as f:
        f.write(str)
def gemm_compute(Batch,M,N,K,dtype):
    # graph
    n = N
    #n = te.var("n")
    n = tvm.runtime.convert(n)
    #m = te.var("m")
    m = M
    m = tvm.runtime.convert(m)
    #k = te.var("k")
    k = K
    k = tvm.runtime.convert(k)
    Batch = tvm.runtime.convert(Batch)
    A = te.placeholder((Batch,n, k), dtype=dtype,name="A")
    B = te.placeholder((Batch,k, m), dtype=dtype,name="B")
    k = te.reduce_axis((0, k), name="k")
    C = te.compute((Batch, m, n), lambda b,i, j: te.sum(A[b,i, k] * B[b,k, j], axis=k), name="C")
    prim_func = te.create_prim_func([A,B,C])
    ir = tvm.ir.IRModule({'main':prim_func})

    return ir

def gemm_native(Batch,M,N,K,dtype):
    ir = gemm_compute(Batch,M,N,K,dtype)
    sch = tir.Schedule(ir,debug_mask='all')
    block_c = sch.get_block("C")
    (b,i,j,k) = sch.get_loops(block_c)
    sch.bind(b, "blockIdx.x")
    sch.bind(i, "blockIdx.y")
    sch.bind(j, "threadIdx.x")
    print("native_gemm:")
    sch.mod.show()
    print('*'*100)
    return sch.mod
def gemm_block(Batch,M,N,K,dtype):
    block_h = 32
    block_w = 32
    ir = gemm_compute(Batch,M,N,K,dtype)
    sch = tir.Schedule(ir,debug_mask='all')
    block_c = sch.get_block("C")
    (b,i,j,k) = sch.get_loops(block_c)
    i = sch.fuse(b,i)
    by, yi = sch.split(i,factors=[None,block_h])
    bx, xi = sch.split(j,factors=[None,block_w])
    sch.reorder(by, bx, yi, xi)
    sch.bind(by, "blockIdx.y")
    sch.bind(bx, "blockIdx.x")
    sch.bind(yi, "threadIdx.y")
    sch.bind(xi, "threadIdx.x")
    # 由于nvcc编译器已经进行了存储优化，此时对ABC进行缓存性能并没有影响，相反还会下降。
    # block_cl = sch.cache_write(block_c, 0, "local")
    # block_c2 = sch.cache_read(block_c,1,'local')
    # block_c3 = sch.cache_read(block_c,0,'local')
    # sch.reverse_compute_at(block_cl, xi, preserve_unit_loops=True)
    # sch.compute_at(block_c2, xi, preserve_unit_loops=True)
    # sch.compute_at(block_c3, xi, preserve_unit_loops=True)
    print("block_gemm:")
    sch.mod.show()
    print('*'*100)
    return sch.mod

def gemm_block(Batch,M,N,K,dtype):
    ir = gemm_compute(Batch,M,N,K,dtype)
    sch = tir.Schedule(ir,debug_mask='all')
    block_c = sch.get_block("C")
    (b,i,j,k) = sch.get_loops(block_c)
    i = sch.fuse(b,i)
    block_h = 32
    block_w = 32
    BK= 32
    ko, ki = sch.split(k, [None, BK])
def profile(irmodule,B, M,N,K,dtype):
    num_flop = 2 * B * M * N * K
    lib = tvm.build(irmodule,target="nvidia/nvidia-a6000")
    dev = tvm.cuda()
    A_np = np.random.uniform(size=(B, M, K)).astype(dtype)
    B_np = np.random.randint(-5, 6, size=(B, K, N)).astype(dtype)
    C_np = (A_np.astype(dtype) @ B_np.astype(dtype)).astype(dtype)

    A_nd = tvm.nd.array(A_np, dev)
    B_nd = tvm.nd.array(B_np, dev)
    C_nd = tvm.nd.array(np.zeros((B, M, N), dtype=dtype), dev)
    evaluator = lib.time_evaluator("main", dev, number=10)
    t = evaluator(A_nd, B_nd, C_nd).mean
    print(t)
    print("MetaSchedule: %f GFLOPS" % (num_flop / (t*1e3) / 1e9))
    delta = np.sum(np.abs(C_nd.numpy().astype("float64") - C_np.astype("float64")))/(M*N)
    print("delta: ", delta)
    print("*"*100)
    write_code(lib.imported_modules[0].get_source(), "tmp.cu")
    
def main():
    B = 1
    M = 1024
    N = 1024
    K = 1024
    dtype = "float32"
    #ir = gemm_native(B,M,N,K,dtype)
    #profile(ir,B,M,N,K,dtype)
    ir = gemm_block(B,M,N,K,dtype)
    profile(ir,B,M,N,K,dtype)
if __name__ == "__main__":
    main()