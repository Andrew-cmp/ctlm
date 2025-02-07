import numpy as np
import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.script import relax as R
from tvm.script import tir as T
from common import write_cuda_code,save_sch_mod_cuda_code_interact
@tvm.script.ir_module
class MyModuleMatmul:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"),
             B: T.Buffer((1024, 1024), "float32"),
             C: T.Buffer((1024, 1024), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
def cache_read_and_coop_fetch(sch, block, nthread, read_idx, read_loc):
    read_cache = sch.cache_read(block=block, read_buffer_index=read_idx, storage_scope="shared")
    save_sch_mod_cuda_code_interact(sch)
    sch.compute_at(block=read_cache, loop=read_loc)
    save_sch_mod_cuda_code_interact(sch)
    # vectorized cooperative fetch
    inner0, inner1 = sch.get_loops(block=read_cache)[-2:]
    inner = sch.fuse(inner0, inner1)
    save_sch_mod_cuda_code_interact(sch)
    _, tx, vec = sch.split(loop=inner, factors=[None, nthread, 4])
    save_sch_mod_cuda_code_interact(sch)
    sch.vectorize(vec)
    save_sch_mod_cuda_code_interact(sch)
    sch.bind(tx, "threadIdx.x")
    save_sch_mod_cuda_code_interact(sch)


def blocking_with_shared(
    sch,
    tile_local_y,
    tile_local_x,
    tile_block_y,
    tile_block_x,
    tile_k):
    save_sch_mod_cuda_code_interact(sch)
    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")

    save_sch_mod_cuda_code_interact(sch)
    i, j, k = sch.get_loops(block=block_C)
    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])
    save_sch_mod_cuda_code_interact(sch)
    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    save_sch_mod_cuda_code_interact(sch)
    sch.reverse_compute_at(C_local, j1)
    save_sch_mod_cuda_code_interact(sch)

    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")
    save_sch_mod_cuda_code_interact(sch)
    tx = sch.fuse(i1, j1)
    save_sch_mod_cuda_code_interact(sch)
    sch.bind(tx, "threadIdx.x")
    save_sch_mod_cuda_code_interact(sch)
    nthread = tile_block_y * tile_block_x
    cache_read_and_coop_fetch(sch, block_C, nthread, 0, k0)
    cache_read_and_coop_fetch(sch, block_C, nthread, 1, k0)
    sch.decompose_reduction(block_C, k0)
    save_sch_mod_cuda_code_interact(sch)
    return sch

sch = tvm.tir.Schedule(MyModuleMatmul)
sch = blocking_with_shared(sch, 8, 8, 8, 8, 8)
sch.mod.show()