import numpy as np
import tvm
from tvm import relax
from tvm.ir.module import IRModule
from tvm.script import relax as R
from tvm.script import tir as T
from common import write_cuda_code,save_sch_mod_cuda_code_interact
def accel_fill_zero(C):
    C[:] = 0

def accel_tmm_add(C, A, B):
    C[:] += A @ B.T

def accel_dma_copy(reg, dram):
    reg[:] = dram[:]

def lnumpy_tmm(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    # a special accumulator memory
    C_accumulator = np.empty((16, 16), dtype="float32")
    A_reg = np.empty((16, 16), dtype="float32")
    B_reg = np.empty((16, 16), dtype="float32")

    for i in range(64):
        for j in range(64):
            accel_fill_zero(C_accumulator[:,:])
            for k in range(64):
                accel_dma_copy(A_reg[:], A[i * 16 : i * 16 + 16, k * 16 : k * 16 + 16])
                accel_dma_copy(B_reg[:], B[j * 16 : j * 16 + 16, k * 16 : k * 16 + 16])
                accel_tmm_add(C_accumulator[:,:], A_reg, B_reg)
            accel_dma_copy(C[i * 16 : i * 16 + 16, j * 16 : j * 16 + 16], C_accumulator[:,:])
# dtype = "float32"
# a_np = np.random.rand(1024, 1024).astype(dtype)
# b_np = np.random.rand(1024, 1024).astype(dtype)
# c_tmm = a_np @ b_np.T
# c_np = np.empty((1024, 1024), dtype="float32")
# lnumpy_tmm(a_np, b_np, c_np)
# np.testing.assert_allclose(c_np, c_tmm, rtol=1e-5)
@T.prim_func
def tmm16_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32", offset_factor=16, scope="global.A_reg")
    B = T.match_buffer(b, (16, 16), "float32", offset_factor=16, scope="global.B_reg")
    C = T.match_buffer(c, (16, 16), "float32", offset_factor=16,  scope="global.accumulator")

    with T.block("root"):
        T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for i, j, k in T.grid(16, 16, 16):
            with T.block(""):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]


@T.prim_func
def tmm16_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    sa = T.int32()
    sb = T.int32()
    sc = T.int32()
    A = T.match_buffer(a, (16, 16), "float32", offset_factor=16, strides=[sa, 1], scope="global.A_reg")
    B = T.match_buffer(b, (16, 16), "float32", offset_factor=16, strides=[sb, 1], scope="global.B_reg")
    C = T.match_buffer(c, (16, 16), "float32", offset_factor=16, strides=[sc, 1], scope="global.accumulator")

    with T.block("root"):
        T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.call_extern(
                "tmm16",
                C.access_ptr("w"),
                A.access_ptr("r"),
                B.access_ptr("r"),
                sa,
                sb,
                sc,
                dtype="int32",
            )
        )
tvm.tir.TensorIntrin.register("tmm16", tmm16_desc, tmm16_impl)

@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(
        A: T.Buffer((1024, 1024), "float32"),
        B: T.Buffer((1024, 1024), "float32"),
        C: T.Buffer((1024, 1024), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] += A[vi, vk] * B[vj, vk]
sch = tvm.tir.Schedule(MatmulModule)
save_sch_mod_cuda_code_interact(sch)
i, j, k = sch.get_loops("matmul")
i, ii = sch.split(i, factors=[None, 16])
j, ji = sch.split(j, factors=[None, 16])
k, ki = sch.split(k, factors=[None, 16])
sch.reorder(i, j, k, ii, ji, ki)
save_sch_mod_cuda_code_interact(sch)
block_mm = sch.blockize(ii)
save_sch_mod_cuda_code_interact(sch)
A_reg = sch.cache_read(block_mm, 0, storage_scope="global.A_reg")
save_sch_mod_cuda_code_interact(sch)
B_reg = sch.cache_read(block_mm, 1, storage_scope="global.B_reg")
save_sch_mod_cuda_code_interact(sch)
sch.compute_at(A_reg, k)
save_sch_mod_cuda_code_interact(sch)
sch.compute_at(B_reg, k)
save_sch_mod_cuda_code_interact(sch)
write_back_block = sch.cache_write(block_mm, 0, storage_scope="global.accumulator")
save_sch_mod_cuda_code_interact(sch)
sch.reverse_compute_at(write_back_block, j)
save_sch_mod_cuda_code_interact(sch)
sch.decompose_reduction(block_mm, k)
save_sch_mod_cuda_code_interact(sch)
sch.tensorize(block_mm, "tmm16")
save_sch_mod_cuda_code_interact(sch)




# def tmm_kernel():
#     cc_code = """
#       extern "C" int tmm16(float *cc, float *aa, float *bb, int stride_a, int stride_b, int stride_c) {
#         for (int i = 0; i < 16; ++i) {
#             for (int j = 0; j < 16; ++j) {
#                 for (int k = 0; k < 16; ++k) {
#                     cc[i * stride_c + j] += aa[i * stride_a + k] * bb[j * stride_b + k];
#                 }
#             }
#         }
#         return 0;
#       }
#     """
#     from tvm.contrib import clang, utils

#     temp = utils.tempdir()
#     ll_path = temp.relpath("temp.ll")
#     # Create LLVM ir from c source code
#     ll_code = clang.create_llvm(cc_code, output=ll_path)
#     return ll_code

# sch.annotate(i, "pragma_import_llvm", tmm_kernel())