import numpy as np
from typing import List
import tvm
import tvm.tir.tensor_intrin
import os
import sch_trace_cpu
from common import build_check_evaluate, get_MatmulModule, generate_candidates, get_ConvModule, get_BatchMatmul, get_Conv3dModule
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["TVM_NUM_THREADS"] = str(2)
print("imported")

is_change = True

M = 64
N = 128
K = 1024

CASCADELAKE_VNNI_TARGET = "llvm -mcpu=cascadelake -num-cores 4"
target = CASCADELAKE_VNNI_TARGET

# target = "llvm -mtriple=aarch64-linux-gnu -mattr=+neon -num-cores 4"

# mod = get_MatmulModule(M, N, K, 'uint8', 'int8', 'int32', True, target)

# mod = get_MatmulModule(M, N, K, 'int8', 'int8', 'int32', True, target)
# mod = get_ConvModule((1, 1024, 16, 16), 'int8', (2048, 1024, 1, 1), 'int8', 
#                      (2, 2), (0, 0), 2048, (1, 1), 'NCHW', 'OIHW', 'int32', target)

# mod = get_ConvModule((1, 1024, 16, 16), 'uint8', (2048, 1024, 1, 1), 'int8', 
#                      (2, 2), (0, 0), 2048, (1, 1), 'NCHW', 'OIHW', 'int32', target)

mod = get_Conv3dModule((1, 72, 36, 4, 128), "uint8", (3, 3, 3, 128, 128), 'int8', (1, 1, 1), (1, 1, 1, 1, 1, 1), input_layout="NDHWC", kernel_layout="DHWIO", out_dtype='int32', kernel_size=(3, 3, 3), channels=128, target=target)

# mod = get_Conv3dModule((1, 256, 28, 14, 2), "uint8", (512, 256, 1, 1, 1), 'int8', (2, 2, 2), (0, 0, 0), input_layout="NCDHW", kernel_layout="OIDHW", out_dtype='int32', target=target)
# mod = get_ConvModule((1, 128, 1024, 1024), 'uint8', (128, 128, 3, 3), 'int8', 
#                      (1, 1), (1, 1), 128, (3, 3), 'NCHW', 'OIHW', 'int32', target)

# mod = get_BatchMatmul((12, 128, 64), "uint8", (12, 128, 64), "int8", "int32", target)
# mod = get_BatchMatmul((12, 128, 64), "int8", (12, 128, 64), "int8", "int32", target)

# from tvm.script import ir as I
# from tvm.script import tir as T

# @I.ir_module
# class Module:
#     @T.prim_func
#     def main(p0: T.Buffer((T.int64(12), T.int64(128), T.int64(64)), "uint8"), p1: T.Buffer((T.int64(12), T.int64(128), T.int64(64)), "int8"), compute: T.Buffer((T.int64(12), T.int64(128), T.int64(128)), "int32")):
#         T.func_attr({"tir.noalias": T.bool(True)})
#         # with T.block("root"):
#         T_layout_trans = T.alloc_buffer((T.int64(12), T.int64(8), T.int64(16), T.int64(16), T.int64(4)), "int8")
#         for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(12), T.int64(8), T.int64(16), T.int64(16), T.int64(4)):
#             with T.block("T_layout_trans"):
#                 v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
#                 T.reads(p1[v_ax0, v_ax1 * T.int64(16) + v_ax3, v_ax2 * T.int64(4) + v_ax4])
#                 T.writes(T_layout_trans[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
#                 T.block_attr({"dst_layout": "BNK16n4k", "input_shape": [12, 128, 64], "schedule_rule": "None", "src_layout": "BNK"})
#                 T_layout_trans[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = T.if_then_else(v_ax0 < T.int64(12) and v_ax1 * T.int64(16) + v_ax3 < T.int64(128) and v_ax2 * T.int64(4) + v_ax4 < T.int64(64), p1[v_ax0, v_ax1 * T.int64(16) + v_ax3, v_ax2 * T.int64(4) + v_ax4], T.int8(0))
#         for b, i, j, k in T.grid(T.int64(12), T.int64(128), T.int64(128), T.int64(64)):
#             with T.block("compute"):
#                 v_b, v_i, v_j, v_k = T.axis.remap("SSSR", [b, i, j, k])
#                 T.reads(p0[v_b, v_i, v_k], T_layout_trans[v_b, v_j // T.int64(16), v_k // T.int64(4), v_j % T.int64(16), v_k % T.int64(4)])
#                 T.writes(compute[v_b, v_i, v_j])
#                 T.block_attr({"schedule_rule": "batch_matmul_int8"})
#                 with T.init():
#                     compute[v_b, v_i, v_j] = 0
#                 compute[v_b, v_i, v_j] = compute[v_b, v_i, v_j] + T.Cast("int32", p0[v_b, v_i, v_k]) * T.Cast("int32", T_layout_trans[v_b, v_j // T.int64(16), v_k // T.int64(4), v_j % T.int64(16), v_k % T.int64(4)])

# mod = Module

if is_change:
    # _, schs = generate_candidates(mod, 'neon', 'neon', 'neon', target)
    _, schs = generate_candidates(mod, 'vnni', 'vnni', 'vnni', target)
    assert len(schs) > 0
    sch = schs[0]
    # with open(str("/home/hwhu/ctlm/tvm_app/tvm/tvm_app/sch_trace_cpu.py"),'w') as f:
    #     f.write(str(sch.trace))
    with open(str("/home/hwhu/ctlm/tvm_app/trash.print/sch_trace_cpu2.py"),'w') as f:
        f.write(str(sch.trace))
    with open(str("/home/hwhu/ctlm/tvm_app/trash.print/mod_cpu2.py"),'w') as f:
        f.write(str(sch.mod))
else:
    sch = tvm.tir.Schedule(mod)
    sch_trace_cpu.apply_trace(sch)
    
print("ToTest")

def check_comp_matmul(a: np.array, b: np.array):
    b = b.reshape((N, K)).astype("float32")
    b = np.transpose(b)
    return a.astype("float32") @ b

# build_check_evaluate(sch, check_comp_matmul, False, target=target)
