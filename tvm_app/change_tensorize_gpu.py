import numpy as np
import tvm
import tvm.tir.tensor_intrin
import os
import sch_trace_gpu
from common import build_check_evaluate, generate_candidates, get_MatmulModule, get_ConvModule
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TVM_NUM_THREADS"] = str(2)
os.environ["TVM_BACKTRACE"] = str(1)
M = 64
N = 256
K = 128
work_dir = '/home/hwhu/ctlm/tvm_app/trash.print/'
is_update_database = True
target = "nvidia/nvidia-a100"
dtype = "float16"
is_change = True
# mod = get_MatmulModule(M, N, K, dtype, dtype, dtype)
# mod = get_ConvModule((1, 16, 16, 1024), 'float16', (1, 1, 1024, 2048), 'float16', 
#                      (2, 2), (0, 0), 2048, (1, 1), 'NHWC', 'HWIO', 'float16', target)
from tvm.script import tir as T

def MyMatmulModule(A_dtype,B_dtype,C_dtype):
    @tvm.script.ir_module
    class MatMulModule:
        @T.prim_func
        def main(
            A: T.Buffer((M, K), A_dtype),
            B: T.Buffer((K, N), B_dtype),
            C: T.Buffer((M, N), C_dtype),
        ):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            for i, j, k in T.grid(M, N, K):
                with T.block("C"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    return MatMulModule
if is_change:
    # _, schs = generate_candidates(mod, 'cuda-tensorcore-83216nn', 
    #                             'cuda-tensorcore', 'cuda-tensorcore', target)
    mod = MyMatmulModule(dtype,dtype,dtype)
    _, schs = generate_candidates(mod,target)
    assert len(schs) > 0
    sch = schs[0]

    with open(str("/home/hwhu/ctlm/tvm_app/trash.print/matmul_sch_trace_gpu_2.py"),'w') as f:
        f.write(str(sch.trace))
else:
    sch = tvm.tir.Schedule(mod)
    sch_trace_gpu.apply_trace(sch)
    
print("ToTest")

def check_comp_matmul(a: np.array, b: np.array):
    return a.astype("float32") @ b.astype("float32")

build_check_evaluate(sch, check_comp_matmul, False,)
