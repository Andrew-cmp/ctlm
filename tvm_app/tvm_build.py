import numpy as np

import tvm
from tvm import te
from tvm.ir.module import IRModule
from tvm.script import tir as T
import tvm.testing
"""Tests for MetaSchedule search space on CUDA"""
from typing import List, Optional, Tuple, Union

# isort: off
from typing_extensions import Literal

# isort: on
from tvm.meta_schedule.testing.space_generation import get_rules
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.te_workload import create_te_workload
from tvm.target import Target
from tvm.ir import IRModule
from tvm.tir import Schedule
from tvm.meta_schedule.testing.te_workload import create_te_workload
@tvm.script.ir_module
class LoweredTIRModule:
    @T.prim_func
    def main(A: T.Buffer((16), "float32"), B: T.Buffer((32), "float32")) -> None:
        T.func_attr({"global_symbol": "default_function", "tir.noalias": True})
        bx = T.env_thread("blockIdx.x")
        tx = T.env_thread("threadIdx.x")
        T.launch_thread(bx, 1)
        T.launch_thread(tx, 32)
        A_local = T.Buffer((32), "float32", scope="local")

        with T.block():
            T.reads(A[0:16])
            T.writes(A_local[0:32])
            A_local[tx] = T.if_then_else(tx % 2 == 0, A[tx // 2], T.float32(0), dtype="float32")
            B[tx] = A_local[tx] + 1.0
mod = LoweredTIRModule
func = mod['main']
func_with_attr = func.with_attr({"register": "register"})
mod.update_func(mod.get_global_var("main"), func_with_attr)
mod.show()
target = "nvidia/nvidia-a100"
runtimemode = tvm.build(mod,target)
