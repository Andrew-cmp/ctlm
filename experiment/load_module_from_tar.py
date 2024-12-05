from tvm.runtime.module import load_module
#from tvm.runtime.module import load
import tvm
from tvm import te
import numpy as np
from tvm import relay
from tvm import tir
shape = (2, 3, 4)
a = te.placeholder(shape, dtype="float16", name="a")
b = tvm.tir.const(0.5, dtype="float16")
c = te.compute(shape, lambda i, j, k: a[i, j, k] > b, name="c")
s = te.create_schedule(c.op)
axes = [axis for axis in c.op.axis]
fused = s[c].fuse(*axes)
bx, tx = s[c].split(fused, factor=64)
s[c].bind(bx, te.thread_axis("blockIdx.x"))
s[c].bind(tx, te.thread_axis("threadIdx.x"))
target = tvm.target.Target('nvidia/nvidia-a6000')
func_runtime = tvm.build(s, [a, c], target=target,target_host='llvm')
print(func_runtime.format)
print(func_runtime.is_binary_serializable)
print(func_runtime.is_runnable)
print(func_runtime.is_dso_exportable)
# #mod = load_module("/tmp/tmpfyz3zado/tvm_tmp_mod.tar")
# #mod = load("/tmp/tmpfyz3zado/tvm_tmp_mod.tar.so")
# print(mod.get_source())
# print(mod.imported_modules[0].get_source())