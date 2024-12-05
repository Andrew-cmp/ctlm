import tvm
from tvm import te
import numpy as np
from tvm import relay
from tvm import tir
def test():
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
    
    func_prim_func = te.create_prim_func([a,c])
    func_ir = tvm.ir.IRModule({"main": func_prim_func})
    sch = tir.Schedule(func_ir, debug_mask="all")
    block_c = sch.get_block("c")
    i,j,k = sch.get_loops(block=block_c)
    sch.bind(i, "blockIdx.x")
    sch.bind(j, "threadIdx.x")
    sch.bind(k, "threadIdx.y")
    #sch.mod.show()
    
    func_ir_runtime=tvm.build(sch.mod,target=target)
    func_ir_module0 = func_ir_runtime.imported_modules[0]
    #func_ir_runtime.save("test.so")
    func_ir_module0.save("test.ptx",fmt='ptx')
    
    func_ir_source = func_ir_module0.get_source(fmt='ptx')
    print(func_ir_source)
    # print(type(func_prim_func_source))
    # print(str(func_prim_func_source))
    # #并不是我们想象中的代码，而是很奇怪的东西
    # func_lower = tvm.lower(s, [a, c])
    # print(type(func_lower))
    # print(str(func_lower))
    
    # print(type(func))
    # print(str(func))
    
    #source = func.get_source()# 这个拿到的好像是host端llvm ir的代码
    ## code gen gpu 代码
    
    #module0 = func_runtime.imported_modules[0]
    #source = module0.get_source()
    
    #print(type(source))
    #print(source)
    

test()