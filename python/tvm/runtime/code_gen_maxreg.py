import tvm

from tvm import _ffi


@tvm._ffi.register_func
def tvm_codegen_maxreg():
    i = -1
    return i
