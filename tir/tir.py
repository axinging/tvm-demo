import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy as np
from tvm import relay

# TIR:
# In: <class 'tvm.ir.module.IRModule'>
# tvm.build
# Out: <class 'tvm.driver.build_module.OperatorModule'>


@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle):
        # We exchange data between function by handles, which are similar to pointer.
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # Create buffer from handles.
        A = T.match_buffer(a, (8,), dtype="float32")
        B = T.match_buffer(b, (8,), dtype="float32")
        for i in range(8):
            # A block is an abstraction for computation.
            with T.block("B"):
                # Define a spatial block iterator and bind it to value i.
                vi = T.axis.spatial(8, i)
                B[vi] = A[vi] + 1.0


ir_module = MyModule
# <class 'tvm.ir.module.IRModule'>
print("ir_module: ")
print(type(ir_module))
print((ir_module))

print('ir_module.script():')
print(type(ir_module.script()))
print(ir_module.script())

"""
ir_module:
<class 'tvm.ir.module.IRModule'>
@main = primfn(a: handle, b: handle) -> ()
  attr = {"tir.noalias": True, "global_symbol": "main"}
  buffers = {A: Buffer(A_1: Pointer(global float32), float32, [8], []),
             B: Buffer(B_1: Pointer(global float32), float32, [8], [])}
  buffer_map = {a: A, b: B} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    for (i: int32, 0, 8) {
      block([8], "B") as [vi] {
        bind(vi, i)
        tir.reads([A[vi]])
        tir.writes([B[vi]])
        B[vi] = (A[vi] + 1f32)
    }
}


ir_module.script():
<class 'tvm.runtime.container.String'>
# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer[8, "float32"], B: T.Buffer[8, "float32"]):
        # function attr dict
        T.func_attr({"tir.noalias": True, "global_symbol": "main"})
        # body
        # with T.block("root")
        for i in T.serial(8):
            with T.block("B"):
                vi = T.axis.spatial(8, i)
                T.reads(A[vi])
                T.writes(B[vi])
                B[vi] = A[vi] + T.float32(1
"""

mod = tvm.build(ir_module, target="llvm")  # The module for CPU backends.
print(type(mod))
a = tvm.nd.array(np.arange(8).astype("float32"))
b = tvm.nd.array(np.zeros((8,)).astype("float32"))
mod(a, b)
print(a)
print(b)
