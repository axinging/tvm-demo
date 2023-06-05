################################################################################################
# Besides, we can also use tensor expression DSL to write simple operators, and convert them
# to an IRModule.
#

from tvm import te
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy as np


A = te.placeholder((8,), dtype="float32", name="A")
B = te.compute((8,), lambda *i: A(*i) + 1.0, name="B")

func = te.create_prim_func([A, B])
ir_module = IRModule({"main": func})


print('ir_module: ')
print(ir_module)
print('ir_module script: ')
print(ir_module.script())

"""
ir_module:
@main = primfn(var_A: handle, var_B: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_1: Pointer(global float32), float32, [8], []),
             B: Buffer(B_1: Pointer(global float32), float32, [8], [])}
  buffer_map = {var_A: A, var_B: B} {
  block([], "root") {
    tir.reads([])
    tir.writes([])
    for (i0: int32, 0, 8) {
      block([8], "B") as [v_i0] {
        bind(v_i0, i0)
        tir.reads([A[v_i0]])
        tir.writes([B[v_i0]])
        B[v_i0] = (A[v_i0] + 1f32)
    }
}


ir_module script:
# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer[8, "float32"], B: T.Buffer[8, "float32"]):
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i0 in T.serial(8):
            with T.block("B"):
                v_i0 = T.axis.spatial(8, i0)
                T.reads(A[v_i0])
                T.writes(B[v_i0])
                B[v_i0] = A[v_i0] + T.float32(1)
"""

################################################################################################
# Build and Run an IRModule
# -------------------------
# We can build the IRModule into a runnable module with specific target backends.
#

'''
<class 'tvm.tir.function.PrimFunc'>
<class 'tvm.ir.module.IRModule'>
<class 'tvm.driver.build_module.OperatorModule'>
'''

print(type(func))
print(type(ir_module))
mod = tvm.build(ir_module, target="llvm")  # The module for CPU backends.
print(type(mod))

################################################################################################
# Prepare the input array and output array, then run the module.
#

a = tvm.nd.array(np.arange(8).astype("float32"))
b = tvm.nd.array(np.zeros((8,)).astype("float32"))
mod(a, b)
print(a)
print(b)
