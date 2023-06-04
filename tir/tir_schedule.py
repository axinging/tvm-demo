import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy as np

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
print(type(ir_module))
# print(ir_module.script())

sch = tvm.tir.Schedule(ir_module)
print("before tir schedule: ")
print(type(sch))
print(type(sch.mod))
print((sch.mod))

# Get block by its name
block_b = sch.get_block("B")
# Get loops surrounding the block
(i,) = sch.get_loops(block_b)
# Tile the loop nesting.
i_0, i_1, i_2 = sch.split(i, factors=[2, 2, 2])
print("After tir schedule: ")
print((sch.mod))
print(sch.mod.script())

sch.reorder(i_0, i_2, i_1)
# print(sch.mod.script())

# mod = tvm.build(ir_module, target="llvm")  # The module for CPU backends.
mod = tvm.build(sch.mod, target="llvm")  # The module for CPU backends.
print(type(mod))

a = tvm.nd.array(np.arange(8).astype("float32"))
b = tvm.nd.array(np.zeros((8,)).astype("float32"))
mod(a, b)
print(a)
print(b)
