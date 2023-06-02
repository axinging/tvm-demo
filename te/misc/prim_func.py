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


print(ir_module.script())


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
