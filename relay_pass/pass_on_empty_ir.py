import numpy as np
import tvm
from tvm import te
import tvm.relay as relay


#@relay.transform.module_pass(opt_level=2)
@tvm.ir.transform.module_pass(opt_level=2)
def transform(mod, ctx):
    tp = relay.TensorType((10,), "float32")
    x = relay.var("x", tp)
    gv = relay.GlobalVar("abs")
    func = relay.Function([x], relay.abs(x))
    new_mod = tvm.IRModule({gv: func})
    new_mod.update(mod)
    return new_mod

module_pass = transform
print(type(module_pass))
assert isinstance(module_pass, tvm.ir.transform.ModulePass)
assert module_pass.info.opt_level == 2

mod = tvm.IRModule()
print("Empty mod:")
print(mod)
print(mod.script())
mod = module_pass(mod)
print("Updated mod:")
print(mod)
print(mod.script())
"""
<class 'tvm.ir.transform.ModulePass'>
Empty mod:

# from tvm.script import tir as T
@tvm.script.ir_module
class Module:

Updated mod:
def @abs(%x: Tensor[(10), float32]) {
  abs(%x)
}

# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
"""
