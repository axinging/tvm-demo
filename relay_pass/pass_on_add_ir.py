import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib import graph_executor

x = relay.var("x", shape=(2, 2), dtype="float32")
y = relay.var("y", shape=(2, 2), dtype="float32")
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

# build the library using graph executor
# params = {"y": np.ones((2, 2), dtype="float32")}
mod = tvm.IRModule.from_expr(relay.Function([x, y], x + y))
#mod = tvm.IRModule.from_expr(relay.Function([x, y], x + abs(y)))

print("Original mod:")
print(mod)
print(mod.script())
mod = module_pass(mod)
print("Updated mod:")
print(mod)
print(mod.script())

lib = relay.build(mod, tvm.target.Target("llvm"))
ns =2
ms =2
aNP = np.empty([ns, ms], dtype = "float32")
aNP[0] = [1.0,-2.0]
aNP[1] = [1.0,-2.0]

dev = tvm.cpu(0)
gmod = graph_executor.GraphModule(lib["default"](dev))
# use the graph module.
# data = tvm.nd.array((np.random.uniform(size=(2,2))).astype("float32"))
data = tvm.nd.array((aNP).astype("float32"))
gmod.set_input("x", data)
gmod.set_input("y", data)
gmod.run()
tvm_output = gmod.get_output(0)
print(tvm_output.shape)
print(tvm_output)
