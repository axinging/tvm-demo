import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib import graph_executor

x = relay.var("x", shape=(2, 2), dtype="float32")
y = relay.var("y", shape=(2, 2), dtype="float32")

# build the library using graph executor
# params = {"y": np.ones((2, 2), dtype="float32")}
input_mod = tvm.IRModule.from_expr(relay.Function([x, y], x + y))
#mod = tvm.IRModule.from_expr(relay.Function([x, y], x + abs(y)))

@relay.transform.function_pass(opt_level=1)
class TestReplaceFunc:
   def __init__(self, new_func):
      self.new_func = new_func
   def transform_function(self, func, mod, ctx):
      # Just for demo purposes
      # Transform func to new_func
      return self.new_func

f1 = relay.Function([x, y], x*y)
# fpass is now a special pass that replaces every
# function to f1
fpass = TestReplaceFunc(f1)
# Now every function in input_mod is replaced by f1
print("Original mod:")
print(input_mod)
print(input_mod.script())
res_mod = fpass(input_mod)

print("Updated mod:")
print(res_mod)
print(res_mod.script())

lib = relay.build(res_mod, tvm.target.Target("llvm"))
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
