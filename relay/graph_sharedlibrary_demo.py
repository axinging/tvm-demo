import tvm
from tvm import relay
from tvm.contrib import graph_executor
import numpy as np
# import pdb; pdb.set_trace()

import threading
# build the library using graph executor
x = relay.var("x", shape=(2, 2), dtype="float32")
y = relay.var("y", shape=(2, 2), dtype="float32")
params = {"y": np.ones((2, 2), dtype="float32")}
mod = tvm.IRModule.from_expr(relay.Function([x, y], x + y))
# build a module

print(threading.get_native_id())
print("Before build")
print("type mod: "+ str(type(mod)))
lib_so = relay.build(mod, tvm.target.Target("llvm"), params=params)
print("type lib_so: " + str(type(lib_so)))
print("After build")
#print(lib_so.get_graph_json())
lib_so.export_library("compiled_lib.so")
# load it back as a runtime
lib: tvm.runtime.Module = tvm.runtime.load_module("compiled_lib.so")

print("type lib after tvm.runtime.load_module: " + str(type(lib)))
# Call the library factory function for default and create
# a new runtime.Module, wrap with graph module.
dev = tvm.cpu(0)
print("type lib[default](dev): "+ str(lib["default"](dev)))
gmod = graph_executor.GraphModule(lib["default"](dev))
# use the graph module.
data = tvm.nd.array((np.random.uniform(size=(2,2))).astype("float32"))
gmod.set_input("x", data)
gmod.run()
print("type(gmod): "+ str(type(gmod)))
tvm_output = gmod.get_output(0)
print(tvm_output.shape)
print(tvm_output)
