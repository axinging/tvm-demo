import tvm
from tvm import relay
from tvm.contrib import graph_executor
import numpy as np
from tvm import te
#import pdb; pdb.set_trace()

n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
s = te.create_schedule(B.op)
# Compile library as dynamic library
mod = tvm.build(s, [A, B], "llvm", name="addone")

fname = "addone"
# Get the function from the module
f = mod.get_function(fname)
# Use tvm.nd.array to convert numpy ndarray to tvm
# NDArray type, so that function can be invoked normally
N = 10
x = tvm.nd.array(np.arange(N, dtype=np.float32))
y = tvm.nd.array(np.zeros(N, dtype=np.float32))
# Invoke the function
f(x, y)
np_x = x.numpy()
np_y = y.numpy()
