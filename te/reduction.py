import tvm
import tvm.testing
from tvm import te
import numpy as np
# import pdb; pdb.set_trace()
tgt = tvm.target.Target(target="llvm", host="llvm")
n = te.var("n")
m = te.var("m")
A = te.placeholder((n, m), name="A")
k = te.reduce_axis((0, m), "k")
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")
print(type(B))
print(type(B.op))
print('Print str(B.op):')
print(str(B.op))
s = te.create_schedule(B.op)
print(type(s))
fadd = tvm.build(s, [A, B], tgt, name="myadd")

print(type(fadd))
# test
dev = tvm.device(tgt.kind.name, 0)
ns = 3
ms = 3
aNP = np.empty([ns, ms], dtype = "float32")
aNP[0] = [1,2,3]
aNP[1] = [1,2,3]
aNP[2] = [1,2,3]
a = tvm.nd.array(aNP, dev)
b = tvm.nd.array(np.zeros(ns, dtype=B.dtype), dev)
fadd(a, b)
print(a.numpy())
print(b.numpy())
