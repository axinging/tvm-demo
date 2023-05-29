import tvm
import tvm.testing
from tvm import te, topi
import numpy as np
# import pdb; pdb.set_trace()
tgt = tvm.target.Target(target="llvm", host="llvm")
n = te.var("n")
m = te.var("m")
A = te.placeholder((n, m), name="A")

C = topi.sum(A, axis=1)
print(type(C))
s = te.create_schedule(C.op)
print(type(s))

print(tvm.lower(s, [A], simple_mode=True))

fadd = tvm.build(s, [A], tgt, name="myadd")

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
# b = tvm.nd.array(np.zeros(ns, dtype=B.dtype), dev)
fadd(a)
print(a.numpy())
#print(b.numpy())
