import tvm
import tvm.testing
from tvm import te
import numpy as np
# import pdb; pdb.set_trace()
tgt = tvm.target.Target(target="llvm", host="llvm")
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = te.create_schedule(C.op)
fadd = tvm.build(s, [A, B, C], tgt, name="myadd")
dev = tvm.device(tgt.kind.name, 0)

nS = 3
#a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
aN = np.empty(nS, dtype="float32")
bN = np.empty(nS, dtype="float32")
for i in range(nS):
    aN[i] = float(i)
for i in range(nS):
    bN[i] = float(i)

print(C.dtype)
dtype = "float32"
a = tvm.nd.array(aN, dev)
b = tvm.nd.array(bN, dev)
c = tvm.nd.array(np.zeros(nS, dtype=C.dtype), dev)
fadd(a, b, c)

a2 = np.from_dlpack(a)
c2 = np.from_dlpack(c)

c2Numpy = np.add(a2,c2)

# tvm.testing.assert_allclose(c.numpy(), c2Numpy)
print(c.numpy())
print(c2Numpy)
