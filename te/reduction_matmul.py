import tvm
import tvm.testing
from tvm import te
import numpy as np

tgt = tvm.target.Target(target="llvm", host="llvm")

N = te.var("N")
L= te.var("L")
M = te.var("M")
dtype = "float32"
A = te.placeholder((N, L), name="A", dtype=dtype)
B = te.placeholder((L, M), name="B", dtype=dtype)

k = te.reduce_axis((0, L), name="k")
C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
print(type(C))
print(type(C.op))
s = te.create_schedule(C.op)

# print(tvm.lower(s, [A, B, C], simple_mode=True))

fmatmul = tvm.build(s, [A, B, C], tgt, name="myadd")

dev = tvm.device(tgt.kind.name, 0)
aNP = np.empty([3, 4], dtype = "float32")
aNP[0] = [1.0,2.0,3.0,4.0]
aNP[1] = [1.0,2.0,3.0,4.0]
aNP[2] = [1.0,2.0,3.0,4.0]
fNP = np.empty([4, 3], dtype = "float32")
fNP[0] = [1.0,1.0,1.0]
fNP[1] = [1.0,1.0,1.0]
fNP[2] = [1.0,1.0,1.0]
fNP[3] = [1.0,1.0,1.0]
a = tvm.nd.array(aNP, dev)
f = tvm.nd.array(fNP, dev)
resultNP = np.empty([3, 3], dtype = "float32")
result = tvm.nd.array(resultNP, dev)

fmatmul(a, f, result)
print(a.numpy())
print(f.numpy())
print(result.numpy())
