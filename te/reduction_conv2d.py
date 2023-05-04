import tvm
import tvm.testing
from tvm import te
import numpy as np

tgt = tvm.target.Target(target="llvm", host="llvm")
n = te.var("n")
Input = te.placeholder((n, n), name="Input")
Filter = te.placeholder((3, 3), name="Filter")
di = te.reduce_axis((0, 3), name="di")
dj = te.reduce_axis((0, 3), name="dj")
Output = te.compute(
    (n - 2, n - 2),
    lambda i, j: te.sum(Input[i + di, j + dj] * Filter[di, dj], axis=[di, dj]),
    name="Output",
)
s = te.create_schedule(Output.op)
print(tvm.lower(s, [Input, Filter, Output], simple_mode=True))

fadd = tvm.build(s, [Input, Filter, Output], tgt, name="myadd")

dev = tvm.device(tgt.kind.name, 0)
ns = 4
ms = 4
fs = 3
aNP = np.empty([ns, ms], dtype = "float32")
aNP[0] = [1.0,2.0,3.0,4.0]
aNP[1] = [1.0,2.0,3.0,4.0]
aNP[2] = [1.0,2.0,3.0,4.0]
aNP[3] = [1.0,2.0,3.0,4.0]
fNP = np.empty([fs, fs], dtype = "float32")
fNP[0] = [1.0,1.0,1.0]
fNP[1] = [1.0,1.0,1.0]
fNP[2] = [1.0,1.0,1.0]
a = tvm.nd.array(aNP, dev)
f = tvm.nd.array(fNP, dev)
resultNP = np.empty([2, 2], dtype = "float32")
result = tvm.nd.array(resultNP, dev)

fadd(a, f, result)
print(a.numpy())
print(f.numpy())
print(result.numpy())
