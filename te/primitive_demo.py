from __future__ import absolute_import, print_function


import tvm
from tvm import te
import numpy as np

# declare some variables for use later
n = te.var("n")
m = te.var("m")

# declare a matrix element-wise multiply
A = te.placeholder((m, n), name="A")
B = te.placeholder((m, n), name="B")
C = te.compute((m, n), lambda i, j: A[i, j] * B[i, j], name="C")

s = te.create_schedule([C.op])
# lower will transform the computation from definition to the real
# callable function. With argument `simple_mode=True`, it will
# return you a readable C like statement, we use it here to print the
# schedule result.
# <class 'tvm.ir.module.IRModule'>
print(type(tvm.lower(s, [A, B, C], simple_mode=True)))
"""
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [m: int32, n: int32], [stride: int32, stride_1: int32], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [m, n], [stride_2: int32, stride_3: int32], type="auto"),
             C: Buffer(C_2: Pointer(float32), float32, [m, n], [stride_4: int32, stride_5: int32], type="auto")}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  for (i: int32, 0, m) {
    for (j: int32, 0, n) {
      C_3: Buffer(C_2, float32, [(stride_4*m)], [], type="auto")[((i*stride_4) + (j*stride_5))] = (A_3: Buffer(A_2, float32, [(stride*m)], [], type="auto")[((i*stride) + (j*stride_1))]*B_3: Buffer(B_2, float32, [(stride_2*m)], [], type="auto")[((i*stride_2) + (j*stride_3))])
    }
  }
}
"""
print((tvm.lower(s, [A, B, C], simple_mode=True)))

tgt = tvm.target.Target(target="llvm", host="llvm")

fMatrixElementwiseMul = tvm.build(s, [A, B, C], tgt, name="myadd")

dev = tvm.device(tgt.kind.name, 0)
aNP = np.empty([3, 4], dtype = "float32")
aNP[0] = [1.0,2.0,3.0,4.0]
aNP[1] = [1.0,2.0,3.0,4.0]
aNP[2] = [1.0,2.0,3.0,4.0]
bNP = np.empty([3, 4], dtype = "float32")
bNP[0] = [1.0,2.0,3.0,4.0]
bNP[1] = [1.0,2.0,3.0,4.0]
bNP[2] = [1.0,2.0,3.0,4.0]
a = tvm.nd.array(aNP, dev)
b = tvm.nd.array(bNP, dev)

resultNP = np.empty([3, 4], dtype = "float32")
result = tvm.nd.array(resultNP, dev)

fMatrixElementwiseMul(a, b, result)
print(a.numpy())
print(b.numpy())
print(result.numpy())



