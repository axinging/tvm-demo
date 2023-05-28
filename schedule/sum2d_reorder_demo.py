import tvm
import tvm.testing
from tvm import te
import numpy as np
# import pdb; pdb.set_trace()

# TE:
# In: <class 'tvm.te.schedule.Schedule'>
# tvm.build
# Out: <class 'tvm.driver.build_module.OperatorModule'>
tgt = tvm.target.Target(target="llvm", host="llvm")
n = te.var("n")
m = te.var("m")
A = te.placeholder((m, n), name="A")
B = te.placeholder((m, n), name="B")
C = te.compute(A.shape, lambda i,j : A[i, j] + B[i,j], name="C")

# Below loop is optional to create schedule

s = te.create_schedule(C.op)
y, x = s[C].op.axis
# <class 'tvm.te.schedule.Schedule'>
print(type(s[C].op.axis))
print(type(s))
print(tvm.lower(s, [A, B, C], simple_mode=True))
# s[C].parallel(C.op.axis[0])
# s[C].reorder(x, y)
s[C].reorder(x, y)
print(tvm.lower(s, [A, B, C], simple_mode=True))
'''
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [m: int32, n: int32], [stride: int32, stride_1: int32], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [m, n], [stride_2: int32, stride_3: int32], type="auto"),
             C: Buffer(C_2: Pointer(float32), float32, [m, n], [stride_4: int32, stride_5: int32], type="auto")}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  for (i: int32, 0, m) {
    for (j: int32, 0, n) {
      C_3: Buffer(C_2, float32, [(stride_4*m)], [], type="auto")[((i*stride_4) + (j*stride_5))] = (A_3: Buffer(A_2, float32, [(stride*m)], [], type="auto")[((i*stride) + (j*stride_1))] + B_3: Buffer(B_2, float32, [(stride_2*m)], [], type="auto")[((i*stride_2) + (j*stride_3))])
    }
  }
}


@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [m: int32, n: int32], [stride: int32, stride_1: int32], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [m, n], [stride_2: int32, stride_3: int32], type="auto"),
             C: Buffer(C_2: Pointer(float32), float32, [m, n], [stride_4: int32, stride_5: int32], type="auto")}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  for (j: int32, 0, n) {
    for (i: int32, 0, m) {
      C_3: Buffer(C_2, float32, [(stride_4*m)], [], type="auto")[((i*stride_4) + (j*stride_5))] = (A_3: Buffer(A_2, float32, [(stride*m)], [], type="auto")[((i*stride) + (j*stride_1))] + B_3: Buffer(B_2, float32, [(stride_2*m)], [], type="auto")[((i*stride_2) + (j*stride_3))])
    }
  }
}

'''


fadd = tvm.build(s, [A, B, C], tgt, name="myadd")

# C: <class 'tvm.te.tensor.Tensor'>
# compute(C, body=[(A[i] + B[i])], axis=[iter_var(i, range(min=0, ext=n))], reduce_axis=[], tag=, attrs={})
# fadd: <class 'tvm.driver.build_module.OperatorModule'>
print(type(fadd))

#print(dir(fadd))
#print(fadd.get_source())

# print(type(fadd)) # tvm.driver.build_module.OperatorModule
# test
dev = tvm.device(tgt.kind.name, 0)
m = 3
n = 4

aNP = np.empty([m, n], dtype = "float32")
aNP[0] = [1.0,2.0,3.0,4.0]
aNP[1] = [1.0,2.0,3.0,4.0]
aNP[2] = [1.0,2.0,3.0,4.0]

a = tvm.nd.array(aNP, dev)
b = tvm.nd.array(aNP, dev)
c = tvm.nd.array(np.zeros([m, n], dtype=C.dtype), dev)
fadd(a, b, c)
tvm.testing.assert_allclose(c.numpy(), np.add(a.numpy(),b.numpy()))
print(c.numpy())
