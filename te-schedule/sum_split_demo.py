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
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

# Below loop is optional to create schedule

s = te.create_schedule(C.op)
# <class 'tvm.te.schedule.Schedule'>
print(type(s))

print(tvm.lower(s, [A, B, C], simple_mode=True))
xo, xi = s[C].split(C.op.axis[0], factor=32)
# s[C].parallel(C.op.axis[0])
print(tvm.lower(s, [A, B, C], simple_mode=True))
'''
@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [n: int32], [stride: int32], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [n], [stride_1: int32], type="auto"),
             C: Buffer(C_2: Pointer(float32), float32, [n], [stride_2: int32], type="auto")}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  for (i: int32, 0, n) {
    C_3: Buffer(C_2, float32, [(stride_2*n)], [], type="auto")[(i*stride_2)] = (A_3: Buffer(A_2, float32, [(stride*n)], [], type="auto")[(i*stride)] + B_3: Buffer(B_2, float32, [(stride_1*n)], [], type="auto")[(i*stride_1)])
  }
}


@main = primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"from_legacy_te_schedule": True, "global_symbol": "main", "tir.noalias": True}
  buffers = {A: Buffer(A_2: Pointer(float32), float32, [n: int32], [stride: int32], type="auto"),
             B: Buffer(B_2: Pointer(float32), float32, [n], [stride_1: int32], type="auto"),
             C: Buffer(C_2: Pointer(float32), float32, [n], [stride_2: int32], type="auto")}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  for (i.outer: int32, 0, floordiv((n + 31), 32)) {
    for (i.inner: int32, 0, 32) {
      if @tir.likely((((i.outer*32) + i.inner) < n), dtype=bool) {
        let cse_var_1: int32 = ((i.outer*32) + i.inner)
        C_3: Buffer(C_2, float32, [(stride_2*n)], [], type="auto")[(cse_var_1*stride_2)] = (A_3: Buffer(A_2, float32, [(stride*n)], [], type="auto")[(cse_var_1*stride)] + B_3: Buffer(B_2, float32, [(stride_1*n)], [], type="auto")[(cse_var_1*stride_1)])
      }
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
n = 4
aNP = np.empty(4, dtype = "float32")
aNP[0] = 1.0;
aNP[1] = 2.0;
aNP[2] = 3.0;
aNP[3] = 4.0;
a = tvm.nd.array(aNP, dev)
b = tvm.nd.array(aNP, dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
fadd(a, b, c)
tvm.testing.assert_allclose(c.numpy(), np.add(a.numpy(),b.numpy()))
print(c.numpy())
