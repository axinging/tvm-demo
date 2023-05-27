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
s[C].parallel(C.op.axis[0])
print(tvm.lower(s, [A, B, C], simple_mode=True))
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
