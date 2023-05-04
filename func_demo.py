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

# Below loop is optional to create schedule


s = te.create_schedule(C.op)
fadd = tvm.build(s, [A, B, C], tgt, name="myadd")

#<class 'tvm.te.tensor.Tensor'>
#compute(C, body=[(A[i] + B[i])], axis=[iter_var(i, range(min=0, ext=n))], reduce_axis=[], tag=, attrs={})
#<class 'tvm.driver.build_module.OperatorModule'>
print(type(C))
print(C.op)
print(type(fadd))

#print(dir(fadd))
#print(fadd.get_source())

# print(type(fadd)) # tvm.driver.build_module.OperatorModule
# test
dev = tvm.device(tgt.kind.name, 0)
n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
fadd(a, b, c)
tvm.testing.assert_allclose(c.numpy(), np.add(a.numpy(),b.numpy()))
print(c.numpy())
