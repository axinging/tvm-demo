import tvm
from tvm import te
import numpy as np

n = te.var("n")
m = te.var("m")


A = te.placeholder((m, n), name="A")
B = te.compute((m, n), lambda i, j: A[i, j], name="B")

s = te.create_schedule(B.op)
# tile to four axes first: (i.outer, j.outer, i.inner, j.inner)
xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
# then fuse (i.inner, j.inner) into one axis: (i.inner.j.inner.fused)
print(tvm.lower(s, [A, B], simple_mode=True))
fused = s[B].fuse(xi, yi)
print(tvm.lower(s, [A, B], simple_mode=True))
