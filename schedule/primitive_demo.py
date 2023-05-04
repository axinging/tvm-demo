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
print(tvm.lower(s, [A, B, C], simple_mode=True))
