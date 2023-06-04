import tvm
import tvm.testing
from tvm import te
import numpy as np
from tvm import autotvm
import time

@autotvm.template("tutorial/matmul_v1")  # 1. use a decorator
def matmul_v1(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    s = te.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    # 2. get the config object
    cfg = autotvm.get_config()

    # 3. define search space
    cfg.define_knob("tile_y", [1, 2, 4, 8, 16])
    cfg.define_knob("tile_x", [1, 2, 4, 8, 16])

    # 4. schedule according to config
    yo, yi = s[C].split(y, cfg["tile_y"].val)
    xo, xi = s[C].split(x, cfg["tile_x"].val)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]


tgt = tvm.target.Target(target="llvm", host="llvm")


N = te.var("N")
L= te.var("L")
M = te.var("M")
dtype = "float32"


s, [A, B, C] = matmul_v1(N, L, M,dtype)
## TODO: 
N, L, M = 512, 512, 512
task = autotvm.task.create("tutorial/matmul_v1", args=(N, L, M, "float32"), target="llvm")
print(task.config_space)


measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=5))

# Begin tuning with RandomTuner, log records to file `matmul.log`
# You can use alternatives like XGBTuner.
tuner = autotvm.tuner.RandomTuner(task)
tuner.tune(
    n_trial=10,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file("matmul.log")],
)

with autotvm.apply_history_best("matmul.log"):
    with tvm.target.Target("llvm"):
        s, arg_bufs = matmul_v1(N, L, M, "float32")
        func_tuned = tvm.build(s, arg_bufs)
with tvm.target.Target("llvm"):
    s, arg_bufs = matmul_v1(N, L, M, "float32")
    func_untuned = tvm.build(s, arg_bufs)

# check correctness
a_np = np.random.uniform(size=(N, L)).astype(np.float32)
b_np = np.random.uniform(size=(L, M)).astype(np.float32)
c_np = a_np.dot(b_np)

c_tvm = tvm.nd.empty(c_np.shape)
ts = time.time()
func_tuned(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)
ts2 = time.time()
print('Tuned: ' + str(ts2-ts))
ts = time.time()
func_untuned(tvm.nd.array(a_np), tvm.nd.array(b_np), c_tvm)
ts2 = time.time()
print('Un Tuned: ' + str(ts2-ts))
tvm.testing.assert_allclose(c_np, c_tvm.numpy(), rtol=1e-4)

'''
Tuned: 0.09827303886413574
Un Tuned: 0.20226001739501953
'''

'''
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
'''
