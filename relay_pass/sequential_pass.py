import numpy as np
import tvm
from tvm import te
import tvm.relay as relay

def example():
    shape = (1, 64, 54, 54)
    c_data = np.empty(shape).astype("float32")
    c = relay.const(c_data)
    weight = relay.var("weight", shape=(64, 64, 3, 3))
    x = relay.var("x", relay.TensorType((1, 64, 56, 56), "float32"))
    conv = relay.nn.conv2d(x, weight)
    y = relay.add(c, c)
    y = relay.multiply(y, relay.const(2, "float32"))
    y = relay.add(conv, y)
    z = relay.add(y, c)
    z1 = relay.add(y, c)
    z2 = relay.add(z, z1)
    return relay.Function([x, weight], z2)

# Let's first create a relay Module which contains one or multiple Relay
# functions for optimization.
f = example()
mod = tvm.IRModule.from_expr(f)
print("tvm.IRModule.from_expr: ")
print(type(mod))
print(mod)
# Glob the interested passes.
seq = tvm.transform.Sequential(
    [
        relay.transform.FoldConstant(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.FuseOps(fuse_opt_level=0),
    ]
)
mod = seq(mod)

print("tvm.transform.Sequential: ")
print(mod)
print(type(mod))
"""
tvm.IRModule.from_expr:
<class 'tvm.ir.module.IRModule'>
def @main(%x: Tensor[(1, 64, 56, 56), float32], %weight: Tensor[(64, 64, 3, 3), float32]) {
  %0 = add(meta[relay.Constant][0], meta[relay.Constant][0]);
  %1 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]);
  %2 = multiply(%0, 2f);
  %3 = add(%1, %2);
  %4 = add(%3, meta[relay.Constant][0]);
  %5 = add(%3, meta[relay.Constant][0]);
  add(%4, %5)
}


tvm.transform.Sequential:
def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %0 = fn (%p03: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %p13: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    nn.conv2d(%p03, %p13, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 56, 56), float32], Tensor[(64, 64, 3, 3), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %1 = %0(%x, %weight) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %2 = fn (%p02: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, %p12: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    add(%p02, %p12) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 54, 54), float32], Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %3 = %2(%1, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %4 = fn (%p01: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, %p11: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    add(%p01, %p11) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 54, 54), float32], Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %5 = fn (%p04: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, %p14: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    add(%p04, %p14) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 54, 54), float32], Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %6 = %4(%3, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %7 = %5(%3, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %8 = fn (%p0: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, %p1: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    add(%p0, %p1) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 54, 54), float32], Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %8(%6, %7) /* ty=Tensor[(1, 64, 54, 54), float32] */
}


<class 'tvm.ir.module.IRModule'>

"""
