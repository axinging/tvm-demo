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

# Now we can apply constant folding on the module.
# fold_const here is a callback that doesn't take any parameters.
fold_const = relay.transform.FoldConstant()
# Then, we can invoke the pass on the given module. Note that the constant
# folding pass works at the function-level. That being said, each function in
# the module will be applied with the optimization. Users don't need to iterate
# through individual functions manually to apply this pass.
mod = fold_const(mod)
# We can see from the updated program that the constants are folded.
print(type(mod))
print(mod)

mod = relay.transform.EliminateCommonSubexpr()(mod)
print(mod)
print(type(mod))

mod = relay.transform.FuseOps(fuse_opt_level=0)(mod)

# We can observe that the optimized module contains functions that only have
# a signle primitive op.
print(type(mod))
print(mod)

'''
def @main(%x: Tensor[(1, 64, 56, 56), float32], %weight: Tensor[(64, 64, 3, 3), float32]) {
  %0 = add(meta[relay.Constant][0], meta[relay.Constant][0]);
  %1 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]);
  %2 = multiply(%0, 2f);
  %3 = add(%1, %2);
  %4 = add(%3, meta[relay.Constant][0]);
  %5 = add(%3, meta[relay.Constant][0]);
  add(%4, %5)
}


def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %0 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %1 = add(%0, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %2 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %3 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  add(%2, %3) /* ty=Tensor[(1, 64, 54, 54), float32] */
}


def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %0 = nn.conv2d(%x, %weight, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %1 = add(%0, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %2 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  add(%2, %2) /* ty=Tensor[(1, 64, 54, 54), float32] */
}


def @main(%x: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %weight: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */) -> Tensor[(1, 64, 54, 54), float32] {
  %0 = fn (%p03: Tensor[(1, 64, 56, 56), float32] /* ty=Tensor[(1, 64, 56, 56), float32] */, %p12: Tensor[(64, 64, 3, 3), float32] /* ty=Tensor[(64, 64, 3, 3), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    nn.conv2d(%p03, %p12, padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 56, 56), float32], Tensor[(64, 64, 3, 3), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %1 = %0(%x, %weight) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %2 = fn (%p02: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, %p11: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    add(%p02, %p11) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 54, 54), float32], Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %3 = %2(%1, meta[relay.Constant][0] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %4 = fn (%p01: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, %p1: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    add(%p01, %p1) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 54, 54), float32], Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %5 = %4(%3, meta[relay.Constant][1] /* ty=Tensor[(1, 64, 54, 54), float32] */) /* ty=Tensor[(1, 64, 54, 54), float32] */;
  %6 = fn (%p0: Tensor[(1, 64, 54, 54), float32] /* ty=Tensor[(1, 64, 54, 54), float32] */, Primitive=1) -> Tensor[(1, 64, 54, 54), float32] {
    add(%p0, %p0) /* ty=Tensor[(1, 64, 54, 54), float32] */
  } /* ty=fn (Tensor[(1, 64, 54, 54), float32]) -> Tensor[(1, 64, 54, 54), float32] */;
  %6(%5) /* ty=Tensor[(1, 64, 54, 54), float32] */
}
'''
