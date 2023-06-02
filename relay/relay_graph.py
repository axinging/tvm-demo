from tvm import te
import numpy as np
from tvm.contrib import graph_executor as runtime
from tvm.relay.op.contrib.cutlass import partition_for_cutlass
from tvm import relay
from tvm.relay import testing
import tvm.testing
from tvm.contrib.cutlass import finalize_modules
import threading
from tvm.contrib import graph_executor

img_size = 4
batch_size = 1
out_channels = 16

def get_network():
    data = relay.var("data", relay.TensorType((batch_size, 3, img_size, img_size), "float16"))
    dense_weight = relay.var(
        "dweight", relay.TensorType((batch_size, 4 * img_size * img_size), "float16")
    )
    weight = relay.var("weight")
    bn_gamma = relay.var("bn_gamma")
    bn_beta = relay.var("bn_beta")
    bn_mmean = relay.var("bn_mean")
    bn_mvar = relay.var("bn_var")
    simple_net = relay.nn.conv2d(
        data=data, weight=weight, kernel_size=(3, 3), channels=out_channels, padding=(0, 0)
    )
    simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
    simple_net = relay.nn.relu(simple_net)
    simple_net = relay.nn.batch_flatten(simple_net)
    simple_net = relay.nn.dense(simple_net, dense_weight)
    simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)
    data_shape = (batch_size, 3, img_size, img_size)
    net, params = testing.create_workload(simple_net)
    return net, params, data_shape


net, params, data_shape = get_network()

print("print net: ")
print(type(net))
print((net))
print("print net.script: ")
print(type(net.script()))
print((net.script()))
"""
print net:
<class 'tvm.ir.module.IRModule'>
def @main(%data: Tensor[(1, 3, 4, 4), float16] /* ty=Tensor[(1, 3, 4, 4), float16] */, %weight: Tensor[(16, 3, 3, 3), float16] /* ty=Tensor[(16, 3, 3, 3), float16] */, %bn_gamma: Tensor[(16), float16] /* ty=Tensor[(16), float16] */, %bn_beta: Tensor[(16), float16] /* ty=Tensor[(16), float16] */, %bn_mean: Tensor[(16), float16] /* ty=Tensor[(16), float16] */, %bn_var: Tensor[(16), float16] /* ty=Tensor[(16), float16] */, %dweight: Tensor[(1, 64), float16] /* ty=Tensor[(1, 64), float16] */) -> Tensor[(1, 1), float16] {
  %0 = nn.conv2d(%data, %weight, padding=[0, 0, 0, 0], channels=16, kernel_size=[3, 3]) /* ty=Tensor[(1, 16, 2, 2), float16] */;
  %1 = nn.batch_norm(%0, %bn_gamma, %bn_beta, %bn_mean, %bn_var) /* ty=(Tensor[(1, 16, 2, 2), float16], Tensor[(16), float16], Tensor[(16), float16]) */;
  %2 = %1.0 /* ty=Tensor[(1, 16, 2, 2), float16] */;
  %3 = nn.relu(%2) /* ty=Tensor[(1, 16, 2, 2), float16] */;
  %4 = nn.batch_flatten(%3) /* ty=Tensor[(1, 64), float16] */;
  nn.dense(%4, %dweight, units=None) /* ty=Tensor[(1, 1), float16] */
}

print net.script:
<class 'tvm.runtime.container.String'>
# from tvm.script import tir as T
@tvm.script.ir_module
class Module:
"""
lib = relay.build(net, tvm.target.Target("llvm"), params=params)
print(type(lib))

# print(lib.get_graph_json())
print(lib.get_lib())
print(lib["default"])

dev = tvm.cpu(0)
print(lib["default"](dev))

gmod = graph_executor.GraphModule(lib["default"](dev))
# use the graph module.
data = tvm.nd.array((np.random.uniform(size=(batch_size, 3, img_size, img_size))).astype("float16"))
gmod.set_input("data", data)
gmod.run()
tvm_output = gmod.get_output(0)
print(tvm_output.shape)
# print(tvm_output)
