import tvm
from tvm import te
import numpy as np
from tvm.contrib import graph_executor as runtime
from tvm.relay.op.contrib.cutlass import partition_for_cutlass
from tvm import relay
from tvm.relay import testing
import tvm.testing
from tvm.contrib.cutlass import finalize_modules

img_size = 8
def get_network():
    out_channels = 16
    batch_size = 1
    data = relay.var("data", relay.TensorType((batch_size, 3, img_size, img_size), "float16"))
    dense_weight = relay.var(
        "dweight", relay.TensorType((batch_size, 16 * img_size * img_size), "float16")
    )
    weight = relay.var("weight")
    bn_gamma = relay.var("bn_gamma")
    bn_beta = relay.var("bn_beta")
    bn_mmean = relay.var("bn_mean")
    bn_mvar = relay.var("bn_var")
    simple_net = relay.nn.conv2d(
        data=data, weight=weight, kernel_size=(3, 3), channels=out_channels, padding=(1, 1)
    )
    simple_net = relay.nn.batch_norm(simple_net, bn_gamma, bn_beta, bn_mmean, bn_mvar)[0]
    simple_net = relay.nn.relu(simple_net)
    simple_net = relay.nn.batch_flatten(simple_net)
    simple_net = relay.nn.dense(simple_net, dense_weight)
    simple_net = relay.Function(relay.analysis.free_vars(simple_net), simple_net)
    data_shape = (batch_size, 3, img_size, img_size)
    net, params = testing.create_workload(simple_net)
    return net, params, data_shape

cutlass = tvm.target.Target("llvm")


def cutlass_build(mod, target, params=None, target_host=None, mod_name="default"):
    target = [target, cutlass]
    lib = relay.build_module.build(
        mod, target=target, params=params, target_host=target_host, mod_name=mod_name
    )
    return lib



net, params, data_shape = get_network()
import inspect
import os

tutorial_dir = os.path.dirname(inspect.getfile(lambda: None))
print(tutorial_dir)
os.sys.path.append(os.path.join(tutorial_dir, "./"))
from test_pipeline_executor import graph_split

split_config = [{"op_name": "nn.relu", "op_index": 0}]
print(type(net["main"]))
subgraphs = graph_split(net["main"], split_config, params)
print(len(subgraphs))
print(type(subgraphs[0]))
print(type(subgraphs[1]))

from tvm.contrib import graph_executor, pipeline_executor, pipeline_executor_build
mod0, mod1 = subgraphs[0], subgraphs[1]
# Use cutlass as the codegen.
mod1 = partition_for_cutlass(mod1)

pipe_config = pipeline_executor_build.PipelineConfig()

pipe_config[mod0].target = "llvm"
pipe_config[mod0].dev = tvm.cpu(0)

pipe_config[mod1].target = "llvm"
pipe_config[mod1].dev = tvm.device("llvm", 0)
pipe_config[mod1].build_func = cutlass_build
# pipe_config[mod1].export_cc = "nvcc"
# Create the pipeline by connecting the subgraph modules.
# The global input will be forwarded to the input interface of the first module named mod0
pipe_config["input"]["data"].connect(pipe_config[mod0]["input"]["data"])
# The first output of mod0 will be forwarded to the input interface of mod1
pipe_config[mod0]["output"][0].connect(pipe_config[mod1]["input"]["data_n_0"])
# The first output of mod1 will be the first global output.
pipe_config[mod1]["output"][0].connect(pipe_config["output"][0])

with tvm.transform.PassContext(opt_level=3):
    pipeline_mod_factory = pipeline_executor_build.build(pipe_config)
