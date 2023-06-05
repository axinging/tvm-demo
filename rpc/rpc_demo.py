# https://tvm.apache.org/docs/tutorial/cross_compilation_and_rpc.html

import numpy as np

import tvm
from tvm import te
from tvm import rpc
from tvm.contrib import utils

n = tvm.runtime.convert(1024)
A = te.placeholder((n,), name="A")
B = te.compute((n,), lambda i: A[i] + 1.0, name="B")
s = te.create_schedule(B.op)

local_demo = True

if local_demo:
    target = "llvm"
else:
    target = "llvm -mtriple=armv7l-linux-gnueabihf"

func = tvm.build(s, [A, B], target=target, name="add_one")
# save the lib at a local temp folder
temp = utils.tempdir()
path = temp.relpath("lib.tar")
func.export_library(path)
