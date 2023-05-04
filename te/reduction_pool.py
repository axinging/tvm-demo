import tvm
import tvm.testing
from tvm import te
import numpy as np
import d2ltvm

# Save to the d2ltvm package.
def pool(pool_type, c, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
    """2D pooling

    pool_type: pooling type, 'max' or 'avg'
    c : channels
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding sizes, default 0
    sh, sw : height and width strides, default 1
    """
    # reduction axes
    rkh = te.reduce_axis((0, kh), name='rkh')
    rkw = te.reduce_axis((0, kw), name='rkw')
    # output height and weights
    oh = d2ltvm.conv_out_size(nh, kh, ph, sh)
    ow = d2ltvm.conv_out_size(nw, kw, pw, sw)
    # pad X and then compute Y
    X = te.placeholder((c, nh, nw), name='X')


    if pool_type == 'max':
        PaddedX = d2ltvm.padding(X, ph, pw, val=te.min_value(X.dtype)) \
            if ph * pw != 0 else X
        Y = te.compute((c, oh, ow), \
                            lambda c, h, w: \
                            te.max(PaddedX[c, h*sh+rkh, w*sw+rkw], \
                                axis=[rkh, rkw]), \
                            tag="pool_max", name='PoolMax')
    elif pool_type == 'avg':
        PaddedX = d2ltvm.padding(X, ph, pw) if ph * pw != 0 else X
        tsum = te.compute((c, oh, ow), \
                            lambda c, h, w: \
                            te.sum(PaddedX[c, h*sh+rkh, w*sw+rkw], \
                                axis=[rkh, rkw]), \
                            tag="pool_avg1", name='PoolSum')
        Y = te.compute((c, oh, ow), \
                            lambda c, h, w: \
                            tsum[c, h, w] / (kh*kw), \
                            tag='pool_avg2', name='PoolAvg')
    else:
        raise ValueError("Pool type should be 'avg' or 'max'.")
    return X, Y, PaddedX

c, n, k, p, s = 4, 12, 3, 1, 1
X, Y, PaddedX = pool('max', c, n, n, k, k, p, p, s, s)
sch = te.create_schedule(Y.op)
mod = tvm.build(sch, [X, Y])
print(tvm.lower(sch, [X, Y], simple_mode=True))
data, _, out_max = d2ltvm.get_conv_data(c, c, n, k, p, s, tvm.nd.array)
mod(data, out_max)

print(out_max.numpy())
