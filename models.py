import jax.numpy as np
from jax import jit, grad, random
from jax.experimental import stax
from jax.experimental.stax import Relu, LogSoftmax
from jax.nn.initializers import glorot_normal, normal, ones, zeros


def GraphConvolution(out_dim, W_init=glorot_normal(), b_init=normal()):
    """
    Layer constructor function for a Graph Convolution layer similar to https://arxiv.org/abs/1609.02907
    """
    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        print(len(inputs))
        features, adj = inputs
        W, b = params
        support = np.dot(features, W) + b
        out = np.matmul(adj, support)
        return out

    return init_fun, apply_fun

def GCN(nhid, nclass, dropout):
    """
    This function implements the GCN model that uses 2 Graph Convolutional layers.
    The code is adapted from jax.experimental.stax.serial to be able to use
    the adjacency matrix as an argument to the GC layers but not the others.
    """
    gc1_init, gc1_fun = GraphConvolution(nhid)
    relu_init, relu_fun = Relu
    gc2_init, gc2_fun = GraphConvolution(nclass)
    ls_init, ls_fun = LogSoftmax

    init_funs = [gc1_init, relu_init, gc2_init, ls_init]
    nlayers = len(init_funs)

    def init_fun(rng, input_shape):
        params = []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        return input_shape, params

    def apply_fun(params, x, adj, **kwargs):
        rng = kwargs.pop('rng', None)
        rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
        
        x = gc1_fun(params[0], (x, adj), rng=rng)
        x = relu_fun(params[1], x, rng=rng)
        x = gc2_fun(params[2], (x, adj), rng=rng)
        x = ls_fun(params[3], x, rng=rng)
        return x
    
    return init_fun, apply_fun