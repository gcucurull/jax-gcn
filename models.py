import jax.numpy as np
from jax import lax, random
from jax.experimental import stax
from jax.experimental.stax import Relu, LogSoftmax
from jax.nn.initializers import glorot_normal, normal, ones, zeros


def Dropout(rate):
    """
    Layer construction function for a dropout layer with given rate.
    This Dropout layer is modified from stax.experimental.Dropout, to use
    `is_training` as an argument to apply_fun, instead of defining it at
    definition time.
    """
    def init_fun(rng, input_shape):
        return input_shape, ()
    def apply_fun(params, inputs, is_training, **kwargs):
        rng = kwargs.get('rng', None)
        if rng is None:
            msg = ("Dropout layer requires apply_fun to be called with a PRNG key "
                   "argument. That is, instead of `apply_fun(params, inputs)`, call "
                   "it like `apply_fun(params, inputs, rng)` where `rng` is a "
                   "jax.random.PRNGKey value.")
            raise ValueError(msg)
        keep = random.bernoulli(rng, rate, inputs.shape)
        outs = np.where(keep, inputs / rate, 0)
        # if not training, just return inputs and discard any computation done
        return lax.cond(is_training, outs, lambda x: x, inputs, lambda x: x)
    return init_fun, apply_fun


def GraphConvolution(out_dim, W_init=glorot_normal(), b_init=normal()):
    """
    Layer constructor function for a Graph Convolution layer similar to https://arxiv.org/abs/1609.02907
    """
    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
        return output_shape, (W, b)

    def apply_fun(params, x, adj, **kwargs):
        W, b = params
        support = np.dot(x, W) + b
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
    drop_init, drop_fun = Dropout(dropout)
    gc2_init, gc2_fun = GraphConvolution(nclass)
    ls_init, ls_fun = LogSoftmax

    init_funs = [gc1_init, relu_init, drop_init, gc2_init, ls_init]
    nlayers = len(init_funs)

    def init_fun(rng, input_shape):
        params = []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        return input_shape, params

    def apply_fun(params, x, adj, is_training=False, **kwargs):
        rng = kwargs.pop('rng', None)
        rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
        
        x = gc1_fun(params[0], x, adj, rng=rng)
        x = relu_fun(params[1], x, rng=rng)
        x = drop_fun(params[2], x, is_training=is_training, rng=rng)
        x = gc2_fun(params[3], x, adj, rng=rng)
        x = ls_fun(params[4], x, rng=rng)
        return x
    
    return init_fun, apply_fun