import math

import jax.numpy as np
from jax import lax, random
from jax.experimental import stax
from jax.experimental.stax import Relu, LogSoftmax
from jax.nn.initializers import glorot_normal, glorot_uniform, normal, uniform, zeros
import jax.nn as nn


def Dropout(rate):
    """
    Layer construction function for a dropout layer with given rate.
    This Dropout layer is modified from stax.experimental.Dropout, to use
    `is_training` as an argument to apply_fun, instead of defining it at
    definition time.

    Arguments:
        rate (float): Probability of keeping and element.
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
        out = lax.cond(is_training, outs, lambda x: x, inputs, lambda x: x)
        return out
    return init_fun, apply_fun


def GraphConvolution(out_dim, bias=False):
    """
    Layer constructor function for a Graph Convolution layer similar to https://arxiv.org/abs/1609.02907
    """
    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        W_init, b_init = glorot_uniform(), zeros
        W = W_init(k1, (input_shape[-1], out_dim))
        if bias:
            b = b_init(k2, (out_dim,))
        else:
            b = None
        return output_shape, (W, b)

    def apply_fun(params, x, adj, **kwargs):
        W, b = params
        support = np.dot(x, W)
        out = np.matmul(adj, support)
        if bias:
            out += b
        return out

    return init_fun, apply_fun

def GCN(nhid, nclass, dropout):
    """
    This function implements the GCN model that uses 2 Graph Convolutional layers.
    The code is adapted from jax.experimental.stax.serial to be able to use
    the adjacency matrix as an argument to the GC layers but not the others.
    """
    gc1_init, gc1_fun = GraphConvolution(nhid)
    _, drop_fun = Dropout(dropout)
    gc2_init, gc2_fun = GraphConvolution(nclass)

    init_funs = [gc1_init, gc2_init]

    def init_fun(rng, input_shape):
        params = []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        return input_shape, params

    def apply_fun(params, x, adj, is_training=False, **kwargs):
        rng = kwargs.pop('rng', None)
        k1, k2, k3, k4 = random.split(rng, 4)

        x = drop_fun(None, x, is_training=is_training, rng=k1)
        x = gc1_fun(params[0], x, adj, rng=k2)
        x = nn.relu(x)
        x = drop_fun(None, x, is_training=is_training, rng=k3)
        x = gc2_fun(params[1], x, adj, rng=k4)
        x = nn.log_softmax(x)
        return x
    
    return init_fun, apply_fun