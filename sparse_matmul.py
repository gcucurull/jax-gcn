import jax

@jax.partial(jax.jit, static_argnums=(2))  # output shape depends on A_shape
def _sp_matmul(A, B, A_shape):
    indexes, values = A
    target, source = indexes
    in_ = B.take(source, axis=0)
    prod = in_*values[:, None]
    res = jax.ops.segment_sum(prod, target, A_shape[0])
    return res

def sp_matmul(A, B):
    """
    Arguments:
        A: (N, M) sparse matrix represented as a tuple (indexes, values, shape)
        B: (M,K) dense matrix
    Returns:
        (N, K) dense matrix
    """
    indexes, values, shape = A
    assert B.ndim == 2 and len(shape) == 2
    return _sp_matmul((indexes, values), B, shape)