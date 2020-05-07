import jax

@jax.partial(jax.jit, static_argnums=(2))
def sp_matmul(A, B, shape):
    """
    Arguments:
        A: (N, M) sparse matrix represented as a tuple (indexes, values)
        B: (M,K) dense matrix
        shape: value of N
    Returns:
        (N, K) dense matrix
    """
    assert B.ndim == 2
    indexes, values = A
    target, source = indexes
    in_ = B.take(source, axis=0)
    prod = in_*values[:, None]
    res = jax.ops.segment_sum(prod, target, shape)
    return res