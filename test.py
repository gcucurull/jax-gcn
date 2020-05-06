import numpy
import jax.numpy as np

from sparse_matmul import sp_matmul

tolerance = 1e-5
sparsity = 0.95

def distance(A, B):
    diff = np.abs(A-B).mean()
    return diff

def test_square_sparse_matrix_1():
    rows = 2000
    mask  = numpy.random.rand(rows, rows) < sparsity
    A = numpy.random.rand(rows, rows)
    A[mask] = 0.0
    B = numpy.random.rand(rows, 1)

    indexes = A.nonzero()
    values = A[indexes]
    sp_A = (indexes, values, A.shape)

    sp_res = sp_matmul(sp_A, B)
    res = np.matmul(A, B)

    diff = distance(sp_res, res)
    print(diff)
    assert diff < tolerance

def test_square_sparse_matrix_fail():
    rows = 2000
    mask  = numpy.random.rand(rows, rows) < sparsity
    A = numpy.random.rand(rows, rows)
    A[mask] = 0.0
    B = numpy.random.rand(rows, 1)

    indexes = A.nonzero()
    values = A[indexes]
    sp_A = (indexes, values, A.shape)

    sp_res = sp_matmul(sp_A, B)
    res = np.matmul(A.T, B)

    diff = distance(sp_res, res)
    print(diff)
    assert diff > tolerance

def test_rectangular_matrix_1():
    rows, cols = 2000, 300
    mask = numpy.random.rand(rows, cols) < sparsity
    A = numpy.random.rand(rows, cols)
    A[mask] = 0.0
    B = numpy.random.rand(cols, 1)

    indexes = A.nonzero()
    values = A[indexes]
    sp_A = (indexes, values, A.shape)

    sp_res = sp_matmul(sp_A, B)
    res = np.matmul(A, B)

    diff = distance(sp_res, res)
    print(diff)
    assert diff < tolerance

def test_rectangular_matrix_2():
    rows, cols = 300, 2000
    mask = numpy.random.rand(rows, cols) < sparsity
    A = numpy.random.rand(rows, cols)
    A[mask] = 0.0
    B = numpy.random.rand(cols, 1)

    indexes = A.nonzero()
    values = A[indexes]
    sp_A = (indexes, values, A.shape)

    sp_res = sp_matmul(sp_A, B)
    res = np.matmul(A, B)

    diff = distance(sp_res, res)
    print(diff)
    assert diff < tolerance

def test_rectangular_matrix_3():
    rows, cols = 2000, 300
    mask = numpy.random.rand(rows, cols) < sparsity
    A = numpy.random.rand(rows, cols)
    A[mask] = 0.0
    B = numpy.random.rand(cols, 64)

    indexes = A.nonzero()
    values = A[indexes]
    sp_A = (indexes, values, A.shape)

    sp_res = sp_matmul(sp_A, B)
    res = np.matmul(A, B)

    diff = distance(sp_res, res)
    print(diff)
    assert diff < tolerance

def test_square_sparse_matrix_2():
    rows = 2000
    mask  = numpy.random.rand(rows, rows) < sparsity
    A = numpy.random.rand(rows, rows)
    A[mask] = 0.0
    B = numpy.random.rand(rows, 128)

    indexes = A.nonzero()
    values = A[indexes]
    sp_A = (indexes, values, A.shape)

    sp_res = sp_matmul(sp_A, B)
    res = np.matmul(A, B)

    diff = distance(sp_res, res)
    print(diff)
    assert diff < tolerance
 
def test_square_sparse_matrix_fail_2():
    rows = 2000
    mask  = numpy.random.rand(rows, rows) < sparsity
    A = numpy.random.rand(rows, rows)
    A[mask] = 0.0
    B = numpy.random.rand(rows, 128)

    indexes = A.nonzero()
    values = A[indexes]
    sp_A = (indexes, values, A.shape)

    sp_res = sp_matmul(sp_A, B)
    res = np.matmul(A, numpy.random.rand(*B.shape))

    diff = distance(sp_res, res)
    print(diff)
    assert diff > tolerance