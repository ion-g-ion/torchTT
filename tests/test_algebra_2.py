"""
Test the advanced multilinear algebra operations between torchtt.TT objects.
Some operations (matvec for large ranks and elemntwise division) can be only computed using optimization (AMEN and DMRG).
"""
import pytest
import torchtt as tntt
import torch as tn
import numpy as np


def err_rel(t, ref): return tn.linalg.norm(t-ref).numpy() / \
    tn.linalg.norm(ref).numpy() if ref.shape == t.shape else np.inf


@pytest.mark.parametrize("dtype", [tn.float64, tn.complex128])
def test_dmrg_hadamard(dtype):
    """
    Test hadamard product using DMRG.
    """
    n = 32
    z = tntt.random([n]*8,[1]+7*[3]+[1], dtype=dtype)
    zm = z + z

    x = tntt.random([n]*8,[1]+7*[5]+[1], dtype=dtype)
    xm = x + x
    xm = xm + xm

    # conventional method 
    y = 8 * (z * x).round(1e-12)

    yf = tntt.dmrg_hadamard(zm, xm, eps=1e-12, verb=False)

    rel_error = (y-yf).norm().numpy()/y.norm().numpy()

    assert rel_error < 1e-12


@pytest.mark.parametrize("dtype", [tn.complex128])
def test_dmrg_matvec(dtype):
    """
    Test the fast matrix vector product using DMRG iterations.
    """
    n = 32
    A = tntt.random([(n, n)]*8, [1]+7*[3]+[1], dtype=dtype)
    Am = A + A

    x = tntt.random([n]*8, [1]+7*[5]+[1], dtype=dtype)
    xm = x + x
    xm = xm + xm

    # conventional method
    y = 8 * (A @ x).round(1e-12)

    # dmrg matvec
    yf = Am.fast_matvec(xm)

    rel_error = (y-yf).norm().numpy()/y.norm().numpy()

    assert rel_error < 1e-12


@pytest.mark.parametrize("dtype", [tn.complex128])
def test_dmrg_matvec_non_square(dtype):
    """
    Test the fast matrix vector product using DMRG iterations for non-square matrices.
    """
    n = 32
    A = tntt.random([(n+2,n)]*8,[1]+7*[3]+[1], dtype=dtype)
    Am = A + A 

    x = tntt.random([n]*8,[1]+7*[5]+[1], dtype=dtype)
    xm = x + x
    xm = xm + xm

    # conventional method 
    y = 8 * (A @ x).round(1e-12)

    # dmrg matvec
    yf = Am.fast_matvec(xm)

    rel_error = (y-yf).norm().numpy()/y.norm().numpy()

    assert rel_error < 1e-12
      

@pytest.mark.parametrize("dtype", [tn.float64])
def test_amen_division(dtype):
    """
    Test the division between tensors performed with AMEN optimization.
    """
    N = [7, 8, 9, 10]
    xs = tntt.meshgrid(
        [tn.linspace(0, 1, n, dtype=dtype) for n in N])
    x = xs[0]+xs[1]+xs[2]+xs[3]+xs[1]*xs[2]+(1-xs[3])*xs[2]+1
    x = x.round(0)
    y = tntt.ones(x.N, dtype=dtype)

    a = y/x
    b = 1/x
    c = tn.tensor(1.0)/x

    assert err_rel(a.full(), y.full()/x.full()) < 1e-11
    assert err_rel(b.full(), 1/x.full()) < 1e-11
    assert err_rel(c.full(), 1/x.full()) < 1e-11

@pytest.mark.parametrize("dtype", [tn.float64])
def test_amen_division_preconditioned(dtype):
    """
    Test the elementwise division using AMEN (use preconditioner for the local subsystem).
    """
    N = [7, 8, 9, 10]
    xs = tntt.meshgrid(
        [tn.linspace(0, 1, n, dtype=dtype) for n in N])
    x = xs[0]+xs[1]+xs[2]+xs[3]+xs[1]*xs[2]+(1-xs[3])*xs[2]+1
    x = x.round(0)
    y = tntt.ones(x.N)

    a = tntt.elementwise_divide(y, x, preconditioner='c')

    assert err_rel(a.full(), y.full()/x.full()) < 1e-11

@pytest.mark.parametrize("dtype", [tn.float64])
def test_amen_mv(dtype):
    """
    Test the AMEn matvec.
    """

    A = tntt.randn([(3, 4), (5, 6), (7, 8), (2, 3)], [1, 2, 2, 3, 1], dtype=dtype)
    x = tntt.randn([4, 6, 8, 3], [1, 4, 3, 3, 1], dtype=dtype)

    Cr = 25 * A @ x

    A = A + A + A + A + A
    x = x + x + x + x + x

    C = tntt.amen_mv(A, x)

    assert ((C-Cr).norm()/Cr.norm()) < 1e-11


@pytest.mark.parametrize("dtype", [tn.float64])
def test_amen_mv_zero_operator(dtype):
    """
    AMEn matvec should preserve exact zero output for a zero operator.
    """
    N = [3, 4, 2]
    A = tntt.zeros([(n, n) for n in N], dtype=dtype)
    x = tntt.randn(N, [1, 2, 2, 1], dtype=dtype)

    C = tntt.amen_mv(A, x, nswp=4, eps=1e-12, kickrank=2)

    assert C.N == N
    assert C.norm() < 1e-12


@pytest.mark.parametrize("dtype", [tn.float64])
def test_amen_mv_zero_vector(dtype):
    """
    AMEn matvec should handle zero right-hand tensors without NaNs.
    """
    N = [3, 4, 2]
    A = tntt.eye(N, dtype=dtype)
    x = tntt.zeros(N, dtype=dtype)

    C = tntt.amen_mv(A, x, nswp=4, eps=1e-12, kickrank=2)

    assert C.N == N
    assert C.norm() < 1e-12


@pytest.mark.parametrize("dtype", [tn.float64])
def test_amen_mv_identity(dtype):
    """
    AMEn matvec should reproduce an input tensor under the identity matrix.
    """
    N = [3, 4, 2]
    A = tntt.eye(N, dtype=dtype)
    x = tntt.randn(N, [1, 2, 2, 1], dtype=dtype)

    C = tntt.amen_mv(A, x, nswp=8, eps=1e-12, kickrank=2)

    assert ((C - x).norm() / x.norm()) < 1e-11

@pytest.mark.parametrize("dtype", [tn.float64])
def test_amen_mm(dtype):
    """
    Test the AMEn matmat.
    """

    A = tntt.randn([(3, 4), (5, 6), (7, 8), (2, 3)], [1, 2, 2, 3, 1], dtype=dtype)
    B = tntt.randn([(4, 2), (6, 4), (8, 5), (3, 7)], [1, 4, 3, 3, 1], dtype=dtype)

    Cr = 25 * A @ B

    A = A + A + A + A + A
    B = B + B + B + B + B

    C = tntt.amen_mm(A, B)

    assert ((C-Cr).norm()/Cr.norm()) < 1e-11


@pytest.mark.parametrize("dtype", [tn.float64])
def test_amen_mm_zero_left_factor(dtype):
    """
    AMEn matmat should return a zero matrix when the left factor is zero.
    """
    A = tntt.zeros([(3, 2), (4, 3)], dtype=dtype)
    B = tntt.randn([(2, 5), (3, 6)], [1, 2, 1], dtype=dtype)

    C = tntt.amen_mm(A, B, nswp=4, eps=1e-12, kickrank=2)

    assert C.M == [3, 4]
    assert C.N == [5, 6]
    assert C.norm() < 1e-12


@pytest.mark.parametrize("dtype", [tn.float64])
def test_amen_mm_zero_right_factor(dtype):
    """
    AMEn matmat should return a zero matrix when the right factor is zero.
    """
    A = tntt.eye([3, 4], dtype=dtype)
    B = tntt.zeros([(3, 2), (4, 5)], dtype=dtype)

    C = tntt.amen_mm(A, B, nswp=4, eps=1e-12, kickrank=2)

    assert C.M == [3, 4]
    assert C.N == [2, 5]
    assert C.norm() < 1e-12


@pytest.mark.parametrize("dtype", [tn.float64])
def test_amen_mm_identity_left_factor(dtype):
    """
    AMEn matmat should reproduce the right factor under a left identity.
    """
    A = tntt.eye([3, 4], dtype=dtype)
    B = tntt.randn([(3, 2), (4, 5)], [1, 2, 1], dtype=dtype)

    C = tntt.amen_mm(A, B, nswp=8, eps=1e-12, kickrank=2)

    assert ((C - B).norm() / B.norm()) < 1e-11


if __name__ == '__main__':
    pytest.main()

