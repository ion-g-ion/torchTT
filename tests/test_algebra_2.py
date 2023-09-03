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


if __name__ == '__main__':
    pytest.main()

