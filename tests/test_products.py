"""
Test the products with rank truncation (torchtt.matvec, torchtt.matmat, torchtt.hadamard) and their methods.
"""
import dataclasses
import pytest
import torchtt as tntt
import torch as tn
import numpy as np


def err_rel(t, ref): return (tn.linalg.norm(t-ref).numpy() / tn.linalg.norm(ref).numpy()
                             if tn.linalg.norm(ref).numpy() > 0 else tn.linalg.norm(t-ref).numpy()) if ref.shape == t.shape else np.inf


def inflate(t, k):
    """
    The same tensor with the rank multiplied by k (redundant representation).
    """
    s = t
    for _ in range(k-1):
        s = s + t
    return s * (1.0 / k)


parameters = [tn.float64, tn.complex128]


def _operands_mv(dtype):
    M = [3, 2, 2, 4]
    N = [2, 3, 4, 2]
    A = tntt.random([(m, n) for m, n in zip(M, N)], [1, 3, 2, 3, 1], dtype=dtype)
    x = tntt.random(N, [1, 2, 2, 5, 1], dtype=dtype)
    return inflate(A, 2), inflate(x, 3), A @ x


def _operands_mm(dtype):
    M = [3, 2, 2, 4]
    N = [2, 3, 4, 2]
    K = [4, 2, 3, 3]
    A = tntt.random([(m, n) for m, n in zip(M, N)], [1, 3, 2, 3, 1], dtype=dtype)
    B = tntt.random([(n, k) for n, k in zip(N, K)], [1, 2, 2, 5, 1], dtype=dtype)
    return inflate(A, 2), inflate(B, 3), A @ B


def _operands_hadamard(dtype, ttm):
    shape = [(3, 2), (2, 3), (2, 4), (4, 2)] if ttm else [2, 3, 4, 2, 3]
    d = len(shape)
    x = tntt.random(shape, [1] + [3, 2, 3, 4, 2][:d-1] + [1], dtype=dtype)
    y = tntt.random(shape, [1] + [2, 2, 5, 4, 3][:d-1] + [1], dtype=dtype)
    return inflate(x, 2), inflate(y, 3), x * y


@pytest.mark.parametrize("dtype", parameters)
@pytest.mark.parametrize("method", ['direct', 'dmrg', 'amen', 'swap'])
def test_matvec(method, dtype):
    """
    Test the matrix-vector product with every method.
    """
    if method == 'amen' and dtype == tn.complex128:
        pytest.skip("AMEn supports only real tensors.")
    A, x, y_ref = _operands_mv(dtype)

    y = tntt.matvec(A, x, 1e-12, method=method)

    assert y.N == y_ref.N
    assert err_rel(y.full(), y_ref.full()) < 1e-10


@pytest.mark.parametrize("dtype", parameters)
@pytest.mark.parametrize("method", ['direct', 'amen', 'swap'])
def test_matmat(method, dtype):
    """
    Test the product of TT matrices with every method.
    """
    if method == 'amen' and dtype == tn.complex128:
        pytest.skip("AMEn supports only real tensors.")
    A, B, C_ref = _operands_mm(dtype)

    C = tntt.matmat(A, B, 1e-12, method=method)

    assert C.M == C_ref.M
    assert C.N == C_ref.N
    assert err_rel(C.full(), C_ref.full()) < 1e-10


@pytest.mark.parametrize("dtype", parameters)
@pytest.mark.parametrize("ttm", [False, True])
@pytest.mark.parametrize("method", ['direct', 'dmrg', 'swap'])
def test_hadamard(method, ttm, dtype):
    """
    Test the elementwise product of TT tensors and of TT matrices with every method.
    """
    x, y, z_ref = _operands_hadamard(dtype, ttm)

    z = tntt.hadamard(x, y, 1e-12, method=method)

    assert z.is_ttm == ttm
    assert z.N == z_ref.N
    if ttm:
        assert z.M == z_ref.M
    assert err_rel(z.full(), z_ref.full()) < 1e-10


def test_method_instances():
    """
    Test the methods given as instances with changed parameters.
    """
    A, x, y_ref = _operands_mv(tn.float64)
    B, C, C_ref = _operands_mm(tn.float64)

    y = tntt.matvec(A, x, 1e-12, method=tntt.methods.DMRG(nswp=10, kickrank=2, use_cpp=False))
    assert err_rel(y.full(), y_ref.full()) < 1e-10

    y = tntt.matvec(A, x, 1e-12, method=tntt.methods.AMEn(nswp=10, kickrank=6, kick2=1))
    assert err_rel(y.full(), y_ref.full()) < 1e-10

    C = tntt.matmat(B, C, 1e-12, method=tntt.methods.AMEn(kickrank=8))
    assert err_rel(C.full(), C_ref.full()) < 1e-10


def test_tt_methods():
    """
    Test the methods of torchtt.TT that call the products.
    """
    A, x, y_ref = _operands_mv(tn.float64)
    B, C, C_ref = _operands_mm(tn.float64)
    u, v, w_ref = _operands_hadamard(tn.float64, False)

    assert err_rel(A.matvec(x, 1e-12).full(), y_ref.full()) < 1e-10
    assert err_rel(A.matvec(x, 1e-12, method='direct').full(), y_ref.full()) < 1e-10
    assert err_rel(B.matmat(C, 1e-12).full(), C_ref.full()) < 1e-10
    assert err_rel(u.hadamard(v, 1e-12).full(), w_ref.full()) < 1e-10


@pytest.mark.parametrize("method", ['dmrg', tntt.methods.DMRG(use_cpp=False), 'amen'])
def test_initial_guess(method):
    """
    Test that an initial guess is used and not modified.
    """
    A, x, y_ref = _operands_mv(tn.float64)
    y0 = tntt.random(A.M, [1, 2, 2, 2, 1], dtype=tn.float64)
    cores0 = [c.clone() for c in y0.cores]

    y = tntt.matvec(A, x, 1e-12, method=method, initial=y0)

    assert err_rel(y.full(), y_ref.full()) < 1e-10
    assert y0.R == [1, 2, 2, 2, 1]
    assert all(tn.equal(c, c0) for c, c0 in zip(y0.cores, cores0))


@pytest.mark.parametrize("method", ['direct', 'dmrg', 'amen', 'swap'])
def test_rmax(method):
    """
    Test that the rank of the result does not exceed rmax.
    """
    A, x, _ = _operands_mv(tn.float64)

    y = tntt.matvec(A, x, 1e-12, rmax=3, method=method)

    assert max(y.R) <= 3


def test_invalid_methods():
    """
    Test the errors for methods that are unknown, do not implement a product or do not use an initial guess.
    """
    A, x, _ = _operands_mv(tn.float64)
    B, C, _ = _operands_mm(tn.float64)
    u, v, _ = _operands_hadamard(tn.float64, False)

    with pytest.raises(tntt.errors.InvalidArguments):
        tntt.matvec(A, x, method='dmrgg')
    with pytest.raises(tntt.errors.InvalidArguments):
        tntt.matvec(A, x, method=3)
    with pytest.raises(tntt.errors.InvalidArguments):
        tntt.matmat(B, C, method='dmrg')
    with pytest.raises(tntt.errors.InvalidArguments):
        tntt.hadamard(u, v, method=tntt.methods.AMEn())
    with pytest.raises(tntt.errors.InvalidArguments):
        tntt.matvec(A, x, method='direct', initial=tntt.random(A.M, 2))
    with pytest.raises(tntt.errors.InvalidArguments):
        tntt.hadamard(u, v, method='swap', initial=u)
    with pytest.raises(tntt.errors.IncompatibleTypes):
        tntt.matvec(A.to(dtype=tn.complex128), x.to(dtype=tn.complex128), method='amen')


def test_invalid_operands():
    """
    Test the errors for operands of the wrong type or shape.
    """
    A, x, _ = _operands_mv(tn.float64)
    B, C, _ = _operands_mm(tn.float64)
    u, v, _ = _operands_hadamard(tn.float64, False)

    with pytest.raises(tntt.errors.IncompatibleTypes):
        tntt.matvec(x, A)
    with pytest.raises(tntt.errors.ShapeMismatch):
        tntt.matvec(A, tntt.random(A.M, 2))
    with pytest.raises(tntt.errors.ShapeMismatch):
        tntt.matvec(A, x, initial=tntt.random(A.N, 2))
    with pytest.raises(tntt.errors.IncompatibleTypes):
        tntt.matmat(B, x)
    with pytest.raises(tntt.errors.ShapeMismatch):
        tntt.matmat(C, B)
    with pytest.raises(tntt.errors.IncompatibleTypes):
        tntt.hadamard(u, A)
    with pytest.raises(tntt.errors.ShapeMismatch):
        tntt.hadamard(u, tntt.random(u.N[::-1], 2))
    with pytest.raises(tntt.errors.InvalidArguments):
        tntt.hadamard(u, u.full())


def test_method_parameters():
    """
    Test that the method parameters are checked and cannot be changed.
    """
    with pytest.raises(tntt.errors.InvalidArguments):
        tntt.methods.DMRG(nswp=0)
    with pytest.raises(tntt.errors.InvalidArguments):
        tntt.methods.DMRG(nswp=2.5)
    with pytest.raises(tntt.errors.InvalidArguments):
        tntt.methods.AMEn(kickrank=-1)
    with pytest.raises(TypeError):
        tntt.methods.DMRG(nswpp=10)
    with pytest.raises(TypeError):
        tntt.methods.Swap(nswp=10)

    method = tntt.methods.DMRG(nswp=40)
    with pytest.raises(dataclasses.FrozenInstanceError):
        method.nswp = 10
    assert method == tntt.methods.DMRG(nswp=40)


def test_deprecated():
    """
    Test that the deprecated functions warn and give the same results.
    """
    A, x, y_ref = _operands_mv(tn.float64)
    B, C, C_ref = _operands_mm(tn.float64)
    u, v, w_ref = _operands_hadamard(tn.float64, False)
    U, V, W_ref = _operands_hadamard(tn.float64, True)

    with pytest.warns(DeprecationWarning):
        assert err_rel(A.fast_matvec(x).full(), y_ref.full()) < 1e-10
    with pytest.warns(DeprecationWarning):
        assert err_rel(tntt.fast_mv(A, x, 1e-12).full(), y_ref.full()) < 1e-10
    with pytest.warns(DeprecationWarning):
        assert err_rel(tntt.amen_mv(A, x, eps=1e-12).full(), y_ref.full()) < 1e-10
    with pytest.warns(DeprecationWarning):
        assert err_rel(tntt.fast_mm(B, C, 1e-12).full(), C_ref.full()) < 1e-10
    with pytest.warns(DeprecationWarning):
        assert err_rel(tntt.amen_mm(B, C, eps=1e-12).full(), C_ref.full()) < 1e-10
    with pytest.warns(DeprecationWarning):
        assert err_rel(tntt.fast_hadammard(u, v, 1e-12).full(), w_ref.full()) < 1e-10
    with pytest.warns(DeprecationWarning):
        assert err_rel(tntt.fast_hadammard(U, V, 1e-12).full(), W_ref.full()) < 1e-10
    with pytest.warns(DeprecationWarning):
        assert err_rel(tntt.dmrg_hadamard(u, v, eps=1e-12).full(), w_ref.full()) < 1e-10


if __name__ == '__main__':
    pytest.main()
