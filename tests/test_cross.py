"""
Test the cross approximation method.
"""
import torchtt as tntt
import torch as tn
import numpy as np
import pytest

err_rel = lambda t, ref: tn.linalg.norm(t - ref).numpy() / tn.linalg.norm(ref).numpy() if ref.shape == t.shape else np.inf


def test_dmrg_cross_interpolation():
    """
    Test the DMRG cross interpolation method.
    """
    func1 = lambda I: 1 / (2 + tn.sum(I + 1, 1).to(dtype=tn.float64))
    N = [20] * 4
    x = tntt.interpolate.dmrg_cross(func1, N, eps=1e-7)
    Is = tntt.meshgrid([tn.arange(0, n, dtype=tn.float64) for n in N])
    x_ref = 1 / (2 + Is[0].full() + Is[1].full() + Is[2].full() + Is[3].full() + 4)

    assert err_rel(x.full(), x_ref) < 1e-6

def test_dmrg_cross_interpolation_nonvect():
    """
    Test the DMRG cross interpolation method for non vectorized function.
    """
    func1 = lambda I,J,K,L: 1 / (6 + I + J + K + L)
    N = [20] * 4
    x = tntt.interpolate.dmrg_cross(func1, N, eps=1e-7, eval_vect=False)
    Is = tntt.meshgrid([tn.arange(0, n, dtype=tn.float64) for n in N])
    x_ref = 1 / (2 + Is[0].full() + Is[1].full() + Is[2].full() + Is[3].full() + 4)

    assert err_rel(x.full(), x_ref) < 1e-6


def test_dmrg_cross_zero_function():
    """
    Test that DMRG cross handles exactly zero tensors without relative-error
    normalization by zero.
    """
    N = [5, 4, 3]
    func = lambda I: tn.zeros(I.shape[0], dtype=tn.float64)

    x = tntt.interpolate.dmrg_cross(func, N, eps=1e-8, nswp=5)

    assert x.N == N
    assert x.norm() < 1e-12

@pytest.mark.parametrize("method", ["dmrg", "amen"])
def test_function_interpolate_multivariable(method):
    """
    Test the cross interpolation method for function approximation.
    """
    func1 = lambda I: 1 / (2 + tn.sum(I + 1, 1).to(dtype=tn.float64))
    N = [20] * 4

    Is = tntt.meshgrid([tn.arange(0, n, dtype=tn.float64) for n in N])
    x_ref = 1 / (2 + Is[0].full() + Is[1].full() + Is[2].full() + Is[3].full() + 4)

    y = tntt.interpolate.function_interpolate(func1, Is, 1e-8, method=method)
    assert err_rel(y.full(), x_ref) < 1e-7


@pytest.mark.skipif(not tn.cuda.is_available(), reason="CUDA device is not available.")
@pytest.mark.parametrize("method", ["dmrg", "amen"])
def test_function_interpolate_cuda_multivariable(method):
    """
    Interpolation should keep index state on CUDA when inputs are CUDA tensors.
    """
    N = [4, 5]
    Is = tntt.meshgrid([tn.linspace(0, 1, n, dtype=tn.float64, device="cuda") for n in N])
    start_tens = tntt.ones(N, dtype=tn.float64, device="cuda")

    func = lambda values: values[:, 0] + 2 * values[:, 1]
    y = tntt.interpolate.function_interpolate(
        func,
        Is,
        eps=1e-4,
        start_tens=start_tens,
        nswp=2,
        kick=1,
        method=method,
    )

    ref = Is[0].full() + 2 * Is[1].full()
    rel_err = tn.linalg.norm(y.full() - ref) / tn.linalg.norm(ref)
    assert y.is_cuda()
    assert rel_err.item() < 1e-3


@pytest.mark.parametrize("method", ["dmrg", "amen"])
def test_function_interpolate_qtt_modes(method):
    """
    Test cross interpolation with mode sizes [2]*d.
    """
    d = 8
    N = [2] * d

    xs = tntt.meshgrid([tn.linspace(0, 1, n, dtype=tn.float64) for n in N])
    func = lambda v: 1 / (1 + tn.sum(v, 1))
    x_ref = sum(x.full() for x in xs)
    x_ref = 1 / (1 + x_ref)

    y = tntt.interpolate.function_interpolate(
        func, xs, eps=1e-6, nswp=30, method=method
    )
    assert err_rel(y.full(), x_ref) < 1e-5


@pytest.mark.parametrize("method", ["dmrg", "amen"])
def test_function_interpolate_qtt_modes_fewer_args(method):
    """
    Test cross interpolation with mode sizes [2]*d where the number
    of function arguments is less than the number of tensor dimensions.
    """
    d = 10
    N = [2] * d

    xs = tntt.meshgrid([tn.linspace(0, 1, n, dtype=tn.float64) for n in N])
    func = lambda v: 1 / (1 + v[:, 0] + v[:, 1])
    x_ref = 1 / (1 + xs[0].full() + xs[1].full())

    y = tntt.interpolate.function_interpolate(
        func, [xs[0], xs[1]], eps=1e-6, nswp=30, method=method
    )
    assert err_rel(y.full(), x_ref) < 1e-5


@pytest.mark.parametrize("method", ["dmrg", "amen"])
def test_function_interpolate_multivariable_fewer_args(method):
    """
    Test multivariate function interpolation where the number of function
    arguments (3) is less than the number of tensor dimensions (4).
    This verifies that len(x) and d are handled independently.
    """
    N = [10] * 4

    Is = tntt.meshgrid([tn.arange(0, n, dtype=tn.float64) for n in N])
    # Use only 3 of the 4 meshgrid tensors as function arguments
    xs = [Is[0], Is[1], Is[2]]

    func = lambda v: 1 / (3 + v[:, 0] + v[:, 1] + v[:, 2])
    x_ref = 1 / (3 + Is[0].full() + Is[1].full() + Is[2].full())

    y = tntt.interpolate.function_interpolate(func, xs, 1e-8, method=method)
    assert err_rel(y.full(), x_ref) < 1e-7


@pytest.mark.parametrize("method", ["dmrg", "amen"])
def test_function_interpolate_multivariable_more_args(method):
    """
    Test multivariate function interpolation where the number of function
    arguments (5) is greater than the number of tensor dimensions (3).
    This verifies that len(x) and d are handled independently.
    """
    N = [10] * 3

    Is = tntt.meshgrid([tn.arange(0, n, dtype=tn.float64) for n in N])
    # Build 5 input tensors on a 3D grid: the originals plus two derived ones
    xs = [Is[0], Is[1], Is[2], Is[0] + Is[1], Is[1] + Is[2]]

    func = lambda v: 1 / (5 + v[:, 0] + v[:, 1] + v[:, 2] + v[:, 3] + v[:, 4])
    x_ref = 1 / (5 + Is[0].full() + Is[1].full() + Is[2].full()
                 + (Is[0] + Is[1]).full() + (Is[1] + Is[2]).full())

    y = tntt.interpolate.function_interpolate(func, xs, 1e-8, method=method)
    assert err_rel(y.full(), x_ref) < 1e-7


@pytest.mark.parametrize("method", ["dmrg", "amen"])
def test_function_interpolate_univariate(method):
    """
    Test the cross interpolation method for function approximation.
    """
    N = [20] * 4
    Is = tntt.meshgrid([tn.arange(0, n, dtype=tn.float64) for n in N])
    x_ref = 1 / (2 + Is[0].full() + Is[1].full() + Is[2].full() + Is[3].full() + 4)
    x = tntt.TT(x_ref)

    y = tntt.interpolate.function_interpolate(lambda x: tn.log(x), x, eps=1e-7, method=method)

    assert err_rel(y.full(), tn.log(x_ref)) < 1e-6


@pytest.mark.parametrize("method", ["dmrg", "amen"])
def test_function_interpolate_zero_multivariable(method):
    """
    Cross interpolation should handle exactly zero functions for both engines.
    """
    N = [5, 4, 3]
    Is = tntt.meshgrid([tn.arange(0, n, dtype=tn.float64) for n in N])
    func = lambda v: tn.zeros(v.shape[0], dtype=tn.float64)

    y = tntt.interpolate.function_interpolate(func, Is, eps=1e-8, nswp=5, method=method)

    assert y.N == N
    assert y.norm() < 1e-12


@pytest.mark.parametrize("method", ["dmrg", "amen"])
def test_function_interpolate_constant_multivariable(method):
    """
    Constant functions are rank-one edge cases for cross interpolation.
    """
    N = [5, 4, 3]
    Is = tntt.meshgrid([tn.arange(0, n, dtype=tn.float64) for n in N])
    func = lambda v: tn.full((v.shape[0],), 2.5, dtype=tn.float64)
    ref = 2.5 * tn.ones(N, dtype=tn.float64)

    y = tntt.interpolate.function_interpolate(func, Is, eps=1e-8, nswp=5, method=method)

    assert err_rel(y.full(), ref) < 1e-10