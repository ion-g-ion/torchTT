
"""
Test the utility functions.
"""
import pytest
import torchtt as tntt
import torch as tn
import numpy as np

def err_rel(t, ref): 
    return (tn.linalg.norm(t-ref).numpy() / tn.linalg.norm(ref).numpy()
            if tn.linalg.norm(ref).numpy() > 0 else tn.linalg.norm(t-ref).numpy()) if ref.shape == t.shape else np.inf

basic_dtype = tn.complex128

def test_set_core():
    '''
    Test the changing of the core. 
    '''
    N = [10, 8, 6, 9, 12]
    x = tntt.random(N, [1, 3, 4, 5, 6, 1], dtype=basic_dtype)
    x.set_core(3, tn.rand((5, 11, 6)))

    assert list(x.N) == [10, 8, 6, 11, 12], "Set core error: TT case"

    A = tntt.random([(5, 6), (7, 8), (4, 5)], [1, 5, 3, 1], dtype=basic_dtype)
    A.set_core(1, tn.rand((5, 6, 4, 3)))

    assert list(A.N) == [6, 4, 5], "Set core error: TTM case"
    assert list(A.M) == [5, 6, 4], "Set core error: TTM case"


@pytest.mark.parametrize("shape", [[2, 3], [(2, 3), (3, 2)]])
def test_clone_detach(shape):
    x = tntt.randn(shape, [1, 2, 1], dtype=tn.float64)
    for core in x.cores:
        core.requires_grad_()

    y = x.clone()
    z = x.detach()

    assert tn.equal(y.full(), x.full())
    assert tn.equal(z.full(), x.full())
    assert all(c.requires_grad for c in y.cores)
    assert all(not c.requires_grad for c in z.cores)
    with tn.no_grad():
        y.cores[0].zero_()
    assert x.norm() > 0
    assert y.norm() == 0


@pytest.mark.parametrize("shape", [[2, 3], [(2, 3), (3, 2)]])
def test_to_dtype(shape):
    x = tntt.randn(shape, [1, 2, 1], dtype=tn.float64)
    y = x.to(device='cpu', dtype=tn.complex128)

    assert y.shape == x.shape
    assert all(c.dtype == tn.complex128 for c in y.cores)
    assert all(c.dtype == tn.float64 for c in x.cores)
    assert err_rel(y.full(), x.full().to(tn.complex128)) < 1e-14


@pytest.mark.parametrize("ttm", [False, True])
@pytest.mark.parametrize("exclude", [[], [0, 2]])
def test_reduce_dims(ttm, exclude):
    N = [1, 3, 1, 4, 1]
    shape = [(n, n) for n in N] if ttm else N
    x = tntt.randn(shape, [1, 2, 3, 2, 2, 1], dtype=basic_dtype)
    ref = x.full()
    remaining = [n for i, n in enumerate(N) if n != 1 or i in exclude]

    x.reduce_dims(exclude=exclude)

    assert x.N == remaining
    assert x.shape == ([(n, n) for n in remaining] if ttm else remaining)
    assert err_rel(x.full(), ref.reshape(remaining * (2 if ttm else 1))) < 1e-13


@pytest.mark.parametrize("index", [-1, 2])
def test_set_core_invalid_index(index):
    x = tntt.ones([2, 3])
    with pytest.raises(tntt.errors.InvalidArguments):
        x.set_core(index, tn.ones(1, 2, 1))


@pytest.mark.parametrize("shape", [[2, 3], [(2, 3), (3, 2)]])
def test_set_core_invalid_rank(shape):
    x = tntt.ones(shape)
    core = tn.ones(2, *x.cores[0].shape[1:])
    with pytest.raises(tntt.errors.InvalidArguments):
        x.set_core(0, core)


@pytest.mark.parametrize("shape", [[3, 4, 5], [(3, 2), (4, 3), (2, 5)]])
@pytest.mark.parametrize("dtype", [tn.float64, tn.complex128])
def test_save_load(shape, dtype, tmp_path):
    x = tntt.randn(shape, [1, 2, 3, 1], dtype=dtype)
    path = tmp_path / 'tensor.tt'

    tntt.save(x, path)
    y = tntt.load(path)

    assert y.shape == x.shape
    assert y.R == x.R
    assert all(c.dtype == dtype for c in y.cores)
    assert all(tn.equal(c, r) for c, r in zip(y.cores, x.cores))
    assert tn.equal(y.full(), x.full())


def test_save_invalid_input(tmp_path):
    path = tmp_path / 'tensor.tt'
    with pytest.raises(tntt.errors.InvalidArguments):
        tntt.save(tn.ones(3), path)
    assert not path.exists()


@pytest.mark.parametrize("ttm", [False, True])
def test_rank1tt(ttm):
    a = tn.arange(1, 7, dtype=tn.float64)
    b = tn.arange(1, 13, dtype=tn.float64)
    if ttm:
        a, b = a.reshape(2, 3), b.reshape(3, 4)
    x = tntt.rank1TT([a, b])
    ref = tn.kron(a, b)

    assert x.R == [1, 1, 1]
    assert tntt.numel(x) == 18
    assert tn.equal(x.full().reshape(ref.shape), ref)


@pytest.mark.parametrize("M, N, shape", [([], [], []), ([3], [2], [(3, 2)]),
    ([3, 4, 5], [2, 5, 4], [(3, 2), (4, 5), (5, 4)])])
def test_shape_conversion(M, N, shape):
    assert tntt.shape_mn_to_tuple(M, N) == shape
    assert tntt.shape_tuple_to_mn(shape) == (M, N)


def test_cpp_available():
    assert isinstance(tntt.cpp_enabled(), bool)
    assert tntt.cpp.cpp_avaible() == tntt.cpp_enabled() == tntt.solvers.cpp_enabled()
