"""
Test the multilinear solvers.
"""
import torchtt 
import torch as tn
import numpy as np
import pytest

err_rel = lambda t, ref :  tn.linalg.norm(t-ref).numpy() / tn.linalg.norm(ref).numpy() if ref.shape == t.shape else np.inf

basic_dtype = tn.complex128


def random_system(seed, shift=0.2, dtype=tn.float64):
    """
    Build a random TT linear system A @ x = b with a modest condition number.

    A bare torchtt.random TT-matrix has cond(A) ~ 1e3-1e5, and the residual
    attainable by amen_solve scales with eps*cond(A) rather than with eps.
    That made these tests a lottery on the draw. Shifting towards the identity
    keeps the system random while bounding the conditioning (median ~5.5,
    max ~15 over 200 draws). The seed keeps CI runs reproducible: both the
    system and the enrichment tensor drawn inside amen_solve come from the
    global torch generator.
    """
    tn.manual_seed(seed)
    A = torchtt.random([(4, 4), (5, 5), (6, 6)], [1, 2, 3, 1], dtype=dtype)
    A = (A * (1 / A.norm()) + shift * torchtt.eye([4, 5, 6], dtype=dtype)).round(1e-14)
    x = torchtt.random([4, 5, 6], [1, 2, 3, 1], dtype=dtype)
    return A, x, A @ x


@pytest.mark.skipif(not torchtt.solvers.cpp_enabled(), reason="C++ extension must be present.")
def test_amen_solve():
    A, x, b = random_system(1)
    xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner=None, use_cpp=True) 
    err = (A@xx-b).norm()/b.norm() # error residual
    assert err.numpy() < 5*1e-8, "AMEN solve failed."

@pytest.mark.skipif(not torchtt.solvers.cpp_enabled(), reason="C++ extension must be present.")
def test_amen_solve_cprec():
    A, x, b = random_system(2)
    xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner='c', use_cpp=True) 
    err = (A@xx-b).norm()/b.norm() # error residual
    assert err.numpy() < 5*1e-8, "AMEN solve failed (c preconditioner)."

@pytest.mark.skipif(not torchtt.solvers.cpp_enabled(), reason="C++ extension must be present.")
def test_amen_solve_rprec():
    A, x, b = random_system(3)
    xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner='r', use_cpp=True) 
    err = (A@xx-b).norm()/b.norm() # error residual
    assert err.numpy() < 5*1e-8, "AMEN solve failed (right preconditioner)."

def test_amen_solve_cprec_nocpp():
    A, x, b = random_system(4)
    xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner='c', use_cpp=False) 
    err = (A@xx-b).norm()/b.norm() # error residual
    assert err.numpy() < 5*1e-8, "AMEN solve failed (c preconditioner, without C++)."

def test_amen_solve_rprec_nocpp():
    A, x, b = random_system(5)
    xx = torchtt.solvers.amen_solve(A, b, verbose = False, eps=1e-10, nswp = 40, preconditioner='r', use_cpp=False) 
    err = (A@xx-b).norm()/b.norm() # error residual
    assert err.numpy() < 5*1e-8, "AMEN solve failed (right preconditioner)."

def test_amen_solve_nocpp():
    A, x, b = random_system(6)
    xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner=None, use_cpp = False) 
    err = (A@xx-b).norm()/b.norm() # error residual
    assert err.numpy() < 5*1e-8, "AMEN solve failed."
    xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner='c', use_cpp = False) 
    err = (A@xx-b).norm()/b.norm() # error residual
    assert err.numpy() < 5*1e-8, "AMEN solve failed (c preconditioner)."

def test_amen_solve_final_sweep_preserves_convergence():
    """
    Regression test: the final AMEn sweep must not discard a converged solution.

    The last sweep runs with the residual enrichment disabled, so accuracy given
    away by the per-core truncation there can no longer be recovered. With the
    previous truncation budget this draw reached a global residual of 2.4e-12 and
    was then degraded to 8.8e-8 by that final sweep -- which is what made
    test_amen_solve_rprec_nocpp fail intermittently in CI.

    This uses a deliberately ill-conditioned matrix (cond(A) ~ 3e4), since the
    conditioning is what makes the degradation cross a visible threshold.
    """
    tn.manual_seed(5173)
    A = torchtt.random([(4, 4), (5, 5), (6, 6)], [1, 2, 3, 1], dtype=tn.float64)
    x = torchtt.random([4, 5, 6], [1, 2, 3, 1], dtype=tn.float64)
    b = A @ x
    xx = torchtt.solvers.amen_solve(A, b, verbose=False, eps=1e-10, nswp=40, preconditioner='r', use_cpp=False)
    err = (A@xx-b).norm()/b.norm()
    assert err.numpy() < 1e-10, "final sweep degraded an already converged solution."


@pytest.mark.parametrize("local_solver", [1, 2])
@pytest.mark.parametrize("preconditioner", [None, 'c', 'r'])
@pytest.mark.parametrize("use_single_precision", [False, True])
def test_amen_solve_iterative(local_solver, preconditioner, use_single_precision):
    A, x, b = random_system(7, shift=1.0)
    eps = 1e-5 if use_single_precision else 1e-10
    # Force iterative local solves, including on the boundary cores.
    xx = torchtt.solvers.amen_solve(
        A, b, eps=eps, max_full=0, local_solver=local_solver,
        preconditioner=preconditioner, use_single_precision=use_single_precision,
        local_iterations=20, resets=3, use_cpp=False)

    assert (A@xx-b).norm()/b.norm() < eps
    assert err_rel(xx.full(), x.full()) < eps


@pytest.mark.parametrize("kickrank, kick2, rmax", [(1, 0, 4), (2, 1, [1, 4, 4, 1])])
@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_amen_solve_initial_guess(kickrank, kick2, rmax, scale):
    A, x, b = random_system(8, shift=1.0)
    x0 = scale*x
    ref = x0.full().clone()

    xx = torchtt.solvers.amen_solve(
        A, b, x0=x0, eps=1e-10, kickrank=kickrank, kick2=kick2, rmax=rmax, use_cpp=False)

    assert err_rel(xx.full(), x.full()) < 1e-9
    assert all(r <= limit for r, limit in zip(xx.R, [1, 4, 4, 1]))
    assert tn.equal(x0.full(), ref), "AMEN solve changed the initial guess."


@pytest.mark.parametrize("band_diagonal", [0, 1])
@pytest.mark.parametrize("max_full", [0, 256])
def test_amen_solve_banded(band_diagonal, max_full):
    tn.manual_seed(9)
    N = [3, 4, 2]
    cores = []
    for n in N:
        core = tn.diag(tn.arange(1, n+1, dtype=tn.float64))
        if band_diagonal == 1:
            core += tn.diag(tn.full((n-1,), 0.1, dtype=tn.float64), 1)
            core += tn.diag(tn.full((n-1,), -0.2, dtype=tn.float64), -1)
        cores.append(core.reshape(1, n, n, 1))
    A = torchtt.TT(cores)
    b = torchtt.randn(N, [1, 2, 2, 1], dtype=tn.float64)
    ref = tn.linalg.solve(A.full().reshape(24, 24), b.full().flatten()).reshape(N)

    xx = torchtt.solvers.amen_solve(
        A, b, eps=1e-10, band_diagonal=band_diagonal, max_full=max_full, use_cpp=False)

    assert err_rel(xx.full(), ref) < 1e-9
    assert (A@xx-b).norm()/b.norm() < 1e-9


@pytest.mark.parametrize("N", [[1, 4, 1], [2, 1, 3]])
def test_amen_solve_singleton_modes(N):
    tn.manual_seed(10)
    A = torchtt.eye(N, dtype=tn.float64)
    b = torchtt.randn(N, [1, 2, 2, 1], dtype=tn.float64)

    x = torchtt.solvers.amen_solve(A, b, eps=1e-10, use_cpp=False)

    assert x.N == N
    assert err_rel(x.full(), b.full()) < 1e-9


def test_amen_solve_single_mode():
    A = torchtt.TT(tn.diag(tn.tensor([1., 2., 3.], dtype=tn.float64)), [(3, 3)])
    b = torchtt.TT(tn.tensor([2., 4., 6.], dtype=tn.float64))

    x = torchtt.solvers.amen_solve(A, b, use_cpp=False)

    assert tn.allclose(x.full(), tn.full((3,), 2., dtype=tn.float64))


@pytest.mark.parametrize("zero_core", [None, 0, 1, 2])
@pytest.mark.parametrize("max_full", [0, 256])
@pytest.mark.parametrize("use_cpp", [False, pytest.param(True, marks=
    pytest.mark.skipif(not torchtt.solvers.cpp_enabled(), reason="C++ extension must be present."))])
def test_amen_solve_zero_rhs(use_cpp, max_full, zero_core):
    A = torchtt.eye([2, 3, 2], dtype=tn.float64)
    b = torchtt.zeros([2, 3, 2], dtype=tn.float64) if zero_core is None else torchtt.ones([2, 3, 2], dtype=tn.float64)
    if zero_core is not None:
        b.cores[zero_core].zero_()

    x = torchtt.solvers.amen_solve(A, b, nswp=4, max_full=max_full, use_cpp=use_cpp)

    assert x.N == b.N
    assert x.norm() < 1e-12


def test_amen_solve_invalid_operands():
    A = torchtt.eye([2, 3])
    b = torchtt.ones([2, 3])

    with pytest.raises(torchtt.errors.InvalidArguments):
        torchtt.solvers.amen_solve(A.full(), b)
    with pytest.raises(torchtt.errors.InvalidArguments):
        torchtt.solvers.amen_solve(A, b.full())
    with pytest.raises(torchtt.errors.IncompatibleTypes):
        torchtt.solvers.amen_solve(b, b)
    with pytest.raises(torchtt.errors.IncompatibleTypes):
        torchtt.solvers.amen_solve(A, A)
    with pytest.raises(torchtt.errors.ShapeMismatch):
        torchtt.solvers.amen_solve(torchtt.ones([(2, 3), (3, 2)]), b)
    with pytest.raises(torchtt.errors.ShapeMismatch):
        torchtt.solvers.amen_solve(A, torchtt.ones([2, 4]))


@pytest.mark.parametrize("use_single_precision", [False, True])
def test_amen_solve_invalid_local_solver(use_single_precision):
    A, x, b = random_system(11)
    with pytest.raises(torchtt.errors.InvalidArguments, match="Solver not implemented"):
        torchtt.solvers.amen_solve(
            A, b, max_full=0, local_solver=0, use_single_precision=use_single_precision, use_cpp=False)


@pytest.mark.skipif(not torchtt.solvers.cpp_enabled(), reason="C++ extension must be present.")
@pytest.mark.parametrize("preconditioner", [None, 'c', 'r'])
def test_amen_solve_iterative_cpp(preconditioner):
    A, x, b = random_system(7, shift=1.0)
    x0 = 0.5*x
    ref = x0.full().clone()

    xx = torchtt.solvers.amen_solve(
        A, b, x0=x0, eps=1e-10, max_full=0, kickrank=2, kick2=1,
        preconditioner=preconditioner, use_cpp=True)

    assert (A@xx-b).norm()/b.norm() < 1e-9
    assert err_rel(xx.full(), x.full()) < 1e-9
    assert tn.equal(x0.full(), ref), "AMEN solve changed the initial guess."


@pytest.mark.skipif(not torchtt.solvers.cpp_enabled(), reason="C++ extension must be present.")
def test_amen_solve_invalid_preconditioner_cpp():
    with pytest.raises(torchtt.errors.InvalidArguments, match="Invalid preconditioner"):
        torchtt.solvers.amen_solve(torchtt.eye([2, 3]), torchtt.ones([2, 3]), preconditioner='invalid', use_cpp=True)


@pytest.mark.skipif(not torchtt.solvers.cpp_enabled(), reason="C++ extension must be present.")
@pytest.mark.parametrize("preconditioner", [None, 'c', 'r'])
def test_amen_solve_single_precision_cpp(preconditioner):
    # Only GMRES runs in single precision. The residuals stay in float64, so the
    # solution reaches a tolerance below the float32 accuracy.
    A, x, b = random_system(7, shift=1.0)

    xx = torchtt.solvers.amen_solve(
        A, b, eps=1e-10, max_full=0, preconditioner=preconditioner,
        use_single_precision=True, use_cpp=True)

    assert all(c.dtype == tn.float64 for c in xx.cores)
    assert (A@xx-b).norm()/b.norm() < 1e-9
    assert err_rel(xx.full(), x.full()) < 1e-9


@pytest.mark.skipif(not torchtt.solvers.cpp_enabled(), reason="C++ extension must be present.")
def test_amen_solve_complex_cpp():
    A, x, b = random_system(3, dtype=tn.complex128)
    with pytest.raises(RuntimeError, match="complex"):
        torchtt.solvers.amen_solve(A, b, use_cpp=True)


@pytest.mark.skipif(not torchtt.solvers.cpp_enabled(), reason="C++ extension must be present.")
@pytest.mark.skipif(not tn.cuda.is_available(), reason="CUDA device is not available.")
@pytest.mark.parametrize("max_full", [0, 256])
def test_amen_solve_cuda_cpp(max_full):
    A, x, b = random_system(5, shift=1.0)
    A, x, b = A.to('cuda'), x.to('cuda'), b.to('cuda')

    xx = torchtt.solvers.amen_solve(A, b, eps=1e-10, max_full=max_full, use_cpp=True)

    assert all(c.is_cuda for c in xx.cores)
    assert (A@xx-b).norm()/b.norm() < 1e-9
    assert (xx-x).norm()/x.norm() < 1e-9


@pytest.mark.parametrize("local_solver", [1, 2])
@pytest.mark.parametrize("use_single_precision", [False, True])
def test_amen_solve_zero_rhs_iterative(local_solver, use_single_precision):
    A = torchtt.eye([2, 3, 2], dtype=tn.float64)
    b = torchtt.zeros([2, 3, 2], dtype=tn.float64)

    x = torchtt.solvers.amen_solve(
        A, b, max_full=0, local_solver=local_solver,
        use_single_precision=use_single_precision, use_cpp=False)

    assert x.norm() < 1e-12
