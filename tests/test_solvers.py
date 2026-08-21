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
