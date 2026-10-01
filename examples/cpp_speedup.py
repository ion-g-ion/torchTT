r"""
# Speedup of the C++ extension

The package `torchtt` has an optional C++ extension that implements two algorithms:

- the AMEn solver for linear systems `torchtt.solvers.amen_solve()` ([Dolgov and Savostyanov, 2014](https://doi.org/10.1137/140953289)),
- the DMRG matrix-vector product `torchtt.matvec()` with `method=torchtt.methods.DMRG()` ([Oseledets, 2011](https://doi.org/10.2478/cmam-2011-0021)).

Both algorithms are also implemented in Python.
The Python implementation is used when the extension is not available or when `use_cpp=False` is passed.
The C++ code calls the same kind of PyTorch functions (tensor contractions, QR and singular value decompositions, linear solves) as the Python code, without the Python overhead between these calls.
The C++ AMEn solver also needs fewer of these calls: its local GMRES solver orthogonalizes a new vector against all previous ones with two matrix-vector products instead of one dot product per previous vector, and its rank truncation checks several ranks at once.
This notebook compares the runtimes of the two implementations for a few problem sizes.
"""

#%% Imports
import time
import torch as tn
import torchtt as tntt

#%% Is the C++ extension available?
# The extension is compiled when the package is installed on Linux or macOS.
# If it is not available, `use_cpp=True` falls back to Python and both implementations below have the same runtime.
print('C++ extension available:', tntt.cpp_enabled())

#%% Timing
# The function `timing()` calls `f()` once to warm up and returns the average runtime of `n_runs` further calls together with the result of the last call.
def timing(f, n_runs=5):
    f()
    tme = time.perf_counter()
    for _ in range(n_runs):
        result = f()
    tme = (time.perf_counter() - tme) / n_runs
    return tme, result

#%% AMEn solver
# We solve $-\Delta u = 1$ in $[0,1]^d$ with $u = 0$ on the boundary using finite differences with $n$ interior points in every direction.
# The finite difference matrix is a sum of Kronecker products of the one-dimensional matrix with identity matrices and is constructed directly in the TT format:
def laplacian(d, n):
    """
    Finite difference matrix of the negative Laplacian in d dimensions with n interior points in every direction.
    """
    h = 1 / (n + 1)
    L1d = (2 * tn.eye(n, dtype=tn.float64) - tn.diag(tn.ones(n - 1, dtype=tn.float64), 1) - tn.diag(tn.ones(n - 1, dtype=tn.float64), -1)) / h**2
    L1d = tntt.TT(L1d, [(n, n)])
    L = L1d ** tntt.eye([n] * (d - 1)) + tntt.eye([n] * (d - 1)) ** L1d
    for i in range(1, d - 1):
        L = L + tntt.eye([n] * i) ** L1d ** tntt.eye([n] * (d - 1 - i))
    return L.round(1e-14)

# The right-hand side is a tensor of ones.
# The system is solved with both implementations and the same arguments for a few values of $d$ and $n$.
# The relative residuals show that both implementations reach the same accuracy.
print('  d    n   C++ [s]   Python [s]   speedup   residual C++   residual Python')
for d, n in [(4, 8), (16, 8), (8, 32), (4, 128)]:
    A = laplacian(d, n)
    b = tntt.ones([n] * d)
    t_cpp, x_cpp = timing(lambda: tntt.solvers.amen_solve(A, b, eps=1e-8, use_cpp=True))
    t_py, x_py = timing(lambda: tntt.solvers.amen_solve(A, b, eps=1e-8, use_cpp=False))
    res_cpp = ((A @ x_cpp - b).norm() / b.norm()).item()
    res_py = ((A @ x_py - b).norm() / b.norm()).item()
    print('%3d %4d %9.3f %12.3f %9.1f %14.1e %17.1e' % (d, n, t_cpp, t_py, t_py / t_cpp, res_cpp, res_py))

#%% DMRG matrix-vector product
# A random TT matrix of rank 3 is multiplied with random TT tensors of increasing rank using the DMRG method.
# Both implementations run the same algorithm, so the comparison holds whether or not DMRG is the fastest method for this product (the example on AMEn and DMRG for fast TT operations compares the methods).
# The errors are relative to the exact product `A @ x`.
d, n = 16, 4
A = tntt.random([(n, n)] * d, [1] + [3] * (d - 1) + [1])
print('rank of x   C++ [s]   Python [s]   speedup   error C++   error Python')
for r in [2, 8, 32]:
    x = tntt.random([n] * d, [1] + [r] * (d - 1) + [1])
    y_ref = A @ x
    t_cpp, y_cpp = timing(lambda: tntt.matvec(A, x, eps=1e-10, method=tntt.methods.DMRG(use_cpp=True)))
    t_py, y_py = timing(lambda: tntt.matvec(A, x, eps=1e-10, method=tntt.methods.DMRG(use_cpp=False)))
    err_cpp = ((y_cpp - y_ref).norm() / y_ref.norm()).item()
    err_py = ((y_py - y_ref).norm() / y_ref.norm()).item()
    print('%9d %9.3f %12.3f %9.1f %11.1e %14.1e' % (r, t_cpp, t_py, t_py / t_cpp, err_cpp, err_py))

#%% Conclusion
# The speedup is largest when the TT cores are small (small mode sizes and ranks): every PyTorch call is then cheap and the Python overhead between the calls dominates the runtime.
# For the largest cores ($n = 128$) more of the time is spent inside PyTorch and the speedup is smaller.
# The DMRG matrix-vector product gains less than the AMEn solver: singular value decompositions dominate its runtime, and the C++ AMEn solver also needs fewer PyTorch calls.
# The timings depend on the hardware and on the number of threads used by PyTorch.
# To check the speedup for your own problem, compare the runtimes with `use_cpp=True` and `use_cpp=False`, an argument of `torchtt.solvers.amen_solve()` and of `torchtt.methods.DMRG()`.
