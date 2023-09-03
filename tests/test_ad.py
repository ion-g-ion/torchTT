"""
Test all the AD related functions.

@author: ion
"""
import torch as tn
import torchtt
import pytest


def err_rel(t, ref): return tn.linalg.norm(t-ref).numpy() / \
    tn.linalg.norm(ref).numpy() if ref.shape == t.shape else np.inf


def test_manifold():
    """
    Compare the result of the manifold projection and the manifold gradient computed using AD.
    """

    target = torchtt.randn([10, 12, 14, 16], [1, 8, 8, 7, 1])

    def func(x): return 0.5*(x-target).norm(True)

    R = [1, 3, 4, 6, 1]
    x = torchtt.randn(target.N, R.copy())

    gr_ad = torchtt.manifold.riemannian_gradient(x, func)

    gr_proj = torchtt.manifold.riemannian_projection(x, (x-target))

    assert gr_ad.R == [
        2*r if r != 1 else 1 for r in R], "TT manifold: Riemannian gradient error: ranks mismatch."
    assert gr_proj.R == [
        2*r if r != 1 else 1 for r in R], "TT manifold: Riemannian projection error: ranks mismatch."
    assert err_rel(gr_ad.full(), gr_proj.full(
    )) < 1e-12, "TT manifold: Riemannian gradient and projected gradient differ."


def test_mainfold_matrix():
    """
    Test the manifold gradient and the manifold projection for the TT matrix case.
    """

    A = torchtt.randn([(2, 3), (4, 5), (6, 7), (4, 2)], [1, 2, 3, 2, 1])
    X = torchtt.randn([(2, 3), (4, 5), (6, 7), (4, 2)], [1, 3, 2, 2, 1])

    def func(x): return 0.5*(x-A).norm(True)

    gr_ad = torchtt.manifold.riemannian_gradient(X, func)

    gr_proj = torchtt.manifold.riemannian_projection(X, (X-A))

    assert gr_ad.R == [
        2*r if r != 1 else 1 for r in X.R], "TT manifold: Riemannian gradient error: ranks mismatch."
    assert gr_proj.R == [
        2*r if r != 1 else 1 for r in X.R], "TT manifold: Riemannian projection error: ranks mismatch."
    assert err_rel(gr_ad.full(), gr_proj.full(
    )) < 1e-12, "TT manifold: Riemannian gradient and projected gradient differ."


def test_ad():
    """
    Test the AD functionality.
    """
    N = [2, 3, 4, 5]
    A = torchtt.randn([(n, n) for n in N], [1]+[2]*(len(N)-1)+[1])
    y = torchtt.randn(N, A.R)
    x = torchtt.ones(N)

    def f(x, A, y):
        z = torchtt.dot(A @ (x-y), (x-y))
        return z.norm()

    torchtt.grad.watch(x)

    val = f(x, A, y)
    grad_cores = torchtt.grad.grad(val, x)

    torchtt.grad.watch(A)

    val = f(x, A, y)
    grad_cores_A = torchtt.grad.grad(val, A)

    assert [c.shape for c in grad_cores] == [
        c.shape for c in x.cores], "TT AD: problem for grad w.r.t. TT tensor."
    assert [c.shape for c in grad_cores_A] == [
        c.shape for c in A.cores], "TT AD: problem for grad w.r.t. TT matrix."


if __name__ == '__main__':
    pytest.main()
