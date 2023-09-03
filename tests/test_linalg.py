"""
Test the basic multilinear algebra operations between torchtt.TT objects.
"""
import pytest
import torchtt as tntt
import torch as tn
import numpy as np


def err_rel(t, ref): return (tn.linalg.norm(t-ref).numpy() / tn.linalg.norm(ref).numpy()
                             if tn.linalg.norm(ref).numpy() > 0 else tn.linalg.norm(t-ref).numpy()) if ref.shape == t.shape else np.inf


parameters = [tn.float64, tn.complex128]

@pytest.mark.parametrize("dtype", parameters)
def test_add(dtype):
    '''
    Test the addition operator 
    '''
    N = [10, 8, 6, 9, 12]

    x = tntt.random(N, [1, 3, 4, 5, 6, 1], dtype=dtype)
    y = tntt.random(N, [1, 2, 4, 5, 4, 1], dtype=dtype)
    z = tntt.random(N, [1, 2, 2, 2, 2, 1], dtype=dtype)
    const = 3.1415926535

    X = x.full()
    Y = y.full()
    Z = z.full()

    w = x+y+z
    t = const+(const+x)+const

    W = X+Y+Z
    T = const+(const+X)+const

    assert err_rel(w.full(), W) < 1e-14, 'Addition error 1'
    assert err_rel(t.full(), T) < 1e-14, 'Addition error 2'

    M = tntt.random([(5, 6), (7, 8), (9, 10)], [1, 5, 5, 1])
    P = tntt.random([(5, 6), (7, 8), (9, 10)], [1, 2, 20, 1])

    Q = M+P+P+M
    Qr = M.full()+P.full()+P.full()+M.full()

    assert err_rel(Q.full(), Qr) < 1e-14, 'Addition error 2: TT-matrix'

    # test broadcasting

    x = tntt.random([2, 3, 4, 5, 6], [1, 2, 4, 8, 4, 1],
                    dtype=dtype)
    y = tntt.random([4, 5, 6], [1, 2, 2, 1], dtype=dtype)
    xr = x.full()
    yr = y.full()

    z = x+y
    zr = xr+yr
    assert err_rel(z.full(), zr) < 1e-13, "Addition broadcasting error 1: TT-tensors."

    x = tntt.random([2, 3, 4, 5, 6], [1, 2, 4, 8, 4, 1],
                    dtype=dtype)
    y = tntt.random([1, 1, 6], [1, 2, 2, 1], dtype=dtype)
    xr = x.full()
    yr = y.full()

    z = x+y
    zr = xr+yr
    assert err_rel(z.full(), zr) < 1e-13, "Addition broadcasting error 2: TT-tensors."

    x = tntt.random([2, 3, 4, 5, 6], [1, 2, 4, 8, 4, 1],
                    dtype=dtype)
    y = tntt.random([1], [1, 1], dtype=dtype)
    xr = x.full()
    yr = y.full()

    z = x+y
    zr = xr+yr
    assert err_rel(z.full(), zr) < 1e-13, "Addition broadcasting error 3: TT-tensors."

    x = tntt.random([2, 3, 4, 5, 6], [1, 2, 4, 8, 4, 1],
                    dtype=dtype)
    y = tntt.random([1, 1, 4, 5, 6], [1, 2, 4, 8, 4, 1],
                    dtype=dtype)
    xr = x.full()
    yr = y.full()

    z = x+y
    zr = xr+yr
    assert err_rel(z.full(), zr) < 1e-13, "Addition broadcasting error 4: TT-tensors."

@pytest.mark.parametrize("dtype", parameters)
def test_sub(dtype):
    '''
    Test the subtraction operator. 
    '''
    N = [10, 8, 6, 9, 12]

    x = tntt.random(N, [1, 3, 4, 5, 6, 1], dtype=dtype)
    y = tntt.random(N, [1, 2, 4, 5, 4, 1], dtype=dtype)
    z = tntt.random(N, [1, 2, 2, 2, 2, 1], dtype=dtype)
    const = 3.1415926535

    X = x.full()
    Y = y.full()
    Z = z.full()

    w = -x+y-z
    t = const+(const-x)-const

    W = -X+Y-Z
    T = const+(const-X)-const

    assert err_rel(w.full(), W) < 1e-14, 'Subtraction error 1'
    assert err_rel(t.full(), T) < 1e-14, 'Subtraction error 2'

    M = tntt.random([(5, 6), (7, 8), (9, 10)], [1, 5, 5, 1])
    P = tntt.random([(5, 6), (7, 8), (9, 10)], [1, 2, 20, 1])

    Q = -M+P-P-M
    Qr = -M.full()+P.full()-P.full()-M.full()

    assert err_rel(Q.full(), Qr) < 1e-14, 'Subtraction error 2: TT-matrix'

    # test broadcasting

    x = tntt.random([2, 3, 4, 5, 6], [1, 2, 4, 8, 4, 1],
                    dtype=dtype)
    y = tntt.random([4, 5, 6], [1, 2, 2, 1], dtype=dtype)
    xr = x.full()
    yr = y.full()

    z = x-y
    zr = xr-yr
    assert err_rel(z.full(), zr) < 1e-13, "Subtraction broadcasting error 1: TT-tensors."

    x = tntt.random([2, 3, 4, 5, 6], [1, 2, 4, 8, 4, 1],
                    dtype=dtype)
    y = tntt.random([1, 1, 6], [1, 2, 2, 1], dtype=dtype)
    xr = x.full()
    yr = y.full()

    z = x-y
    zr = xr-yr
    assert err_rel(z.full(), zr) < 1e-13, "Subtraction broadcasting error 2: TT-tensors."

    x = tntt.random([2, 3, 4, 5, 6], [1, 2, 4, 8, 4, 1],
                    dtype=dtype)
    y = tntt.random([1], [1, 1], dtype=dtype)
    xr = x.full()
    yr = y.full()

    z = x-y
    zr = xr-yr
    assert err_rel(z.full(), zr) < 1e-13, "Subtraction broadcasting error 3: TT-tensors."

    x = tntt.random([2, 3, 4, 5, 6], [1, 2, 4, 8, 4, 1],
                    dtype=dtype)
    y = tntt.random([1, 1, 4, 5, 6], [1, 2, 4, 8, 4, 1],
                    dtype=dtype)
    xr = x.full()
    yr = y.full()

    z = x-y
    zr = xr-yr
    assert err_rel(z.full(), zr) < 1e-13, "Subtraction broadcasting error 4: TT-tensors."

@pytest.mark.parametrize("dtype", parameters)
def test_mult(dtype):
    """
    Test the pointwise multiplication between TT-objects.
    """
    A = tntt.random([(5, 6), (7, 8), (9, 10), (4, 5)], [
                    1, 5, 5, 3, 1], dtype=dtype)
    B = tntt.random([(5, 6), (7, 8), (9, 10), (4, 5)], [
                    1, 5, 5, 3, 1], dtype=dtype)
    Ar = A.full()
    Br = B.full()
    x = tntt.random([2, 3, 4, 5, 6], [1, 2, 4, 8, 4, 1],
                    dtype=dtype)
    y = tntt.random([2, 3, 4, 5, 6], [1, 2, 5, 6, 2, 1],
                    dtype=dtype)
    xr = x.full()
    yr = y.full()
    c = 2.5

    z = c*x*(-y*c)
    zr = c*xr*(-yr*c)

    assert err_rel(z.full(), zr) < 1e-13, "Multiplication error: TT-tensors."

    C = c*A*(B*c)
    Cr = c*Ar*(Br*c)

    assert err_rel(C.full(), Cr) < 1e-13, "Multiplication error: TT-matrices."

    z = 0*x
    zr = 0*xr
    assert err_rel(z.full(), zr) < 1e-13, "Multiplication error: TT-tensor 0 with scalar."
    assert z.R == [1, 1, 1, 1, 1, 1], "Multiplication error: TT-tensor 0 with scalar."

    C = 0*A
    Cr = 0*Ar
    assert err_rel(C.full(), Cr) < 1e-13, "Multiplication error: TT-matrix 0 with scalar."
    assert C.R == [1, 1, 1, 1, 1], "Multiplication error: TT-matrix 0 with scalar."

    # test broadcasting

    x = tntt.random([2, 3, 4, 5, 6], [1, 2, 4, 8, 4, 1],
                    dtype=dtype)
    y = tntt.random([4, 5, 6], [1, 2, 2, 1], dtype=dtype)
    xr = x.full()
    yr = y.full()

    z = x*y
    zr = xr*yr
    assert err_rel(z.full(), zr) < 1e-13, "Multiplication broadcasting error 1: TT-tensors."

    x = tntt.random([2, 3, 4, 5, 6], [1, 2, 4, 8, 4, 1],
                    dtype=dtype)
    y = tntt.random([1, 1, 6], [1, 2, 2, 1], dtype=dtype)
    xr = x.full()
    yr = y.full()

    z = x*y
    zr = xr*yr
    assert err_rel(z.full(), zr) < 1e-13, "Multiplication broadcasting error 2: TT-tensors."

    x = tntt.random([2, 3, 4, 5, 6], [1, 2, 4, 8, 4, 1],
                    dtype=dtype)
    y = tntt.random([1], [1, 1], dtype=dtype)
    xr = x.full()
    yr = y.full()

    z = x*y
    zr = xr*yr
    assert err_rel(z.full(), zr) < 1e-13, "Multiplication broadcasting error 3: TT-tensors."

    x = tntt.random([2, 3, 4, 5, 6], [1, 2, 4, 8, 4, 1],
                    dtype=dtype)
    y = tntt.random([1, 1, 4, 5, 6], [1, 2, 4, 8, 4, 1],
                    dtype=dtype)
    xr = x.full()
    yr = y.full()

    z = x*y
    zr = xr*yr
    assert err_rel(z.full(), zr) < 1e-13, "Multiplication broadcasting error 4: TT-tensors."

@pytest.mark.parametrize("dtype", parameters)
def test_matmult(dtype):
    """
    Test the matrix multiplication operations.
    """


    A = tntt.random([(5, 6), (7, 8), (4, 5)], [1, 5, 3, 1], dtype=dtype)
    B = tntt.random([(5, 6), (7, 8), (4, 5)], [1, 5, 3, 1], dtype=dtype).t()
    x = tntt.randn([5, 7, 4], [1, 3, 2, 1], dtype=dtype)
    y = tntt.randn([6, 8, 5], [1, 1, 2, 1], dtype=dtype)

    A_ref = A.full()
    B_ref = B.full()
    x_ref = x.full()
    y_ref = y.full()

    # matrix matrix
    C = A @ B
    C_ref = tn.einsum('ijkabc,abcmno->ijkmno', A_ref, B_ref)
    assert err_rel(C.full(), C_ref) < 1e-13, "torchtt.TT.__matmul__() error: 2 TT matrices."

    # matrix vector
    z = B @ x
    z_ref = tn.einsum('abcijk,ijk->abc', B_ref, x_ref)
    assert err_rel(z.full(), z_ref) < 1e-13, "torchtt.TT.__matmul__() error: TT matrix with TT vector."

@pytest.mark.parametrize("dtype", parameters)
def test_matvecdense(dtype):
    """
    Test the multiplication between a TT-matrix and a dense tensor

    """

    A = tntt.random([(5, 6), (7, 8), (9, 10), (4, 5)], [
                    1, 5, 5, 3, 1], dtype=dtype)

    x = tn.rand([6, 8, 10, 5], dtype=dtype)
    y = A @ x
    yr = tn.einsum('abcdijkl,ijkl->abcd', A.full(), x)
    assert err_rel(y, yr) < 1e-14, 'Dense matvec error 1.'

    x = tn.rand([32, 4, 33, 6, 8, 10, 5], dtype=dtype)
    y = A @ x
    yr = tn.einsum('abcdijkl,mnoijkl->mnoabcd', A.full(), x)
    assert y.shape == yr.shape, 'Dense matvec shape mismatch.'
    assert err_rel(y, yr) < 1e-14, 'Dense matvec error 2.'

    x = tn.rand([1, 22, 6, 8, 10, 5], dtype=dtype)
    y = A @ x
    yr = tn.einsum('abcdijkl,nmijkl->nmabcd', A.full(), x)
    assert y.shape == yr.shape, 'Dense matvec shape mismatch.'
    assert err_rel(y, yr) < 1e-14, 'Dense matvec error 2.'
    
@pytest.mark.parametrize("dtype", parameters)
def test_mode_product(dtype):
    """
    Test the n-mode tensor product.
    """

    x = tntt.randn([2, 3, 4, 5, 6], [1, 3, 3, 3, 3, 1],
                   dtype=dtype)
    M1 = tn.rand((8, 3), dtype=dtype)
    M2 = tn.rand((7, 2), dtype=dtype)
    M3 = tn.rand((10, 5), dtype=dtype)

    y = x.mprod(M1, 1)
    yr = tn.einsum('ijklm,aj->iaklm', x.full(), M1)
    assert err_rel(y.full(), yr) < 1e-14, 'torchtt.tt.mprod() error: case 1.'

    z = x.mprod([M2, M3], [0, 3])
    zr = tn.einsum('ijklm,ai,bl->ajkbm', x.full(), M2, M3)
    assert err_rel(z.full(), zr) < 1e-14, 'torchtt.tt.mprod() error: case 2.'

@pytest.mark.parametrize("dtype", parameters)
def test_dot(dtype):
    '''
    Test the dot product between TT tensors.
    '''

    a = tntt.random([4, 5, 6, 7, 8, 9], [1, 2, 10, 16, 20, 7, 1], dtype=dtype)
    b = tntt.random([4, 5, 6, 7, 8, 9], [1, 3, 4, 10, 10, 4, 1], dtype=dtype)
    c = tntt.random([5, 7, 9], [1, 2, 7, 1], dtype=dtype)
    d = tntt.random([4, 5, 9], [1, 2, 2, 1], dtype=dtype)

    x = tntt.dot(a, b)
    y = tntt.dot(a, c, [1, 3, 5])
    z = tntt.dot(b, d, [0, 1, 5])

    assert err_rel(x, tn.einsum('abcdef,abcdef->', a.full(), tn.conj(b.full()))) < 1e-12, 'Dot product error. Test: equal sized tensors.'
    assert err_rel(y.full(), tn.einsum('abcdef,bdf->ace', a.full(), tn.conj(c.full()))) < 1e-12, 'Dot product error. Test: different sizes 1.'
    assert err_rel(z.full(), tn.einsum('abcdef,abf->cde', b.full(), tn.conj(d.full()))) < 1e-12, 'Dot product error. Test: different sizes 2.'

@pytest.mark.parametrize("dtype", parameters)
def test_sum(dtype):
    '''
    Test the sum method.
    '''
    a = tntt.random([4, 5, 6, 7, 8], [1, 2, 10, 16, 7, 1], dtype=dtype)

    afull = a.full()

    assert err_rel(a.sum(), afull.sum()) < 1e-13, "Test TT.sum() error 1."

@pytest.mark.parametrize("dtype", parameters)
def test_kron(dtype):
    """
    Test the Kronecker product.
    """
    a = tntt.random([5, 7, 9], [1, 2, 7, 1], dtype=dtype)
    b = tntt.random([4, 5, 9], [1, 2, 2, 1], dtype=dtype)

    c = a ** b
    assert err_rel(c.full(), tn.einsum('abc,def->abcdef', a.full(), b.full())) < 1e-12, 'Kronecker product error: 2 tensors.'

    A = tntt.random([(2, 3), (4, 5)], [1, 2, 1], dtype=dtype)
    B = tntt.random([(3, 3), (4, 2)], [1, 3, 1], dtype=dtype)

    C = A ** B
    assert err_rel(C.full(), tn.einsum('abcd,mnop->abmncdop', A.full(), B.full())) < 1e-12, 'Kronecker product error: 2 tensor operators.'

    c = a ** None
    assert err_rel(a.full(), c.full()) < 1e-14, 'Kronecker product error: tensor and None.'

    c = a ** tntt.ones([])
    assert err_rel(a.full(), c.full()) < 1e-14, 'Kronecker product error: tensor and None.'

@pytest.mark.parametrize("dtype", parameters)
def test_combination(dtype):
    """
    Test sequence of linear algebra operations.
    """

    x = tntt.random([4, 7, 13, 14, 19], [1, 2, 10, 13, 10, 1], dtype=dtype)
    y = tntt.random([4, 7, 13, 14, 19], [1, 2, 4, 2, 4, 1], dtype=dtype)

    x = x / x.norm()
    y = y / y.norm()

    z = x * x - 2 * x * y + y * y
    u = (x - y) * (x - y)
    norm = (z - u).norm()

    assert norm.numpy() < 1e-14, "Error: Multiple operations. Part 1 fails."

@pytest.mark.parametrize("dtype", parameters)
def test_slicing(dtype):
    cores = [tn.rand([1, 9, 3], dtype=dtype), tn.rand([3, 10, 4], dtype=dtype), tn.rand(
        [4, 15, 5], dtype=dtype), tn.rand([5, 15, 1], dtype=dtype)]
    Att = tntt.TT(cores)
    A = Att.full()

    assert err_rel(A[1, 2, 3, 4], Att[1, 2, 3, 4]) < 1e-15, "Tensor slicing error: slice to a scalar."
    assert err_rel(A[1:3, 2:4, 3:10, 4], Att[1:3, 2:4, 3:10, 4].full()) < 1e-15, "Tensor slicing error: eliminate some dimension."
    assert err_rel(A[1, :, 3, 4], Att[1, :, 3, 4].full()) < 1e-15, "Tensor slicing error: slice to 1d."
    assert err_rel(A[None, :, 2, :, None, 4, None], Att[None, :, 2, :, None, 4, None].full()) < 1e-15, "Tensor slicing error: add dimensions."
    assert err_rel(A[None, None, 1, 2, 2, None, 4, None, None], Att[None, None, 1, 2, 2, None, 4, None, None].full()) < 1e-15, "Tensor slicing error: add more dimensions."
    assert err_rel(A[..., 1, 1], Att[..., 1, 1].full()) < 1e-15, "Tensor slicing error: Ellipsis in the beginning."
    assert err_rel(A[1, ...], Att[1, ...].full()) < 1e-15, "Tensor slicing error: ellipsis in the end."
    assert err_rel(A[...], Att[...].full()) < 1e-15, "Tensor slicing error: ellipsis only."
    assert err_rel(A[None, None, ...], Att[None, None, ...].full()) < 1e-15, "Tensor slicing error: ellipsis and None only."
    assert err_rel(A[..., None], Att[..., None].full()) < 1e-15, "Tensor slicing error: None and ellipsis only."

@pytest.mark.parametrize("dtype", parameters)
def test_qtt(dtype):
    N = [16, 8, 64, 128]
    R = [1, 2, 10, 12, 1]
    x = tntt.random(N, R, dtype=dtype)
    x_qtt = x.to_qtt()
    x_full = x.full()

    assert err_rel(tn.reshape(x_qtt.full(), x.N), x_full) < 1e-12, 'Tensor to QTT failed.'

    x = tntt.random([256, 128, 1024, 128], [1, 40, 50, 20, 1], dtype=dtype)
    N = x.N
    xq = x.to_qtt()
    xx = xq.qtt_to_tens(N)

    assert np.abs((x-xx).norm(True)/x.norm(True)) < 1e-12, 'TT to QTT and back not working.'

@pytest.mark.parametrize("dtype", parameters)
def test_reshape(dtype):
    '''
    Test the reshape function.
    '''

    T = tntt.ones([3, 2], dtype=dtype)
    Tf = T.full()
    Tr = tntt.reshape(T, [6])

    assert err_rel(Tr.full(), tn.reshape(Tf, [6]))< 1e-11, 'TT-tensor reshape fail: test 1'

    T = tntt.random([6, 8, 9], [1, 4, 5, 1], dtype=dtype)
    Tf = T.full()
    Tr = tntt.reshape(T, [2, 6, 12, 3])

    assert err_rel(Tr.full(), tn.reshape(Tf, [2, 6, 12, 3])) < 1e-11, 'TT-tensor reshape fail: test 2'

    T = tntt.random([6, 8, 9], [1, 4, 5, 1], dtype=dtype)
    Tf = T.full()
    Tr = tntt.reshape(T, [2, 3, 4, 2, 3, 3])

    assert err_rel(Tr.full(), tn.reshape(Tf, [2, 3, 4, 2, 3, 3])) < 1e-11, 'TT-tensor reshape fail: test 3'

    T = tntt.random([2, 3, 4, 2, 3, 2, 5], [1, 2, 3, 4,
                    4, 5, 2, 1], dtype=dtype)
    Tf = T.full()
    Tr = tntt.reshape(T, [6, 24, 10])

    assert err_rel(Tr.full(), tn.reshape(Tf, [6, 24, 10])) < 1e-11, 'TT-tensor reshape fail: test 4'

    T = tntt.random([32, 32], [1, 4, 1], dtype=dtype)
    Tf = T.full()
    Tr = tntt.reshape(T, [2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1])

    assert err_rel(Tr.full(), tn.reshape(Tf, [2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1])) < 1e-11, 'TT-tensor reshape fail: test 5'

    T = tntt.random([2]*10, [1] + [8]*9 + [1], dtype=dtype)
    Tf = T.full()
    Tr = tntt.reshape(T, [2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1])

    assert err_rel(Tr.full(), tn.reshape(Tf, [2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1])) < 1e-11, 'TT-tensor reshape fail: test 6'

    # test TT-matrix

    A = tntt.random([(9, 4), (16, 6)], [1, 4, 1], dtype=dtype)
    Af = A.full()
    Ar = tntt.reshape(A, [(3, 2), (3, 2), (4, 2), (4, 3)])

    assert tn.linalg.norm(tn.reshape(Af, Ar.M+Ar.N)-Ar.full()).numpy() < 1e-12, 'TT-matrix reshape fail: test 1'

    A = tntt.random([(9, 4), (16, 6), (3, 5)], [
                    1, 4, 5, 1], dtype=dtype)
    Af = A.full()
    Ar = tntt.reshape(A, [(3, 2), (6, 6), (24, 10)])

    assert err_rel(Ar.full(), tn.reshape(Af, Ar.M+Ar.N)) < 1e-13, 'TT-matrix reshape fail: test 2'

    A = tntt.random([(4, 8), (16, 12), (2, 8), (6, 4)], [
                    1, 4, 7, 2, 1], dtype=dtype)
    T = tntt.random([8, 12, 8, 4], [1, 3, 9, 3, 1], dtype=dtype)
    Ar = tntt.reshape(A, [(2, 4), (4, 6), (4, 2), (8, 32), (3, 2)])
    Tr = tntt.reshape(T, [4, 6, 2, 32, 2])
    Af = A.full()
    Tf = T.full()
    Ur = Ar@Tr
    U = A@T
    
    assert err_rel(Ur.full(), tn.reshape(U.full(), Ur.N)) < 1e-13, 'TT-matrix reshape fail: test 3'

    A = tntt.random([(2, 2), (4, 2), (2, 2)], [1, 3, 4, 1])
    Af = A.full()
    Ar = tntt.reshape(A, [(2, 2), (2, 2), (2, 1), (2, 2)])
    assert err_rel(Ar.full(), tn.reshape(Af, Ar.M+Ar.N)) < 1e-13, 'TT-matrix reshape fail: test 4'

@pytest.mark.parametrize("dtype", parameters)
def test_mask(dtype):
    """
    Test the apply_mask() method.
    """
    indices = tn.randint(0, 20, (1000, 4))

    x = tntt.random([21, 22, 23, 21], [1, 10, 10, 10, 1],
                    dtype=dtype)
    xf = x.full()

    vals = x.apply_mask(indices)
    vals_ref = 0*vals
    for i in range(len(indices)):
        vals_ref[i] = xf[tuple(indices[i])]

    assert tn.linalg.norm(vals-vals_ref) < 1e-12, "Mask method error."

@pytest.mark.parametrize("dtype", parameters)
def test_bilinear(dtype):
    """
    Test the method torchtt.bilinear_form()
    """
    A = tntt.random([(5, 6), (7, 8), (2, 3), (4, 5)], [
                    1, 5, 5, 3, 1], dtype=dtype)
    x = tntt.randn([5, 7, 2, 4], [1, 2, 3, 4, 1], dtype=dtype)
    y = tntt.randn([6, 8, 3, 5], [1, 6, 5, 4, 1], dtype=dtype)

    res = tntt.bilinear_form(x, A, y)
    res_ref = tn.einsum('abcd,abcdijkl,ijkl->',
                        tn.conj(x.full()), A.full(), y.full())

    assert err_rel(res, res_ref) < 5e-13, "torchtt.bilinear_form() failed."

@pytest.mark.parametrize("dtype", parameters)
def test_conj(dtype):
    """
    Test the conjugate.
    """

    A = tntt.random([(5, 6), (7, 8), (2, 3), (4, 5)], [
                    1, 5, 5, 3, 1], dtype=dtype)
    x = tntt.randn([5, 7, 2, 4], [1, 2, 3, 4, 1], dtype=dtype)

    assert err_rel(x.conj().full(), tn.conj(x.full())) <5e-13, "torchtt.TT.conj() failed."
    assert err_rel(A.conj().full(), tn.conj(A.full())) < 5e-13, "torchtt.TT.conj() failed fpr TT matrix."

@pytest.mark.parametrize("dtype", parameters)
def test_cat(dtype):
    """
    Test the concatenation of tensors.
    """

    a1 = tntt.randn((3, 4, 2, 6, 7), [1, 2, 3, 2, 4, 1])
    a2 = tntt.randn((3, 4, 8, 6, 7), [1, 3, 2, 2, 1, 1])
    a3 = tntt.randn((3, 4, 15, 6, 7), [1, 3, 7, 7, 5, 1])

    a = tntt.cat((a1, a2, a3), 2)

    af = tn.cat((a1.full(), a2.full(), a3.full()), 2)
    assert tn.linalg.norm(a.full()-af) / tn.linalg.norm(af) < 1e-14, "torchtt.cat() failed."

@pytest.mark.parametrize("dtype", parameters)
def test_pad(dtype):
    """
    Test the tensor padding in TT.
    """

    A = tntt.randn((5, 6, 7, 8), (1, 2, 3, 4, 1))

    Ap = tntt.pad(A, ((1, 2), (1, 4), (3, 5), (2, 1)), value=1/2)

    assert tn.linalg.norm(A.full()-Ap.full()[1:6, 1:7, 3:10, 2:10])/Ap.norm() < 1e-15, "torchtt.pad() fail 1."
    assert abs(tn.mean(Ap.full()[6:, 7:, 10:, 10:])-1/2) < 1e-15, "torchtt.pad() fail 2."
    assert abs(tn.mean(Ap.full()[:1, :1, :3, :2])-1/2) < 1e-15, "torchtt.pad() fail 3."

    # TTM case

    M = tntt.randn(((3, 2), (4, 4), (5, 2)), (1, 3, 2, 1))
    Mp = tntt.pad(M, ((1, 2), (1, 4), (3, 5)), value=1/2)

    err = tn.linalg.norm(M.full() - Mp.full()
                            [1:4, 1:5, 3:8, 1:3, 1:5, 3:5])
    assert err/M.norm() < 1e-15, "torchtt.pad() TTM fail 1."

    n = 3
    err = tn.linalg.norm(tn.reshape(
        Mp.full()[:1, :1, :3, :1, :1, :3], [n, n]) - 1/2*tn.eye(n))
    assert err/n <1e-15, "torchtt.pad() TTM fail 2."

    n = 40
    err = tn.linalg.norm(tn.reshape(
        Mp.full()[4:, 5:, 8:, 3:, 5:, 5:], [n, n])-tn.eye(n)/2)
    assert err/n < 1e-15, "torchtt.pad() TTM fail 3."

@pytest.mark.parametrize("dtype", parameters)
def test_diag(dtype):
    """
    Test the torchtt.diag() function.
    """

    n = [3, 4, 5]
    I = tntt.eye(n, dtype=tn.float64)
    dg = tntt.diag(I)

    assert err_rel(dg.full(), tntt.ones(n).full()) < 1e-13, "torchtt.diag() TTM->TT failed."

    o = tntt.ones(n)
    E = tntt.diag(o)

    assert err_rel(E.full(), tntt.eye(n).full()) < 1e-13, "torchtt.diag() TT->TTM failed."


