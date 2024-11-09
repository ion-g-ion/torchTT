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
def test_hadamard(dtype):
    '''
    Test the hadamard fast multiplication betwenn TTs 
    '''
    N = [2, 3, 4, 2, 3]

    x = tntt.random(N, [1, 3, 2, 3, 4, 1], dtype=dtype)
    y = tntt.random(N, [1, 2, 2, 5, 4, 1], dtype=dtype)
    
    X = x.clone()
    X = X + X
    X = X + X
#    X += 1e2*x 

    Y = y.clone()
    Y = Y + Y
    Y = Y + Y
 #   Y += 1e2*x   
    
    z_ref = 16*x*y 

    z = tntt.fast_hadammard(X, Y, 1e-9)

    assert z.N == z_ref.N
    assert err_rel(z.full(), z_ref.full()) < 1e-9

@pytest.mark.parametrize("dtype", parameters)
def test_hadamard_ttm(dtype):
    '''
    Test the hadamard fast multiplication betwenn TTMs 
    '''
    M = [3, 2, 2, 4]
    N = [2, 3, 4, 2]
    MN = [(M[i], N[i]) for i in range(4)]
    
    x = tntt.random(MN, [1, 3, 2, 3, 1], dtype=dtype)
    y = tntt.random(MN, [1, 2, 2, 5, 1], dtype=dtype)

    X = x.clone()
    X = X + X
    X = X + X
#    X += 1e2*x 

    Y = y.clone()
    Y = Y + Y
    Y = Y + Y
 #   Y += 1e2*x   
    
    z_ref = 16*x*y 

    z = tntt.fast_hadammard(X, Y, 1e-9)

    assert z.N == z_ref.N
    assert z.M == z_ref.M
    assert err_rel(z.full(), z_ref.full()) < 1e-9
 
@pytest.mark.parametrize("dtype", parameters)
def test_mv(dtype):
    '''
    Test the fast multiplication betwenn TTM and TT 
    '''
    M = [3, 2, 2, 4]
    N = [2, 3, 4, 2]
    MN = [(M[i], N[i]) for i in range(4)]
    
    x = tntt.random(MN, [1, 3, 2, 3, 1], dtype=dtype)
    y = tntt.random(N, [1, 2, 2, 5, 1], dtype=dtype)

    X = x.clone()
    X = X + X
    X = X + X
#    X += 1e2*x 

    Y = y.clone()
    Y = Y + Y
    Y = Y + Y
 #   Y += 1e2*x   
    
    z_ref = 16*x@y 

    z = tntt.fast_mv(X, Y, 1e-9)

    assert z.N == z_ref.N

    assert err_rel(z.full(), z_ref.full()) < 1e-9

@pytest.mark.parametrize("dtype", parameters)
def test_mm(dtype):
    '''
    Test the fast multiplication betwenn TTM and TTM 
    '''
    M = [3, 2, 2, 4]
    N = [2, 3, 4, 2]
    K = [4, 2, 3, 3]
    MN = [(M[i], N[i]) for i in range(4)]
    NK = [(N[i], K[i]) for i in range(4)]

    x = tntt.random(MN, [1, 3, 2, 3, 1], dtype=dtype)
    y = tntt.random(NK, [1, 2, 2, 5, 1], dtype=dtype)

    X = x.clone()
    X = X + X
    X = X + X
#    X += 1e2*x 

    Y = y.clone()
    Y = Y + Y
    Y = Y + Y
 #   Y += 1e2*x   
    
    z_ref = 16*x@y 

    z = tntt.fast_mm(X, Y, 1e-9)

    assert z.N == z_ref.N
    assert z.M == z_ref.M

    assert err_rel(z.full(), z_ref.full()) < 1e-9