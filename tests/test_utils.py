
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

