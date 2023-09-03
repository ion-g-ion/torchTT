import pytest
import torchtt as tntt
import torch as tn
import numpy as np


def err_rel(t, ref): return tn.linalg.norm(t-ref).numpy() / \
    tn.linalg.norm(ref).numpy() if ref.shape == t.shape else np.inf


parameters = [tn.float64, tn.complex128]


@pytest.mark.parametrize("dtype", parameters)
def test_init(dtype):
    """
    Checks the constructor and the TT.full() function. 
    A list of cores is passed and is checked if the recomposed tensor is correct.
    """

    # print('Testing: Initialization from list of cores.')
    cores = [tn.rand([1, 20, 3], dtype=dtype), tn.rand(
        [3, 10, 4], dtype=dtype), tn.rand([4, 5, 1], dtype=dtype)]

    T = tntt.TT(cores)

    Tfull = T.full()
    T_ref = tn.squeeze(tn.einsum('ijk,klm,mno->ijlno',
                       cores[0], cores[1], cores[2]))

    assert err_rel(Tfull, T_ref) < 1e-14


@pytest.mark.parametrize("dtype", parameters)
def test_decomposition_random(dtype):
    '''
    Perform a TT decomposition of a random full random tensor and check if the decomposition is accurate.
    '''
    # print('Testing: TT-decomposition from full (random tensor).')
    T_ref = tn.rand([10, 20, 30, 5], dtype=dtype)

    T = tntt.TT(T_ref, eps=1e-19, rmax=1000)

    Tfull = T.full()

    assert err_rel(Tfull, T_ref) < 1e-12


@pytest.mark.parametrize("dtype", parameters)
def test_decomposition_lowrank(dtype):
    """
    Check the decomposition of a tensor which is already in the low rank format.

    """
    # print('Testing: TT-decomposition from full (already low-rank).')
    cores = [tn.rand([1, 200, 30], dtype=dtype), tn.rand(
        [30, 100, 4], dtype=dtype), tn.rand([4, 50, 1], dtype=dtype)]
    T_ref = tn.squeeze(tn.einsum('ijk,klm,mno->ijlno',
                       cores[0], cores[1], cores[2]))

    T = tntt.TT(T_ref, eps=1e-19)

    Tfull = T.full()

    assert err_rel(Tfull, T_ref) < 1e-12


@pytest.mark.parametrize("dtype", parameters)
def test_decomposition_highd(dtype):
    """
    Decompose a 20d tensor with all modes 2.

    Returns
    -------
    None.

    """
    # print('Testing: TT-decomposition from full (long  20d TT).')
    cores = [tn.rand([1, 2, 16], dtype=dtype)] + [tn.rand([16, 2, 16], dtype=dtype)
                                                  for i in range(18)] + [tn.rand([16, 2, 1], dtype=dtype)]
    T_ref = tntt.TT(cores).full()

    T = tntt.TT(T_ref, eps=1e-12)
    Tfull = T.full()

    assert err_rel(Tfull, T_ref) < 1e-12


@pytest.mark.parametrize("dtype", parameters)
def test_decomposition_ttm(dtype):
    """
    Decompose a TT-matrix.

    Returns
    -------
    bool
        True if test is passed.

    """

    T_ref = tn.rand([10, 11, 12, 15, 17, 19], dtype=dtype)

    T = tntt.TT(T_ref, shape=[(10, 15), (11, 17),
                (12, 19)], eps=1e-19, rmax=1000)
    Tfull = T.full()

    assert err_rel(Tfull, T_ref) < 1e-12


@pytest.mark.parametrize("dtype", parameters)
def test_decomposition_orthogonal(dtype):
    """
    Checks the lr_orthogonal function. The reconstructed tensor should remain the same.
    """
    # print('Testing: TT-orthogonalization.')
    cores = [tn.rand([1, 20, 3], dtype=dtype), tn.rand([3, 10, 4], dtype=dtype), tn.rand(
        [4, 5, 20], dtype=dtype), tn.rand([20, 5, 2], dtype=dtype), tn.rand([2, 10, 1], dtype=dtype)]
    T = tntt.TT(cores)
    T = tntt.random([3, 4, 5, 3, 8, 7, 10, 3, 5, 6], [
                    1, 20, 12, 34, 3, 50, 100, 12, 2, 80, 1], dtype=dtype)
    T_ref = T.full()

    cores, R = tntt._decomposition.lr_orthogonal(T.cores, T.R, T.is_ttm)
    Tfull = tntt.TT(cores).full()

    assert err_rel(Tfull, T_ref) < 1e-12, 'Left to right ortho error too high.'

    for i in range(len(cores)):
        c = cores[i]
        L = tn.reshape(c, [-1, c.shape[-1]]).numpy()
        assert np.linalg.norm(L.T @ np.conj(L) - np.eye(L.shape[1])) < 1e-12 or i == len(
            cores)-1, 'Cores are not left orthogonal after LR orthogonalization.'

    cores, R = tntt._decomposition.rl_orthogonal(T.cores, T.R, T.is_ttm)
    Tfull = tntt.TT(cores).full()

    assert err_rel(Tfull, T_ref) < 1e-12, 'Right to left ortho error too high.'

    for i in range(len(cores)):
        c = cores[i]
        R = tn.reshape(c, [c.shape[0], -1]).numpy()
        assert np.linalg.norm(
            np.conj(R) @ R.T - np.eye(R.shape[0])) < 1e-12 or i == 0


@pytest.mark.parametrize("dtype", parameters)
def test_decomposition_orthogonal_ttm(dtype):
    """
    Test the lr and rt orthogonal functions for a TT matrix.
    """
    T = tntt.random([(3, 4), (5, 6), (7, 8), (9, 4)],
                    [1, 2, 3, 4, 1], dtype=dtype)
    T_ref = T.full()

    cores, R = tntt._decomposition.lr_orthogonal(T.cores, T.R, T.is_ttm)
    Tfull = tntt.TT(cores).full()

    assert err_rel(Tfull, T_ref) < 1e-12, 'Left to right ortho error too high.'

    for i in range(len(cores)):
        c = cores[i]
        L = tn.reshape(c, [-1, c.shape[-1]]).numpy()
        assert np.linalg.norm(L.T @ np.conj(L) - np.eye(L.shape[1])) < 1e-12 or i == len(
            cores)-1, 'Cores are not left orthogonal after LR orthogonalization.'

    cores, R = tntt._decomposition.rl_orthogonal(T.cores, T.R, T.is_ttm)
    Tfull = tntt.TT(cores).full()

    assert err_rel(Tfull, T_ref) < 1e-12, 'Right to left ortho error too high.'

    for i in range(len(cores)):
        c = cores[i]
        R = tn.reshape(c, [c.shape[0], -1]).numpy()
        assert np.linalg.norm(
            np.conj(R) @ R.T - np.eye(R.shape[0])) < 1e-12 or i == 0


@pytest.mark.parametrize("dtype", parameters)
def test_decomposition_rounding(dtype):
    """
    Testing the rounding of a TT-tensor.
    A rank-4tensor is constructed and successive approximations are performed.
    """
    # print('Testing: TT-rounding.')

    T1 = tn.einsum('i,j,k->ijk', tn.rand([20], dtype=dtype),
                   tn.rand([30], dtype=dtype), tn.rand([32], dtype=dtype))
    T2 = tn.einsum('i,j,k->ijk', tn.rand([20], dtype=dtype),
                   tn.rand([30], dtype=dtype), tn.rand([32], dtype=dtype))
    T3 = tn.einsum('i,j,k->ijk', tn.rand([20], dtype=dtype),
                   tn.rand([30], dtype=dtype), tn.rand([32], dtype=dtype))
    T4 = tn.einsum('i,j,k->ijk', tn.rand([20], dtype=dtype),
                   tn.rand([30], dtype=dtype), tn.rand([32], dtype=dtype))

    T_ref = T1 / tn.linalg.norm(T1) + 1e-3*T2 / tn.linalg.norm(T2) + \
        1e-6*T3 / tn.linalg.norm(T3) + 1e-9*T4 / tn.linalg.norm(T4)
    T3 = T1 / tn.linalg.norm(T1) + 1e-3*T2 / \
        tn.linalg.norm(T2) + 1e-6*T3 / tn.linalg.norm(T3)
    T2 = T1 / tn.linalg.norm(T1) + 1e-3*T2 / tn.linalg.norm(T2)
    T1 = T1 / tn.linalg.norm(T1)

    T = tntt.TT(T_ref)
    T = T.round(1e-9)
    Tfull = T.full()
    assert T.R == [1, 3, 3, 1], 'Case 1: Ranks not equal'
    assert err_rel(Tfull, T_ref) < 1e-9, 'Case 1: error too high'

    T = tntt.TT(T_ref)
    T = T.round(1e-6)
    Tfull = T.full()
    assert T.R == [1, 2, 2, 1], 'Case 2: Ranks not equal'
    assert err_rel(Tfull, T_ref) < 1e-6, 'Case 1: error too high'

    T = tntt.TT(T_ref)
    T = T.round(1e-3)
    Tfull = T.full()
    assert T.R == [1, 1, 1, 1], 'Case 3: Ranks not equal'
    assert err_rel(Tfull, T_ref) < 1e-3, 'Case 1: error too high'


@pytest.mark.parametrize("dtype", parameters)
def test_dimension_permute(dtype):
    """
    Test the permute function.
    """
    x_tt = tntt.random([5, 6, 7, 8, 9], [1, 2, 3, 4, 2, 1])
    x_ref = x_tt.full()
    xp_tt = tntt.permute(x_tt, [4, 3, 2, 1, 0], 1e-10)
    xp_ref = tn.permute(x_ref, [4, 3, 2, 1, 0])

    assert tuple(xp_tt.N) == tuple(
        xp_ref.shape), 'Permute modex of a TT tensor: shape mismatch.'
    assert err_rel(
        xp_tt.full(), xp_ref) < 1e-10, 'Permute modex of a TT tensor: error too high.'

    # Test for TT matrices
    A_tt = tntt.random([(2, 3), (4, 5), (3, 2), (6, 7),
                       (5, 3)], [1, 2, 3, 4, 2, 1])
    A_ref = A_tt.full()
    Ap_tt = tntt.permute(A_tt, [3, 2, 4, 0, 1])
    Ap_ref = tn.permute(A_ref, [3, 2, 4, 0, 1, 8, 7, 9, 5, 6])

    assert Ap_tt.M == [6, 3, 5, 2,
                       4], 'Permute modex of a TT matrix: shape mismatch.'
    assert Ap_tt.N == [7, 2, 3, 3,
                       5], 'Permute modex of a TT matrix: shape mismatch.'
    assert err_rel(
        Ap_tt.full(), Ap_ref) < 1e-10, 'Permute modex of a TT tensor: error too high.'
