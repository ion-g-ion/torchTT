"""
Fast products in TT.
Taken from [https://arxiv.org/pdf/2410.19747](https://arxiv.org/pdf/2410.19747).
The products are implemented in :class:`torchtt.methods.Swap`.

@author: ion
"""
import warnings
import torchtt
import torch as tn
from torchtt._decomposition import rank_chop, QR, SVD
import opt_einsum as oe
import sys
from torchtt.errors import *

def swap_cores(core_a, core_b, eps, rmax=None):
    """
    Swap two condsecutive TT or TTM cores.

        -- A ---- B --   =>  -- B ---- A --
           |      |             |      |

    Args:
        core_a (torch.Tensor): first TT/TTM core
        core_b (torch.Tensor): second TT/TTM core
        eps (float): accuracy
        rmax (int, optional): maximum rank. Defaults to None.

    Raises:
        Exception: The cores must be wither 3d or 4d tensors.

    Returns:
        torch.Tensor, torch.Tensor: the swapped cores
    """
    if len(core_a.shape) == 3 and len(core_b.shape) == 3:
        supercore = oe.contract("rms,snR->rnmR", core_a, core_b)
        U, S, V = SVD(tn.reshape(supercore, (core_a.shape[0] * core_b.shape[1], -1)))
    elif len(core_a.shape) == 4 and len(core_b.shape) == 4:
        supercore = oe.contract("rmas,snbR->rnbmaR", core_a, core_b)
        U, S, V = SVD(tn.reshape(supercore, (core_a.shape[0] * core_b.shape[1] * core_b.shape[2], -1)))
    else:
        raise Exception("The cores must be wither 3d or 4d tensors.")
    
    if S.is_cuda:
        r_now = min([rank_chop(S.detach().cpu().numpy(),tn.linalg.norm(S).detach().cpu().numpy()*eps)])
    else:
        r_now = min([rank_chop(S.detach().numpy(),tn.linalg.norm(S).detach().numpy()*eps)])
                
    if rmax is not None:
        r_now = min(r_now, rmax)
    US = U[:,:r_now] @ tn.diag(S[:r_now])
    V = V[:r_now,:]
    

    if len(core_a.shape) == 3 and len(core_b.shape) == 3:
        return tn.reshape(US, (core_a.shape[0], core_b.shape[1], -1)), tn.reshape(V, (-1, core_a.shape[1], core_b.shape[2]))
    elif len(core_a.shape) == 4 and len(core_b.shape) == 4:
        return tn.reshape(US, (core_a.shape[0], core_b.shape[1], core_b.shape[2], -1)), tn.reshape(V, (-1, core_a.shape[1], core_a.shape[2], core_b.shape[3]))


def fast_hadammard(tt_a, tt_b, eps=1e-10, rmax=None):
    """
    Performs the elementwise multiplication between two TTs or TT matrices (TTMs) and rounds the result.
    Equivalent to `(tt_a * tt_b).round(eps)`.
    Method described in :cite:p:`michailidis2025elementwise`.

    .. deprecated:: 0.6.0
        Use :func:`torchtt.hadamard` with ``method='swap'``.

    Args:
        tt_a (torchtt.TT): first operand.
        tt_b (torchtt.TT): second operand.
        eps (float, optional): relative tolerance. Defaults to 1e-10.
        rmax (int, optional): maximum rank. Defaults to None.

    Returns:
        torchtt.TT: the result.
    """
    warnings.warn("torchtt.fast_hadammard() is deprecated, use torchtt.hadamard(x, y, eps, method='swap') instead.", DeprecationWarning, stacklevel=2)
    return torchtt.hadamard(tt_a, tt_b, eps, sys.maxsize if rmax is None else rmax, method='swap')

def fast_mv(tt_a, tt_b, eps=1e-10, rmax=None):
    """
    Performs the matvec product between a TT matrix (TTM) and a TT tensor.
    Equivalent to `(tt_a @ tt_b).round(eps)`.
    Method described in :cite:p:`michailidis2025elementwise`.

    .. deprecated:: 0.6.0
        Use :func:`torchtt.matvec` with ``method='swap'``.

    Args:
        tt_a (torchtt.TT): the first operand. Must be a TTM.
        tt_b (torchtt.TT): the second operand. Must be TT.
        eps (float, optional): Relative tolerance. Defaults to 1e-10.
        rmax (int, optional): maximum rank. Defaults to None.

    Returns:
        torchtt.TT: the result. This is a TT.
    """
    warnings.warn("torchtt.fast_mv() is deprecated, use torchtt.matvec(A, x, eps, method='swap') instead.", DeprecationWarning, stacklevel=2)
    return torchtt.matvec(tt_a, tt_b, eps, sys.maxsize if rmax is None else rmax, method='swap')

def fast_mm(tt_a, tt_b, eps=1e-10, rmax=None):
    """
    Performs the matmat product between two TT matrices (TTMs).
    Equivalent to `(tt_a @ tt_b).round(eps)`.
    Method described in :cite:p:`michailidis2025elementwise`.

    .. deprecated:: 0.6.0
        Use :func:`torchtt.matmat` with ``method='swap'``.

    Args:
        tt_a (torchtt.TT): the first operand. Must be a TTM.
        tt_b (torchtt.TT): the second operand. Must be TTM.
        eps (float, optional): Relative tolerance. Defaults to 1e-10.
        rmax (int, optional): maximum rank. Defaults to None.

    Returns:
        torchtt.TT: the result. This is a TTM.
    """
    warnings.warn("torchtt.fast_mm() is deprecated, use torchtt.matmat(A, B, eps, method='swap') instead.", DeprecationWarning, stacklevel=2)
    return torchtt.matmat(tt_a, tt_b, eps, sys.maxsize if rmax is None else rmax, method='swap')
