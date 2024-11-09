"""
Fast products in TT.
Taken from [https://arxiv.org/pdf/2410.19747](https://arxiv.org/pdf/2410.19747).

@author: ion
"""
import torchtt
import torch as tn
from torchtt._decomposition import rank_chop, QR, SVD
import opt_einsum as oe
from torchtt.errors import *

def swap_cores(core_a, core_b, eps):
    """
    Swap two condsecutive TT or TTM cores.

        -- A ---- B --   =>  -- B ---- A --
           |      |             |      |

    Args:
        core_a (torch.Tensor): first TT/TTM core
        core_b (torch.Tensor): second TT/TTM core
        eps (float): accuracy

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
        r_now = min([rank_chop(S.cpu().numpy(),tn.linalg.norm(S).cpu().numpy()*eps)])
    else:
        r_now = min([rank_chop(S.numpy(),tn.linalg.norm(S).numpy()*eps)])
                
    US = U[:,:r_now] @ tn.diag(S[:r_now])
    V = V[:r_now,:]
    

    if len(core_a.shape) == 3 and len(core_b.shape) == 3:
        return tn.reshape(US, (core_a.shape[0], core_b.shape[1], -1)), tn.reshape(V, (-1, core_a.shape[1], core_b.shape[2]))
    elif len(core_a.shape) == 4 and len(core_b.shape) == 4:
        return tn.reshape(US, (core_a.shape[0], core_b.shape[1], core_b.shape[2], -1)), tn.reshape(V, (-1, core_a.shape[1], core_a.shape[2], core_b.shape[3]))


def fast_hadammard(tt_a, tt_b, eps=1e-10):
    """
    Performs the elementwise multiplication between two TTs to TTMs and tround the result.
    Equivalent to `(tt_a * tt_b).round(eps)`.
    Method described in [https://arxiv.org/pdf/2410.19747](https://arxiv.org/pdf/2410.19747).

    Args:
        tt_a (torchtt.TT): first operand.
        tt_b (torchtt.TT): second operand.
        eps (float, optional): relative tolerance. Defaults to 1e-10.

    Returns:
        torchtt.TT: the result.
    """
    if tt_a.is_ttm != tt_b.is_ttm:
        raise InvalidArguments("The two tensors should be either TT or TTMs.")
    
    if tt_a.is_ttm:
        if tt_a.N != tt_b.N  or tt_a.M != tt_b.M :
           raise ShapeMismatch("The two tensors should have the same shapes.") 
        
        d = len(tt_a.N)

        cores = [tn.permute(c, [3, 1, 2, 0]) for c in tt_b.cores[::-1]]
        for i in range(d):
            cores[0] = oe.contract("maAk,kbBn,AB,ab->maAn", tt_a.cores[d-i-1], cores[0], tn.eye(tt_a.N[d-i-1], device=tt_a.cores[d-i-1].device, dtype=cores[0].dtype), tn.eye(tt_a.M[d-i-1], device=tt_a.cores[d-i-1].device, dtype=cores[0].dtype))
            
            if i != d-1:
                for j in range(i, -1, -1):
                    cores[j], cores[j+1] = swap_cores(cores[j], cores[j+1], eps)
                
                
        #cores[1], cores[2] = swap_cores(cores[1], cores[2], 1e-8)
            
        return torchtt.TT(cores)
    else:
        if tt_a.N != tt_b.N:
           raise ShapeMismatch("The two tensors should have the same shapes.") 

        d = len(tt_a.N)

        cores = [tn.permute(c, [2, 1, 0]) for c in tt_b.cores[::-1]]
        for i in range(d):
            cores[0] = oe.contract("mak,kbn,ab->man", tt_a.cores[d-i-1], cores[0], tn.eye(tt_a.N[d-i-1], device=tt_a.cores[d-i-1].device, dtype=cores[0].dtype))
            
            if i != d-1:
                for j in range(i, -1, -1):
                    cores[j], cores[j+1] = swap_cores(cores[j], cores[j+1], eps)
                
                
        #cores[1], cores[2] = swap_cores(cores[1], cores[2], 1e-8)
            
        return torchtt.TT(cores)

def fast_mv(tt_a, tt_b, eps=1e-10):
    """
    Performs the matvec product between a TTM and a TT.
    Equivalent to `(tt_a * tt_b).round(eps)`.
    Method described in [https://arxiv.org/pdf/2410.19747](https://arxiv.org/pdf/2410.19747).

    Args:
        tt_a (torchtt.TT): the first operand. Must be a TTM.
        tt_b (torchtt.TT): the second operand. Must be TT.
        eps (float, optional): Relative tolerance. Defaults to 1e-10.

    Raises:
        InvalidArguments: The first should be e TTM and the second a TT.
        ShapeMismatch: The shapes of the two operands must be compatible: tt_a.N == tt_b.N.

    Returns:
        torchtt.TT: the result. This is a TT.
    """
    
    if not tt_a.is_ttm or tt_b.is_ttm:
        raise InvalidArguments("The first should be e TTM and the second a TT.")

    if tt_a.N != tt_b.N:
        raise ShapeMismatch("The shapes of the two operands must be compatible: tt_a.N == tt_b.N.") 

    d = len(tt_a.N)

    cores = [tn.permute(c, [2, 1, 0]) for c in tt_b.cores[::-1]]
    for i in range(d):
        cores[0] = oe.contract("mabk,kbn->man", tt_a.cores[d-i-1], cores[0])
        
        if i != d-1:
            for j in range(i, -1, -1):
                cores[j], cores[j+1] = swap_cores(cores[j], cores[j+1], eps)
            
            
    #cores[1], cores[2] = swap_cores(cores[1], cores[2], 1e-8)
        
    return torchtt.TT(cores) 

def fast_mm(tt_a, tt_b, eps=1e-10):
    """
    Performs the matmat product between a TTM and a TTM.
    Equivalent to `(tt_a * tt_b).round(eps)`.
    Method described in [https://arxiv.org/pdf/2410.19747](https://arxiv.org/pdf/2410.19747).
    
    Args:
        tt_a (torchtt.TT): the first operand. Must be a TTM.
        tt_b (torchtt.TT): the second operand. Must be TTM.
        eps (float, optional): Relative tolerance. Defaults to 1e-10.

    Raises:
        InvalidArguments: Both arguments should be TTMs.
        ShapeMismatch: The shapes of the two operands must be compatible: tt_a.N == tt_b.M

    Returns:
        torchtt.TT: the result. This is a TTM.
    """
    
    if not tt_a.is_ttm or not tt_b.is_ttm:
        raise InvalidArguments("Both arguments should be TTMs.")

    if tt_a.N != tt_b.M:
        raise ShapeMismatch("The shapes of the two operands must be compatible: tt_a.N == tt_b.M") 

    d = len(tt_a.N)

    cores = [tn.permute(c, [3, 1, 2, 0]) for c in tt_b.cores[::-1]]
    for i in range(d):
        cores[0] = oe.contract("mabk,kbcn->macn", tt_a.cores[d-i-1], cores[0])
        
        if i != d-1:
            for j in range(i, -1, -1):
                cores[j], cores[j+1] = swap_cores(cores[j], cores[j+1], eps)
            
            
    #cores[1], cores[2] = swap_cores(cores[1], cores[2], 1e-8)
        
    return torchtt.TT(cores) 
