"""
Manifold gradient module.

"""

import torch as tn
from torchtt._decomposition import mat_to_tt, to_tt, lr_orthogonal, round_tt, rl_orthogonal
from . import TT
from torchtt.errors import *

def _delta2cores(tt_cores, R, Sds, is_ttm = False, ortho = None):
    """
    Convert the detla notation to TT.
    Implements Algorithm 5.1 from "AUTOMATIC DIFFERENTIATION FOR RIEMANNIAN OPTIMIZATION ON LOW-RANK MATRIX AND TENSOR-TRAIN MANIFOLDS".

    Args:
        tt_cores (list[torch.tensor]): the TT cores.
        R (list[int]): the rank of the tensor.
        Sds (list[torch.tensor]): deltas.
        is_ttm (bool, optional): is TT amtrix or not. Defaults to False.
        ortho (list[list[torch.tensor]], optional): the left and right orthogonal cores of tt_cores. Defaults to None.

    Returns:
        list[torch.tensor]: the resulting TT cores.
    """
    
    if ortho == None:
        l_cores,_  = lr_orthogonal(tt_cores, R, is_ttm)
        r_cores,_  = rl_orthogonal(tt_cores, R, is_ttm)
    else:
        l_cores = ortho[0]
        r_cores = ortho[1]
    
    # first
    cores_new = [tn.cat((Sds[0],l_cores[0]),2 if not is_ttm else 3)]
    # 2...d-1
    for k in range(1,len(tt_cores)-1):
        up = tn.cat((r_cores[k],tn.zeros((r_cores[k].shape),dtype = l_cores[0].dtype, device = l_cores[0].device)),2 if not is_ttm else 3)
        down = tn.cat((Sds[k],l_cores[k]),2 if not is_ttm else 3)
        cores_new.append(tn.cat((up,down),0))
    # last
    cores_new.append(tn.cat((r_cores[-1],Sds[-1]),0))
    
    return cores_new

def riemannian_gradient(x,func):
    """
    Compute the Riemannian gradient using AD.

    Args:
        x (torchtt.TT): the point on the manifold where the gradient is computed.
        func ([type]): function that has to be differentiated. The function takes as only argument `torchtt.TT` instances.

    Returns:
        torchtt.TT: the gradient projected on the tangent space of x.
    """

    l_cores,_  = lr_orthogonal(x.cores, x.R, x.is_ttm)
    r_cores,_  = rl_orthogonal(l_cores, x.R, x.is_ttm)
    
    is_ttm = x.is_ttm

    
    R = x.R
    d = len(x.N)
    
    Rs = [ r_cores[0] ]
    Rs += [ x.cores[i]*0 for i in range(1,d)]
    
    # AD part
    for i in range(d):
        Rs[i].requires_grad_(True)
    Ghats = _delta2cores(x.cores, R, Rs, is_ttm = is_ttm,ortho = [l_cores,r_cores])
    fval = func(TT(Ghats))
    fval.backward() 

    # Sds = tape.gradient(fval, Rs)
    Sds = [r.grad for r in Rs]
    # print('Sds ',Sds)
  
    
    # compute Sdeltas
    for k in range(d-1):
        D = tn.reshape(Sds[k],[-1,R[k+1]])
        UL = tn.reshape(l_cores[k],[-1,R[k+1]])
        D = D - UL @ (UL.T @ D)
        Sds[k] = tn.reshape(D,l_cores[k].shape)
        
        
        
    # print([tf.einsum('ijk,ijl->kl',l_cores[i],Sds[i]).numpy() for i in range(d-1)])
    # delta to TT
    grad_cores = _delta2cores(x.cores, R, Sds, is_ttm,ortho = [l_cores,r_cores])
    return TT(grad_cores)
        
def riemannian_projection(Xspace,z):
    """
    Project the tensor z onto the tangent space defined at xspace

    Args:
        Xspace (torchtt.TT): the target where the tensor should be projected.
        z (torchtt.TT): the tensor that should be projected.

    Raises:
        IncompatibleTypes: Both must be of same type.

    Returns:
        torchtt.TT: the projection.
    """

    if Xspace.is_ttm != z.is_ttm:
        raise IncompatibleTypes('Both must be of same type.')
       
    is_ttm = Xspace.is_ttm
     
    l_cores,R  = lr_orthogonal(Xspace.cores, Xspace.R, Xspace.is_ttm)
    r_cores,_  = rl_orthogonal(l_cores, R, Xspace.is_ttm)
    
    d = len(Xspace.N)

    N = Xspace.N
    
    # Pleft = [tf.ones((1,1,1),dtype=Xspace.cores[0].dtype)]
    Pleft = []
    tmp = tn.ones((1,1),dtype=Xspace.cores[0].dtype, device = Xspace.cores[0].device)
    for k in range(d-1):
        if is_ttm:
            tmp = tn.einsum('rs,rijR,sijS->RS',tmp,l_cores[k],z.cores[k]) # size rk x sk
        else:
            tmp = tn.einsum('rs,riR,siS->RS',tmp,l_cores[k],z.cores[k]) # size rk x sk
        Pleft.append(tmp)
        
   
    
    Pright = []
    tmp = tn.ones((1,1), dtype = Xspace.cores[0].dtype, device = Xspace.cores[0].device)
    for k in range(d-1,0,-1):
        if is_ttm:
            tmp = tn.einsum('RS,rijR,sijS->rs',tmp,r_cores[k],z.cores[k]) # size rk x sk
        else:
            tmp = tn.einsum('RS,riR,siS->rs',tmp,r_cores[k],z.cores[k]) # size rk x sk
        Pright.append(tmp)
    Pright = Pright[::-1]
    
    
    # compute elements of the tangent space
    Sds = []
    for k in range(d):
  
        if k==0:
            L = tn.ones((1,1),dtype=Xspace.cores[0].dtype, device = Xspace.cores[0].device)
        else:
            L = Pleft[k-1]
        if k==d-1:
            if is_ttm:
                Sds.append(tn.einsum('rs,sjiS->rjiS',L,z.cores[k]))   
            else:
                Sds.append(tn.einsum('rs,siS->riS',L,z.cores[k]))           
        else:
            R = Pright[k]
            if is_ttm:
                tmp1 = tn.einsum('rs,sijS->rijS',L,z.cores[k])
                tmp2 = tn.einsum('rijR,RS->rijS',l_cores[k],tn.einsum('rs,rijR,sijS->RS',L,l_cores[k],z.cores[k]))
                Sds.append(tn.einsum('rijS,RS->rijR',tmp1-tmp2,R))
            else:
                tmp1 = tn.einsum('rs,siS->riS',L,z.cores[k])
                tmp2 = tn.einsum('riR,RS->riS',l_cores[k],tn.einsum('rs,riR,siS->RS',L,l_cores[k],z.cores[k]))
                Sds.append(tn.einsum('riS,RS->riR',tmp1-tmp2,R))  
        
    # convert Sds to TT
    grad_cores = _delta2cores(Xspace.cores, R, Sds, Xspace.is_ttm,ortho = [l_cores,r_cores])

    return TT(grad_cores)
