"""
Additional operations.

@author: ion
"""
import torch as tn


def apply_mask(cores, R, indices):
    """
    compute the entries 

    Args:
        cores ([type]): [description]
        R ([type]): [description]
        indices ([type]): [description]
    """
    d = len(cores)
    dt = cores[0].dtype
    M = len(indices)
    # result = tn.zeros((M), dtype = dt)

    # for i in range(M):
    #     tmp = tn.ones(1)
    #     for k in range(d):
    #         tmp = tn.einsum('i,ik->k',tmp,cores[k][:,indices[i][k],:])
    #     result[i] = tn.sum(tmp)

    result = tn.ones((M,1), dtype = dt)
    for i in range(d):
        result = tn.einsum('ij,jik->ik',result,cores[i][:,indices[:,i],:])

    return tn.squeeze(result)


def dense_matvec(cores, other):
    """
    Performs multiplication between a TT-matrix and a full tensor.
    Compatible to tailing dimensions broadcasting.

    Args:
        cores (list[torch.tensor]): the TT-cores of the TT-matrix. The TT-matrix should be of shape (M1 x ... x Md) x (N1 x ... x Nd). 
        other (torch.tensor): The tensor with shape B1 x ... x Bn x N1 x ... x Nd.  

    Returns:
        torch.tensor: The result. Shape is B1 x ... x Bn x M1 x ... x Md.  
    """
    result = tn.unsqueeze(other,-1)

    d = len(cores)
    D = len(other.shape)

    for i in range(d):
        result = tn.tensordot(result,cores[i],([D-d,-1],[2,0]))
    result = tn.squeeze(result,-1)

    return result

def bilinear_form_aux(x_cores, A_cores, y_cores, d):
    """
    Computes the bilinear form xT A y given the TT cores.

    Args:
        x_cores (list[torch.tensor]): the TT cores.
        A_cores (list[torch.tensor]): the TT cores.
        y_cores (list[torch.tensor]): the TT cores.
        d (int): number of modes.

    Returns:
        torch.tensor: the result as 1 element torch.tensor.
    """
    result = tn.ones((1,1,1), dtype = A_cores[0].dtype, device = A_cores[0].device)
    
    for i in range(d):
        result = tn.einsum('lsr,lmL->srmL',result,tn.conj(x_cores[i])) 
        result = tn.einsum('srmL,smnS->LSrn',result,A_cores[i]) 
        result = tn.einsum('LSrn,rnR->LSR',result,y_cores[i]) 
        
    return tn.squeeze(result)


