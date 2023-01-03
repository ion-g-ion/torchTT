"""
Basic decomposition and orthogonalization.

@author: ion
"""

import torch as tn
import numpy as np


        
def QR(mat):
    """
    Compute the QR decomposition. Backend can be changed.

    Parameters
    ----------
    mat : tn array
        DESCRIPTION.

    Returns
    -------
    Q : the Q matrix
    R : the R matrix

    """
    Q,R = tn.linalg.qr(mat)
    return Q, R
    
def SVD(mat):
    """
    Computes the SVD of a matrix

    Args:
        mat (torch.tensor): the matrix

    Returns:
        U, S, V: the SVD factors.
    """
    if mat.shape[0] < 10*mat.shape[1]:
        try:
            u, s, v = tn.linalg.svd(mat,full_matrices=False)
            s = s.to(v.dtype)
            return u, s, v
        except:
            u, s, v = np.linalg.svd(mat.numpy(),full_matrices=False)
            return tn.tensor(u, dtype = mat.dtype, device = mat.device), tn.tensor(s, dtype = mat.dtype, device = mat.device), tn.tensor(v, dtype = mat.dtype, device = mat.device)
    else:
        try:    
            u, s, v = tn.linalg.svd(mat.t(),full_matrices=False)
            s = s.to(v.dtype)
            return  v.t(), s, u.t()
        except:
            u, s, v = np.linalg.svd((mat.t()).numpy(),full_matrices=False)
            return  tn.tensor(v.t(), dtype = mat.dtype, device = mat.device), tn.tensor(s, dtype = mat.dtype, device = mat.device), tn.tensor(u.t(), dtype = mat.dtype, device = mat.device)
    # u, s, v = tn.linalg.svd(mat,full_matrices=False)
    # return u, s, v




def lr_orthogonal(tt_cores, R, is_ttm, no_gpu = False):
    """
    Orthogonalize the TT-cores left to right.

    Parameters
    ----------
    tt_cores : list of torch tensors.
        The TT-cores as a list.

    Returns
    -------
    tt_cores : list of torch tensors.
        The orthogonal TT-cores as a list.

    """  
    
    d = len(tt_cores)

    rank_next = R[0]
    
    core_now = tt_cores[0]
    cores_new = d*[None]
    for i in range(d-1):
        
               
        if is_ttm:
            mode_shape = [core_now.shape[1],core_now.shape[2]]
            core_now = tn.reshape(core_now,[core_now.shape[0]*core_now.shape[1]*core_now.shape[2],-1])
        else:
            mode_shape = [core_now.shape[1]]
            core_now = tn.reshape(core_now,[core_now.shape[0]*core_now.shape[1],-1])
            
        # perform QR
        Qmat, Rmat = QR(core_now)
        core_now= Qmat
             
        # take next core
        core_next = tt_cores[i+1]
        shape_next = list(core_next.shape[1:])
        core_next = tn.reshape(core_next,[core_next.shape[0],-1])
        core_next = Rmat @ core_next
        core_next = tn.reshape(core_next,[core_now.shape[1]]+shape_next)
        
        # update the cores
        cores_new[i] = tn.reshape(core_now,[R[i]]+mode_shape+[-1])
        R[i+1] = core_now.shape[1]
        cores_new[i+1] = core_next
        
        core_now = core_next
        
        
    return cores_new, R

def rl_orthogonal(tt_cores, R, is_ttm, no_gpu = False):
    """
    Orthogonalize the TT-cores right to left.

    Parameters
    ----------
    tt_cores : list of torch tensors.
        The TT-cores as a list.

    Returns
    -------
    tt_cores : list of torch tensors.
        The orthogonal TT-cores as a list.

    """  
    
    d = len(tt_cores)
        
    
    
    cores_new = d*[None]
    cores_new[-1] = tt_cores[-1]+0
    for i in range(d-1,0,-1):

        if is_ttm:
            mode_shape = [cores_new[i].shape[1],cores_new[i].shape[2]]
            core_now = tn.reshape(cores_new[i],[cores_new[i].shape[0],cores_new[i].shape[2]*cores_new[i].shape[3]*cores_new[i].shape[1]]).t()
        else:
            mode_shape = [cores_new[i].shape[1]]
            core_now = tn.reshape(cores_new[i],[cores_new[i].shape[0],cores_new[i].shape[1]*cores_new[i].shape[2]]).t()
        
        # perform QR
        
        Qmat, Rmat = QR(core_now)
            # print('QR ',list(Qmat.shape),list(Rmat.shape))
        rnew = min([core_now.shape[0],core_now.shape[1]])
        rnew = Rmat.shape[0]
        # update current core
        cores_new[i] = tn.reshape(Qmat.T,[rnew]+mode_shape+[-1])
        # print('R ',tt_cores[i].shape,cores_new[i].shape,tt_cores[i-1].shape)
        R[i] = cores_new[i].shape[0]
        # and the k-1 one
        if is_ttm:
            mode_shape = [tt_cores[i-1].shape[1],tt_cores[i-1].shape[2]]
            core_next = tn.reshape(tt_cores[i-1],[tt_cores[i-1].shape[0]*tt_cores[i-1].shape[1]*tt_cores[i-1].shape[2],tt_cores[i-1].shape[3]]) @ Rmat.T
        else:
            mode_shape = [tt_cores[i-1].shape[1]]
            core_next = tn.reshape(tt_cores[i-1],[tt_cores[i-1].shape[0]*tt_cores[i-1].shape[1],tt_cores[i-1].shape[2]]) @ Rmat.T
        cores_new[i-1] = tn.reshape(core_next,[tt_cores[i-1].shape[0]]+mode_shape+[-1])
        
    return cores_new, R

    

def round_tt(tt_cores,R,eps,Rmax,is_ttm=False):
    """
    Rounds a TT-tensor (tt_cores have to be orthogonal)

    Parameters
    ----------
    tt_cores : list of torch tensors.
        Orthogonal TT cores.
    R : list of integers of length d+1.
        ranks of the TT-decomposition.
    eps : double.
        desired rounding accuracy.
    Rmax : list of integers
        the maximum rank that is allowed.

    Returns
    -------
    tt_cores : list of torch tensors.
        The TT-cores of the rounded tensor.
    R : list of inteders of length d+1.
        rounded ranks.

    """
    d = len(tt_cores)
    if d == 1:
        tt_cores = [tt_cores[0].clone()]
        return tt_cores, R
    tt_cores, R = lr_orthogonal(tt_cores, R, is_ttm)
    core_now = tt_cores[-1]
    eps = eps / np.sqrt(d-1) 

    
    for i in range(d-1,0,-1):
        core_next = tt_cores[i-1]
        
        core_now = tn.reshape(core_now,[R[i],-1])
        core_next = tn.reshape(core_next,[-1,R[i]])
        
        
        U, S, V = SVD(core_now)
        if S.is_cuda:
            r_now = min([Rmax[i],rank_chop(S.cpu().numpy(),tn.linalg.norm(S).cpu().numpy()*eps)])
        else:
            r_now = min([Rmax[i],rank_chop(S.numpy(),tn.linalg.norm(S).numpy()*eps)])
    
        U = U[:,:r_now]
        S = S[:r_now]
        V = V[:r_now,:]
        
        U = U @ tn.diag(S)
        R[i] = r_now
        core_next = core_next @ U
        core_now = V
        
        tt_cores[i] = tn.reshape(core_now,[R[i]]+list(tt_cores[i].shape[1:-1])+[R[i+1]])
        tt_cores[i-1] = tn.reshape(core_next,[R[i-1]]+list(tt_cores[i-1].shape[1:-1])+[R[i]])
        
        core_now = core_next
    
    return tt_cores, R

    

def mat_to_tt(A,M,N,eps,rmax = 1000,is_sparse=False):
    """
    Computes the TT-matrix decomposition of A. A has the shape M x N, where M, N are of length d.
    The eps and rmax are given.

    Parameters
    ----------
    A : torch tensor
        the array.
    M : list of integers
        shape.
    N : list.of integers.
        shape.
    eps : float
        desired accuracy.
    rmax : int, optional
        Masixum rank. The default is 100.
    is_sparse : bool, optional
        is A in sparse foramt. The default is False.

    Returns
    -------
    cores : list of 4d cores
        the cores of the TT-matrix decomposition.
    R : list of integers
        ranks.

    """
    d = len(M)
    if len(M)!=len(N):
        raise('Dimension mismatch')
        return
    
    if is_sparse:
        pass
    else:
        
        A = tn.reshape(A, M+N)

        permute = tuple( np.arange(2*d).reshape([2,d]).transpose().flatten() )
        A = tn.permute(A,permute)

        A = tn.reshape(A, [i[0]*i[1] for i in zip(M,N)])
        
        ttv, R = to_tt(A,eps=eps,rmax=rmax)
    
    cores= []
    # cores have to be in the TT-matrix format ( rIr' -> rijr')    
    for i in range(d):
        tmp = tn.permute(ttv[i],  [1,0,2])
        tmp = tn.reshape(tmp,[M[i],N[i],tmp.shape[1],tmp.shape[2]])
        tmp = tn.permute(tmp, [2,0,1,3])
        cores.append(tmp)


    return cores, R

def rank_chop(s,eps):
    """
    Chop the rank.

    Parameters
    ----------
    s : numpy vector
        Vector of singular values.
    eps : double
        Desired accuracy.

    Returns
    -------
    R : int
        Rank.
    """
    if np.linalg.norm(s) == 0.0:
        return 1
    
    if eps <= 0.0:
        return s.size
    
    R = s.size - 1
   
    sc = np.cumsum(np.abs(s[::-1])**2)[::-1]
    R = np.argmax(sc<eps**2)
   #  print(sc,eps**2,sc<eps**2,R)
   #  while R>0:
   #      if np.sum(s[R:]**2) >= eps**2:
   #          break;
   #      R -= 1
        
    R = R if R>0 else 1
    R = s.size if sc[-1]>eps**2 else R

    return R
    
def to_tt(A,N=None,eps=1e-14,rmax=100,is_sparse=False):
    """
    Computes the TT cores of a full tensor A given the tolerance eps and the maximum rank.
    The TT-cores are returned as a list.
    
    Parameters
    ----------
    A : torch tensor
        Tensor to decompose.
    N : vector of integers, optional
        DESCRIPTION. The default is None.
    eps : double, optional
        DESCRIPTION. The default is 1e-14.
    rmax : int or list of integers, optional
        maximum rand either as scalar or list. The default is 100.
   is_sparse : boolean, optional
        Is True if the tensor is of type sparse type. The default is False.

    Returns
    -------
    cores : list of torch tensors.
        The TT-cores of the decomposition.
    r : list of integers.
        The TT-ranks.

    """
    
    if N == None:
        N = list(A.shape)
      
    d = len(N)
    r = [1]*(d+1)
    
    # check if rmax is a list
    if not isinstance(rmax,list):
        rmax = [1] + (d-1)*[rmax] + [1]
        
    C = A   
    cores = [] 
    ep = eps/np.sqrt(d-1)
    
   
    for i in range(d-1):
        
        m = N[i]*r[i]
        
        # reshape C to a matrix 
        C = tn.reshape(C, [m,-1])
        
        # tme = datetime.datetime.now()
        # perform svd 
        
        u, s, v = SVD(C)
        
        # tme = datetime.datetime.now()-tme
        # print('time1',tme)
      
        # tme = datetime.datetime.now()
        # choose the rank according to eps tolerance
        r1 = rank_chop(s.cpu().numpy(), ep*tn.linalg.norm(s).cpu().numpy())
        r1 = min([r1,rmax[i+1]])
        
        u = u[:,:r1]
        s = s[:r1]
        r[i+1] = r1
        
        # reshape and append the core
        cores.append(tn.reshape(u,[r[i],N[i],r1]))
        
        # truncate the right singular vector
        v = v[:r1,:]
        
        # update the core    
        v = tn.diag(s) @ v

        C = v
        # tme = datetime.datetime.now()-tme
        # print('time2',tme)
    cores.append(tn.reshape(C,[r[-2],N[-1],-1]))
    return cores, r
    
