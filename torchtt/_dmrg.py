"""
DMRG implementation for fast matvec product.
Inspired by TT-Toolbox from MATLAB.

@author: ion
"""
import torchtt
import torch as tn
from torchtt._decomposition import rank_chop, QR, SVD
import datetime
import opt_einsum as oe


try:
    import torchttcpp 
    _flag_use_cpp = True
except:
    import warnings
    warnings.warn("\x1B[33m\nC++ implementation not available. Using pure Python.\n\033[0m")
    _flag_use_cpp = False
    
def dmrg_matvec(A, x, y0 = None,nswp = 20, eps = 1e-12, rmax = 32768, kickrank = 4, verb = False, use_cpp = True):
    """
    Perform fast matrix vector multiplication `y = Ax` in the TT using the DMRG algorithm.
    Uses C++ backend if available.
    
    Args:
        A (TT): TT matrix
        x (TT): TT tensor
        y0 (TT, optional): initial guess of the result (if None is provided a random tensor is generated as a guess). Defaults to None.
        nswp (int, optional): numebr of sweeps. Defaults to 20.
        eps (float, optional): relative accuracy. Defaults to 1e-12.
        rmax (int, optional): maximum rank. Defaults to 32768.
        kickrank (int, optional): kickrank. Defaults to 4.
        verb (bool, optional): show debug info. Defaults to False.
        use_cpp (bool, optional): flag to choose between the python and C++ implementation (if available). Defaults to False.

    Returns:
        TT: the result.
    """
    if _flag_use_cpp and use_cpp:
        return torchtt.TT(torchttcpp.dmrg_mv(A.cores, x.cores, [] if y0 is None else y0.cores, A.M, A.N, x.R, [] if y0 is None else y0.R, nswp, eps, rmax, kickrank, verb))
        #return dmrg_matvec_python(A, x, y0, nswp, eps, rmax, kickrank, verb)
    else:
        return dmrg_matvec_python(A, x, y0, nswp, eps, rmax, kickrank, verb)
    
def dmrg_matvec_python(A, x, y0 = None, nswp = 20, eps = 1e-12, rmax = 32768, kickrank = 4, verb = False):
    """
    Perform fast matrix vector multiplication `y = Ax` in the TT using the DMRG algorithm.

    Args:
        A (TT): TT matrix
        x (TT): TT tensor
        y0 (TT, optional): initial guess of the result (if None is provided a random tensor is generated as a guess). Defaults to None.
        nswp (int, optional): numebr of sweeps. Defaults to 20.
        eps (float, optional): relative accuracy. Defaults to 1e-12.
        rmax (int, optional): maximum rank. Defaults to 32768.
        kickrank (int, optional): kickrank. Defaults to 4.
        verb (bool, optional): show debug info. Defaults to False.

    Returns:
        TT: the result.
    """
    if y0 == None:
        y0 = torchtt.random(A.M,2, dtype=A.cores[0].dtype, device = A.cores[0].device)

    y_cores = y0.cores
    Ry = y0.R.copy()

    d = len(x.N)
    if isinstance(rmax, int):
        rmax = [1] + [rmax]*(d-1) + [1]

    N = x.N
    M = A.M
    r_enlarge = [2]*d

    Phis = [tn.ones((1, 1, 1), dtype=A.cores[0].dtype, device=A.cores[0].device)] + \
        [None]*(d-1) + [tn.ones((1, 1, 1),
                                dtype=A.cores[0].dtype, device=A.cores[0].device)]
    delta_cores = [1.0]*(d-1)
    delta_cores_prev = [1.0]*(d-1)
    last = False

    for i in range(nswp):
        if verb:
            print('sweep ', i)

        # TME = datetime.datetime.now()
        for k in range(d-1, 0, -1):
            core = y_cores[k]

            core = tn.reshape(tn.permute(core,[1,2,0]),[M[k]*Ry[k+1],Ry[k]])

            Q, R = QR(core)
            rnew = min([core.shape[0], core.shape[1]])
            # update current core

            y_cores[k] = (tn.reshape(Q.T,[rnew,M[k],-1]))
            Ry[k] = rnew
            # and the k-1 one
            core_next = tn.reshape(y_cores[k-1],[y_cores[k-1].shape[0]*y_cores[k-1].shape[1],y_cores[k-1].shape[2]]) @ R.T
            y_cores[k-1] = (tn.reshape(core_next,[-1,M[k-1],rnew]))
            
            # update Phi
            Phi = tn.einsum('ijk,mnk->ijmn',Phis[k+1],tn.conj(x.cores[k])) # shape  rk x rAk x rxk-1 x Nk
            Phi = tn.einsum('ijkl,mlnk->ijmn',tn.conj(A.cores[k]),Phi) # shape  rAk-1 x Nk x rk x rxk-1
            Phi = tn.einsum('ijkl,mjk->mil',Phi,y_cores[k]) # shape  rk-1 x rAk-1 x rxk-1
            
            # Phi = tn.einsum('YAX,amnA,ymY,xnX->yax', Phis[k+1], tn.conj(A.cores[k]), y_cores[k], x.cores[k])

            Phis[k] = Phi
        # TME = datetime.datetime.now()-TME
        # print('first ',TME.total_seconds())

        # DMRG
        for k in range(d-1):
              if verb: print('\tcore ',k)
              W_prev = tn.einsum('ijk,klm->ijlm',y_cores[k],y_cores[k+1])
              
              # TME = datetime.datetime.now()
              if not last:
                  # from left
                  W1 = tn.einsum('ijk,klm->ijlm',Phis[k],tn.conj(x.cores[k])) # shape rk-1 x rAk-1 x Nk x rxk
                  W1 = tn.einsum('ijkl,mikn->mjln',tn.conj(A.cores[k]),W1) # shape rk-1 x Mk x rAk x rxk 
                  
                  # from right
                  W2 = tn.einsum('ijk,mnk->njmi',Phis[k+2],tn.conj(x.cores[k+1])) # shape Nk+1 x rAk+1 x rxk x rk+1
                  W2 = tn.einsum('ijkl,klmn->ijmn',tn.conj(A.cores[k+1]),W2) # shape rAk x Mk+1 x rxk x rk+1
                  
                  # new supercore
                  W = tn.einsum('ijkl,kmln->ijmn',W1,W2)
              else:
                  W = tn.conj(W_prev)
                  
              b = tn.linalg.norm(W)
              if b != 0:
                  a = tn.linalg.norm(W-tn.conj(W_prev))
                  delta_cores[k] = (a/b).cpu().numpy()
              else:
                  delta_cores[k] = 0
                
              if delta_cores[k]/delta_cores_prev[k] >= 1 and delta_cores[k]>eps:
                  r_enlarge[k] += 1
                  
              if delta_cores[k]/delta_cores_prev[k] < 0.1 and delta_cores[k]<eps:
                  r_enlarge[k] = max(1,r_enlarge[k]-1)
              
              # SVD 
              U, S, V = SVD(tn.reshape(W,[W.shape[0]*W.shape[1],-1]))
              # new rank is...

              r_new = rank_chop(S.cpu().numpy(),(b.cpu()*eps/(d**(0.5 if last else 1.5))).numpy())
              
              # enlarge ranks
              if not last: r_new += r_enlarge[k]
              
              # ranks must remain valid
              r_new = min([r_new,S.shape[0],rmax[k+1]])
              r_new = max(1,r_new)
              
              # truncate the SVD matrices and spit into 2 cores
              W1 = U[:,:r_new]
              
              W2 = ( V[:r_new,:].T @ tn.diag(S[:r_new]))
              
              
              # TME = datetime.datetime.now()
              if i < nswp-1:
                  # kick-rank
                  W1, Rmat = QR(tn.cat((W1,tn.randn((W1.shape[0],kickrank),dtype=W1.dtype,device=A.cores[0].device)),axis=1))
                  W2 = tn.cat((W2,tn.zeros((W2.shape[0],kickrank),dtype=W2.dtype, device = W2.device)),axis=1)
                  W2 = tn.einsum('ij,kj->ki',W2,Rmat)
                  r_new = W1.shape[1]
              else:
                  W2 = W2.t()       
              # TME = datetime.datetime.now()-TME   
              # print('\t\t ',TME.total_seconds())
              
              # TME = datetime.datetime.now()
              if verb: print('\tcore ',k,': delta ',delta_cores[k],' rank ',Ry[k+1],' ->',r_new)
              Ry[k+1] = r_new 
              # print(k,W1.shape,W2.shape,Ry,N)
              y_cores[k] = tn.conj(tn.reshape(W1,[Ry[k],M[k],r_new]))
              y_cores[k+1] = tn.conj(tn.reshape(W2,[r_new,M[k+1],Ry[k+2]]))
              
              #Wc = tn.einsum('ijk,klm->ijlm', tn.conj(y_cores[k]), tn.conj(y_cores[k+1]))
              
              # print('decomposition ',tn.linalg.norm(Wc-W)/tn.linalg.norm(W))
              Phi_next = tn.einsum('ijk,kmn->ijmn',Phis[k],tn.conj(x.cores[k])) # shape rk-1 x rAk-1 x Nk x rxk
              Phi_next = tn.einsum('ijkl,jmkn->imnl',Phi_next,tn.conj(A.cores[k])) # shape  rk-1 x Mk x rAk x rxk
              Phi_next = tn.einsum('ijm,ijkl->mkl',y_cores[k],Phi_next) # shape rk x rAk x rxk
              
              Phis[k+1] = Phi_next+0
              # TME = datetime.datetime.now()-TME   
              # print('\t\t ',TME.total_seconds())
        
        if last : break
        

        if max(delta_cores) < eps:
            last = True

        delta_cores_prev = delta_cores.copy()

    return torchtt.TT(y_cores)

              

def dmrg_hadamard(x, y, z0 = None, nswp = 20, eps = 1e-12, rmax = 32768, kickrank = 4, verb = False, use_cpp = True):
    """
    Perform fast elementwise multiplication `z = x * y` in the TT using the DMRG algorithm.
    C++ backend not yet ready if available.
    
    Args:
        z (TT): TT tensor
        x (TT): TT tensor
        z0 (TT, optional): initial guess of the result (if None is provided a random tensor is generated as a guess). Defaults to None.
        nswp (int, optional): numebr of sweeps. Defaults to 20.
        eps (float, optional): relative accuracy. Defaults to 1e-12.
        rmax (int, optional): maximum rank. Defaults to 32768.
        kickrank (int, optional): kickrank. Defaults to 4.
        verb (bool, optional): show debug info. Defaults to False.
        use_cpp (bool, optional): flag to choose between the python and C++ implementation (if available). Defaults to False.

    Returns:
        TT: the result.
    """
    if False and _flag_use_cpp and use_cpp:
        return torchtt.TT(torchttcpp.dmrg_mv(A.cores, x.cores, [] if y0 is None else y0.cores, A.M, A.N, x.R, [] if y0 is None else y0.R, nswp, eps, rmax, kickrank, verb))
        #return dmrg_matvec_python(A, x, y0, nswp, eps, rmax, kickrank, verb)
    else:
        return dmrg_hadamard_python(x, y, z0, nswp, eps, rmax, kickrank, verb)
    
def dmrg_hadamard_python(z, x, y0 = None, nswp = 20, eps = 1e-12, rmax = 32768, kickrank = 4, verb = False):
    """
    Perform fast matrix vector multiplication `y = z * x` in the TT using the DMRG algorithm.

    Args:
        z (TT): TT matrix
        x (TT): TT tensor
        y0 (TT, optional): initial guess of the result (if None is provided a random tensor is generated as a guess). Defaults to None.
        nswp (int, optional): numebr of sweeps. Defaults to 20.
        eps (float, optional): relative accuracy. Defaults to 1e-12.
        rmax (int, optional): maximum rank. Defaults to 32768.
        kickrank (int, optional): kickrank. Defaults to 4.
        verb (bool, optional): show debug info. Defaults to False.

    Returns:
        TT: the result.
    """
    if y0 == None:
        y0 = torchtt.random(z.N, 2, dtype = z.cores[0].dtype, device = z.cores[0].device)
    y_cores = y0.cores
    Ry = y0.R.copy()
    
    d = len(x.N)
    if isinstance(rmax,int):
        rmax = [1] + [rmax]*(d-1) + [1]
        
    N = x.N
    M = z.N
    r_enlarge = [2]*d
    
    Phis = [tn.ones((1,1,1), dtype=z.cores[0].dtype, device = z.cores[0].device)] + [None]*(d-1) + [tn.ones((1,1,1),dtype=z.cores[0].dtype, device = z.cores[0].device)]
    delta_cores = [1.0]*(d-1)
    delta_cores_prev = [1.0]*(d-1)
    last = False
    
    for i in range(nswp):
        if verb: print('sweep ',i)
        
        # TME = datetime.datetime.now()
        for k in range(d-1,0,-1):
            core = y_cores[k]
            core = tn.reshape(tn.permute(core,[1,2,0]),[M[k]*Ry[k+1],Ry[k]])
            Q, R = QR(core)
            rnew = min([core.shape[0],core.shape[1]])
            # update current core
            y_cores[k] = (tn.reshape(Q.T,[rnew,M[k],-1]))
            Ry[k] = rnew
            # and the k-1 one
            core_next = tn.reshape(y_cores[k-1],[y_cores[k-1].shape[0]*y_cores[k-1].shape[1],y_cores[k-1].shape[2]]) @ R.T
            y_cores[k-1] = (tn.reshape(core_next,[-1,M[k-1],rnew]))
            
            # update Phi
            Phi = tn.einsum('ijk,mnk->ijmn',Phis[k+1],tn.conj(x.cores[k])) # shape  rk x rAk x rxk-1 x Nk
            Phi = tn.einsum('ikl,mlnk->ikmn',tn.conj(z.cores[k]),Phi) # shape  rAk-1 x Nk x rk x rxk-1
            Phi = tn.einsum('ijkl,mjk->mil',Phi,y_cores[k]) # shape  rk-1 x rAk-1 x rxk-1
            
            # Phi = tn.einsum('YAX,amnA,ymY,xnX->yax', Phis[k+1], tn.conj(A.cores[k]), y_cores[k], x.cores[k])
            Phis[k] = Phi
        # TME = datetime.datetime.now()-TME    
        # print('first ',TME.total_seconds())
        
        # DMRG
        for k in range(d-1):
              if verb: print('\tcore ',k)
              W_prev = tn.einsum('ijk,klm->ijlm',y_cores[k],y_cores[k+1])
              
              # TME = datetime.datetime.now()
              if not last:
                  # from left
                  W1 = tn.einsum('ijk,klm->ijlm',Phis[k],tn.conj(x.cores[k])) # shape rk-1 x rAk-1 x Nk x rxk
                  W1 = tn.einsum('ikl,mikn->mkln',tn.conj(z.cores[k]),W1) # shape rk-1 x Mk x rAk x rxk 
                  
                  # from right
                  W2 = tn.einsum('ijk,mnk->njmi',Phis[k+2],tn.conj(x.cores[k+1])) # shape Nk+1 x rAk+1 x rxk x rk+1
                  W2 = tn.einsum('ikl,klmn->ikmn',tn.conj(z.cores[k+1]),W2) # shape rAk x Mk+1 x rxk x rk+1
                  
                  # new supercore
                  W = tn.einsum('ijkl,kmln->ijmn',W1,W2)
              else:
                  W = tn.conj(W_prev)
                  
              b = tn.linalg.norm(W)
              if b != 0:
                  a = tn.linalg.norm(W-tn.conj(W_prev))
                  delta_cores[k] = (a/b).cpu().numpy()
              else:
                  delta_cores[k] = 0
                
              if delta_cores[k]/delta_cores_prev[k] >= 1 and delta_cores[k]>eps:
                  r_enlarge[k] += 1
                  
              if delta_cores[k]/delta_cores_prev[k] < 0.1 and delta_cores[k]<eps:
                  r_enlarge[k] = max(1,r_enlarge[k]-1)
              
              # SVD 
              U, S, V = SVD(tn.reshape(W,[W.shape[0]*W.shape[1],-1]))
              # new rank is...

              r_new = rank_chop(S.cpu().numpy(),(b.cpu()*eps/(d**(0.5 if last else 1.5))).numpy())
              
              # enlarge ranks
              if not last: r_new += r_enlarge[k]
              
              # ranks must remain valid
              r_new = min([r_new,S.shape[0],rmax[k+1]])
              r_new = max(1,r_new)
              
              # truncate the SVD matrices and spit into 2 cores
              W1 = U[:,:r_new]
              
              W2 = ( V[:r_new,:].T @ tn.diag(S[:r_new]))
              
              
              # TME = datetime.datetime.now()
              if i < nswp-1:
                  # kick-rank
                  W1, Rmat = QR(tn.cat((W1,tn.randn((W1.shape[0],kickrank),dtype=W1.dtype,device=z.cores[0].device)),axis=1))
                  W2 = tn.cat((W2,tn.zeros((W2.shape[0],kickrank),dtype=W2.dtype, device = W2.device)),axis=1)
                  W2 = tn.einsum('ij,kj->ki',W2,Rmat)
                  r_new = W1.shape[1]
              else:
                  W2 = W2.t()       
              # TME = datetime.datetime.now()-TME   
              # print('\t\t ',TME.total_seconds())
              
              # TME = datetime.datetime.now()
              if verb: print('\tcore ',k,': delta ',delta_cores[k],' rank ',Ry[k+1],' ->',r_new)
              Ry[k+1] = r_new 
              # print(k,W1.shape,W2.shape,Ry,N)
              y_cores[k] = tn.conj(tn.reshape(W1,[Ry[k],M[k],r_new]))
              y_cores[k+1] = tn.conj(tn.reshape(W2,[r_new,M[k+1],Ry[k+2]]))
              
              #Wc = tn.einsum('ijk,klm->ijlm', tn.conj(y_cores[k]), tn.conj(y_cores[k+1]))
              
              # print('decomposition ',tn.linalg.norm(Wc-W)/tn.linalg.norm(W))
              Phi_next = tn.einsum('ijk,kmn->ijmn',Phis[k],tn.conj(x.cores[k])) # shape rk-1 x rAk-1 x Nk x rxk
              Phi_next = tn.einsum('ijkl,jkn->iknl',Phi_next,tn.conj(z.cores[k])) # shape  rk-1 x Mk x rAk x rxk
              Phi_next = tn.einsum('ijm,ijkl->mkl',y_cores[k],Phi_next) # shape rk x rAk x rxk
              
              Phis[k+1] = Phi_next+0
              # TME = datetime.datetime.now()-TME   
              # print('\t\t ',TME.total_seconds())
        
        if last : break
        
        if max(delta_cores) < eps:
            last = True

        delta_cores_prev = delta_cores.copy()
        
        
    return torchtt.TT(y_cores)
              

