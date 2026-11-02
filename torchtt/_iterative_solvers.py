"""
Contains iteratiove solvers like GMRES and BiCGSTAB

@author: ion
"""

import torch as tn
import datetime
import numpy as np

def BiCGSTAB(Op, rhs, x0, eps=1e-6, nmax = 40):
    pass

def BiCGSTAB_reset(Op,rhs,x0,eps=1e-6,nmax=40):
    """
    BiCGSTAB solver.
    The inner products conjugate their first argument, so complex systems are supported.
    """ 
    # initial residual
    r = rhs - Op.matvec(x0)
    
    # choose rop
    r0p = tn.rand(r.shape,dtype = x0.dtype, device = x0.device)
    while tn.vdot(r0p.squeeze(),r.squeeze()) == 0:
        r0p = tn.rand(r.shape,dtype = x0.dtype, device = x0.device)
        
    p = r
    x = x0
    
    norm_rhs = tn.linalg.norm(rhs)
    r_nn = tn.linalg.norm(r)
    nit = 0 
    converged = False
    for k in range(nmax):
        nit += 1
        Ap = Op.matvec(p)
        alpha = tn.vdot(r0p.squeeze(),r.squeeze()) / tn.vdot(r0p.squeeze(),Ap.squeeze())
        s = r - alpha * Ap
        if tn.linalg.norm(s)<eps:
            x_n = x+alpha*p
            r_nn = tn.linalg.norm(s)
            converged = True
            break
        
        As = Op.matvec(s)
        omega = tn.vdot(As.squeeze(),s.squeeze()) / tn.vdot(As.squeeze(),As.squeeze())
        
        x_n = x + alpha*p + omega*s
        r_n = s - omega*As
        r_nn = tn.linalg.norm(r_n)
        if r_nn < eps * norm_rhs:
            converged = True
            break
        
        beta = (alpha/omega)*tn.vdot(r0p.squeeze(),r_n.squeeze())/tn.vdot(r0p.squeeze(),r.squeeze())
        p = r_n+beta*(p-omega*Ap)
        
        if abs(tn.vdot(r0p.squeeze(),r_n.squeeze())) < 1e-6:
            r0p = r_n
            p_n = r_n
        # updates
        r = r_n
        x = x_n
        
    flag = converged
    
    relres = r_nn/norm_rhs 
    
    return x_n,flag,nit,relres

    

def gmres_restart(LinOp, b, x0 , N, max_iterations, threshold, resets = 4):
    
    iters = 0
    converged = False
    for r in range(resets):
        x0, flag, it = gmres(LinOp,b,x0, N, max_iterations,threshold)
        iters += it
        if flag:
            converged = True
            break
    return x0, converged, iters
                 

def gmres( LinOp, b, x0, N, max_iterations, threshold):

    converged = False
    
    r = b - LinOp.matvec(x0)
    
    b_norm = tn.linalg.norm(b)
    error = tn.linalg.norm(r) / b_norm

    sn = tn.zeros((max_iterations), dtype = b.dtype, device = b.device)
    cs = tn.zeros((max_iterations), dtype = b.dtype, device = b.device)
    e1 = tn.zeros((max_iterations+1), dtype = b.dtype, device = b.device)
    e1[0] = 1

    err = [error]
    
    r_norm = tn.linalg.norm(r)
    if not r_norm>0:
        return x0, True, 0

    Q = tn.zeros((N,max_iterations+1), dtype = b.dtype, device = b.device) 
    Q[:,0] = r[:,0] / r_norm
    H = tn.zeros((max_iterations+1,max_iterations), dtype = b.dtype, device = b.device)
    
    beta = r_norm * e1
  
    for k in range(max_iterations):
        
        tme = datetime.datetime.now()
        q = LinOp.matvec(Q[:,k])
        tme = datetime.datetime.now() - tme
        
        tme = datetime.datetime.now()
        for i in range(k+1):
            H[i,k] = tn.vdot(Q[:,i],q.squeeze())
            q = q - tn.reshape(H[i,k]*Q[:,i],[-1,1])
        h = tn.linalg.norm(q)
        
        tme = datetime.datetime.now()
        q = q / h
        H[k+1,k] = h
        Q[:,k+1] = q[:,0]
        tme2 = datetime.datetime.now()
        h, c, s = apply_givens_rotation(H[:(k+2),k]+0,cs,sn,k+1)
        tme2 = datetime.datetime.now() - tme2
        H[:(k+2),k] = h
        cs[k] = c
        sn[k] = s
       
        tme = datetime.datetime.now() - tme
        
        beta[k+1] = -sn[k]*beta[k]
        beta[k] = tn.conj(cs[k])*beta[k]
        error = tn.abs(beta[k+1]) / b_norm
        err.append(error)
        
        if error <= threshold:
            converged = True
            break
    y = tn.linalg.solve(H[:k+1,:k+1],tn.reshape(beta[:k+1],[-1,1]))
    x = x0 + Q[:,:k+1] @ y     
    return x, converged, k
    

  

def apply_givens_rotation(h, cs, sn, k):
    # The rotation with c and s is [[conj(c), conj(s)], [-s, c]]. It is unitary for complex c and s.
    dev = h.device
    h = h.cpu().numpy()
    cs = cs.cpu().numpy()
    sn = sn.cpu().numpy()
    for i in range(k-1):
        temp   =  np.conj(cs[i])* h[i] + np.conj(sn[i]) * h[i+1]
        h[i+1] = -sn[i] * h[i] + cs[i] * h[i+1]
        h[i]   = temp
  
    cs_k, sn_k = givens_rotation(h[k-1], h[k])

 
    h[k-1] = np.conj(cs_k) * h[k-1] + np.conj(sn_k) * h[k]
    h[k] = 0.0
    return tn.tensor(h).to(dev), tn.tensor(cs_k).to(dev), tn.tensor(sn_k).to(dev)

def givens_rotation(v1,v2):
   
    den = np.sqrt(np.abs(v1)**2+np.abs(v2)**2)
    return v1/den, v2/den
