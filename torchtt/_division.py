"""
Elementwise division using AMEN

@author: ion
"""
import torch as tn
import numpy as np
import datetime
from torchtt._decomposition import QR, SVD, rl_orthogonal, lr_orthogonal
from torchtt._iterative_solvers import BiCGSTAB_reset, gmres_restart
import opt_einsum as oe

def local_product(Phi_right, Phi_left, coreA, core, shape):
    """
    Compute local matvec product

    Args:
        Phi (torch.tensor): right tensor of shape r x R x r.
        Psi (torch.tensor): left tensor of shape lp x Rp x lp.
        coreA (torch.tensor): current core of A, shape is rp x N x r.
        x (torch.tensor): the current core of x, shape is rp x N x r.
        shape (torch.Size): the shape of x. 

    Returns:
        torch.tensor: the reuslt.
    """
    w = oe.contract('lsr,smS,LSR,rmR->lmL',Phi_left,coreA,Phi_right,core)
    return w

class LinearOp():
    def __init__(self,Phi_left,Phi_right,coreA,shape,prec):
        self.Phi_left = Phi_left
        self.Phi_right = Phi_right
        self.coreA = coreA
        self.shape = shape
        self.prec = prec
        # self.contraction = oe.contract_expression('lsr,smS,LSR,rmR->lmL', Phi_left.shape, coreA.shape, Phi_right.shape, shape)
        if prec == 'c':
            Jl = tn.einsum('sd,smS->dmS',tn.diagonal(Phi_left,0,0,2),coreA)
            Jr = tn.diagonal(Phi_right,0,0,2)
            J = tn.einsum('dmS,SD->dmD',Jl,Jr)
            self.J = 1/J
            
    def apply_prec(self,x):
        
        if self.prec == 'c':
            y = x * self.J # no improvement using opt_einsum
            return y
        
    def matvec(self, x, apply_prec = True):
        if self.prec == None or not apply_prec:
            x = tn.reshape(x,self.shape)
            # tme = datetime.datetime.now()
            # w = oe.contract('lsr,smS,LSR,rmR->lmL',self.Phi_left,self.coreA,self.Phi_right,x)
            
            w1 = tn.tensordot(self.coreA,self.Phi_left,([0],[1])) # smS,lsr->mSlr
            w2 = tn.tensordot(x,self.Phi_right,([2],[2])) # rmR,LSR->rmLS
            w = tn.einsum('rmLS,mSlr->lmL',w2,w1) # rmLS,mSlr->lmL 
            
            
            # w = self.contraction(self.Phi_left,self.coreA,self.Phi_right,x)
            
        elif self.prec == 'c':
            x = tn.reshape(x,self.shape)
            x = self.apply_prec(x)
            # w = self.contraction(self.Phi_left,self.coreA,self.Phi_right,x)
            w1 = tn.tensordot(self.coreA,self.Phi_left,([0],[1])) # smS,lsr->mSlr
            w2 = tn.tensordot(x,self.Phi_right,([2],[2])) # rmR,LSR->rmLS
            w = tn.einsum('rmLS,mSlr->lmL',w2,w1) # rmLS,mSlr->lmL 
            # w = oe.contract('lsr,smS,LSR,rmR->lmL',self.Phi_left,self.coreA,self.Phi_right,x)
            
        else:
            raise Exception('Preconditioner '+str(self.prec)+' not defined.')
        return tn.reshape(w,[-1,1])

def amen_divide(a, b, nswp = 22, x0 = None, eps = 1e-10,rmax = 100, max_full = 500, kickrank = 4, kick2 = 0, trunc_norm = 'res', local_iterations = 40, resets = 2, verbose = True, preconditioner = None):
    
    

    if verbose: time_total = datetime.datetime.now()
    
    dtype = a.cores[0].dtype 
    device = a.cores[0].device
    rank_search = 1 # binary rank search
    damp = 2

    rA = a.R
    N = b.N
    d = len(N)
    
    if x0 == None:
        rx = [1] + (d-1)*[2] + [1]
        x_cores = [ tn.ones([rx[k],N[k],rx[k+1]], dtype = dtype, device = device) for k in range(d)]
    else:
        x = x0
        x_cores = x.cores.copy()
        rx = x.R.copy()
        
    # check if rmax is a list
    if isinstance(rmax, int):
        rmax = [1] + (d-1) * [rmax] + [1]

    # z cores
    rz = [1]+(d-1)*[kickrank+kick2]+[1]
    z_cores = [ tn.randn([rz[k],N[k],rz[k+1]], dtype = dtype, device = device) for k in range(d)]
    z_cores, rz = rl_orthogonal(z_cores, rz, False)
    
    norms = np.zeros(d)
    Phiz = [tn.ones((1,1,1), dtype = dtype, device = device)] + [None] * (d-1) + [tn.ones((1,1,1), dtype = dtype, device = device)] # size is rzk x Rk x rxk
    Phiz_b = [tn.ones((1,1), dtype = dtype, device = device)] + [None] * (d-1) + [tn.ones((1,1), dtype = dtype, device = device)]   # size is rzk x rzbk
    
    Phis = [tn.ones((1,1,1), dtype = dtype, device = device)] + [None] * (d-1) + [tn.ones((1,1,1), dtype = dtype, device = device)] # size is rk x Rk x rk
    Phis_b = [tn.ones((1,1), dtype = dtype, device = device)] + [None] * (d-1) + [tn.ones((1,1), dtype = dtype, device = device)] # size is rk x rbk

    last = False

    normA = np.ones((d-1))
    normb = np.ones((d-1))
    normx = np.ones((d-1))
    nrmsc = 1.0

    for swp in range(nswp):
        tme_sweep = datetime.datetime.now()
        # right to left orthogonalization

        if verbose:
            print()
            print('Starting sweep %d %s...'%(swp,"(last one) " if last else ""))
            tme_sweep = datetime.datetime.now() 
        

        tme = datetime.datetime.now()
        for k in range(d-1,0,-1):
            
            # update the z part (ALS) update
            if not last:
                if swp > 0:
                    czA = local_product(Phiz[k+1],Phiz[k],a.cores[k],x_cores[k],x_cores[k].shape) # shape rzp x N x rz
                    czy = tn.einsum('br,bnB,BR->rnR',Phiz_b[k],b.cores[k],Phiz_b[k+1]) # shape is rzp x N x rz
                    cz_new = czy*nrmsc - czA
                    _,_,vz = SVD(tn.reshape(cz_new,[cz_new.shape[0],-1]))
                    cz_new = vz[:min(kickrank,vz.shape[0]),:].t() # truncate to kickrank
                    if k < d-1: # extend cz_new with random elements
                        cz_new = tn.cat((cz_new,tn.randn((cz_new.shape[0],kick2),  dtype = dtype, device = device)),1)
                else:
                    cz_new = tn.reshape(z_cores[k],[rz[k],-1]).t()

                qz, _ = QR(cz_new)
                rz[k] = qz.shape[1]
                z_cores[k] = tn.reshape(qz.t(),[rz[k],N[k],rz[k+1]]) 
            
            # norm correction ?
            if swp > 0: nrmsc = nrmsc * normA[k-1] * normx[k-1] / normb[k-1] 
        
            
            core = tn.reshape(x_cores[k],[rx[k],N[k]*rx[k+1]]).t()
            Qmat, Rmat = QR(core)
            
            core_prev = tn.einsum('ijk,km->ijm',x_cores[k-1],Rmat.T)
            rx[k] = Qmat.shape[1]
            
            current_norm = tn.linalg.norm(core_prev)
            if current_norm>0:
                core_prev /= current_norm
            else:
                current_norm = 1.0
            normx[k-1] = normx[k-1]*current_norm
             
            x_cores[k] = tn.reshape(Qmat.t(),[rx[k],N[k],rx[k+1]]) 
            x_cores[k-1] = core_prev[:]
            
            # update phis (einsum)
            # print(x_cores[k].shape,A.cores[k].shape,x_cores[k].shape)
            Phis[k] = compute_phi_bck_A(Phis[k+1],x_cores[k],a.cores[k],x_cores[k])
            Phis_b[k] = compute_phi_bck_rhs(Phis_b[k+1],b.cores[k],x_cores[k])
            
            # ... and norms 
            norm = tn.linalg.norm(Phis[k])
            norm = norm if norm>0 else 1.0
            normA[k-1] = norm 
            Phis[k] /= norm
            norm = tn.linalg.norm(Phis_b[k])
            norm = norm if norm>0 else 1.0
            normb[k-1] = norm 
            Phis_b[k] /= norm
            
            # norm correction
            nrmsc = nrmsc * normb[k-1]/ (normA[k-1] * normx[k-1])      

            # compute phis_z
            if not last:
                Phiz[k] = compute_phi_bck_A(Phiz[k+1], z_cores[k], a.cores[k], x_cores[k]) / normA[k-1]
                Phiz_b[k] = compute_phi_bck_rhs(Phiz_b[k+1], b.cores[k], z_cores[k]) / normb[k-1]


        # start loop
        max_res = 0
        max_dx = 0

        for k in range(d):
            if verbose: print('\tCore',k) 
            previous_solution = tn.reshape(x_cores[k],[-1,1])
            
        
            # assemble rhs 
            rhs = tn.einsum('br,bmB,BR->rmR',Phis_b[k] , b.cores[k] * nrmsc, Phis_b[k+1])
            rhs = tn.reshape(rhs,[-1,1])
            norm_rhs = tn.linalg.norm(rhs)
            
            #residuals
            real_tol = (eps/np.sqrt(d))/damp
        
            # solve the local system
            use_full = rx[k]*N[k]*rx[k+1] < max_full
            if use_full: 
                # solve the full system
                if verbose: print('\t\tChoosing direct solver (local size %d)....'%(rx[k]*N[k]*rx[k+1]))  
                Bp = tn.einsum('smS,LSR->smRL',a.cores[k],Phis[k+1]) # shape is Rp x N x N x r x r
                #B = tn.einsum('lsr,smnRL->rmRlnL',Phis[k],Bp)
                B = oe.contract('lsr,smRL,mn->lmLrnR',Phis[k],Bp,tn.eye(N[k],dtype=dtype,device=device)) 
                B = tn.reshape(B,[rx[k]*N[k]*rx[k+1],rx[k]*N[k]*rx[k+1]])

                solution_now = tn.linalg.solve(B,rhs)   
                
                res_old = tn.linalg.norm(B@previous_solution-rhs)/norm_rhs
                res_new = tn.linalg.norm(B@solution_now-rhs)/norm_rhs
            else:
                # iterative solver
                if verbose: 
                    print('\t\tChoosing iterative solver (local size %d)....'%(rx[k]*N[k]*rx[k+1])) 
                    time_local = datetime.datetime.now()
                shape_now = [rx[k],N[k],rx[k+1]]
                Op = LinearOp(Phis[k],Phis[k+1],a.cores[k],shape_now, preconditioner)
                
                # solution_now, flag, nit, res_new = BiCGSTAB_reset(Op, rhs,previous_solution[:], eps_local, local_iterations) 
                eps_local = real_tol * norm_rhs
                drhs = Op.matvec(previous_solution, False)
                drhs = rhs-drhs
                eps_local = eps_local / tn.linalg.norm(drhs) 
                solution_now, flag, nit = gmres_restart(Op, drhs, previous_solution*0, rhs.shape[0], local_iterations+1, eps_local, resets)
                if preconditioner != None:
                    solution_now = Op.apply_prec(tn.reshape(solution_now,shape_now))
                    solution_now = tn.reshape(solution_now,[-1,1])
                
                solution_now = previous_solution + solution_now
                res_old = tn.linalg.norm(Op.matvec(previous_solution, False)-rhs)/norm_rhs
                res_new = tn.linalg.norm(Op.matvec(solution_now, False)-rhs)/norm_rhs
                if verbose:
                    print('\t\tFinished with flag %d after %d iterations with relres %g (from %g)'%(flag,nit,res_new,eps_local)) 
                    time_local = datetime.datetime.now() - time_local
                    print('\t\tTime needed ',time_local)
            # residual damp check
            if res_old/res_new < damp and res_new > real_tol:
                if verbose: print('WARNING: residual increases. res_old %g, res_new %g, real_tol %g'%(res_old,res_new,real_tol)) # warning (from tt toolbox)

            # compute residual and step size
            dx = tn.linalg.norm(solution_now-previous_solution)/tn.linalg.norm(solution_now)
            if verbose: 
                print('\t\tdx = %g, res_now = %g, res_old = %g'%(dx,res_new,res_old))
                

            max_dx = max(dx,max_dx)
            max_res = max(max_res,res_old)

            solution_now = tn.reshape(solution_now,[rx[k]*N[k],rx[k+1]])
            # truncation
            if k<d-1:
                u, s, v = SVD(solution_now)
                # print('\t\tTruncation of solution of shape',[rx[k]*N[k],rx[k+1]],' into u', u.shape, ' and v ',v.shape)
                if trunc_norm == 'fro':
                    pass
                else:
                    # search for a rank such that offeres small enough residuum
                    # TODO: binary search?
                    r = 0
                    for r in range(u.shape[1]-1,0,-1):
                        solution = u[:,:r] @ tn.diag(s[:r]) @ v[:r,:] # solution has the same size
                        # res = tn.linalg.norm(tn.reshape(local_product(Phis[k+1],Phis[k],a.cores[k],tn.reshape(solution,[rx[k],N[k],rx[k+1]]),solution_now.shape),[-1,1]) - rhs)/norm_rhs
                        if use_full:
                            res = tn.linalg.norm(B@tn.reshape(solution,[-1,1])-rhs)/norm_rhs
                        else:
                            # res = tn.linalg.norm(tn.reshape(local_product(Phis[k+1],Phis[k],a.cores[k],tn.reshape(solution,[rx[k],N[k],rx[k+1]]),solution_now.shape),[-1,1]) - rhs)/norm_rhs
                            res = tn.linalg.norm(Op.matvec(solution)-rhs)/norm_rhs
                        if res > max(real_tol*damp,res_new):
                            break
                    r += 1

                    r = min([r,tn.numel(s),rmax[k+1]])
            else:
                u, v = QR(solution_now)
                # v = v.t()
                r = u.shape[1]
                s = tn.ones(r,  dtype = dtype, device = device)

            u = u[:,:r]
            v = tn.diag(s[:r]) @ v[:r,:]
            v = v.t()

            if not last:
                czA = local_product(Phiz[k+1], Phiz[k], a.cores[k], tn.reshape(u@v.t(),[rx[k],N[k],rx[k+1]]), [rx[k],N[k],rx[k+1]]) # shape rzp x N x rz
                czy = tn.einsum('br,bnB,BR->rnR',Phiz_b[k],b.cores[k]*nrmsc,Phiz_b[k+1]) # shape is rzp x N x rz
                cz_new = czy - czA
                # print('Phiz_b',[plm.shape for plm in Phiz_b])
                # print('czA',czA.shape,' czy',czy.shape)
                # print('rz',rz)
                # print('rx',rx)

                uz,_,_ = SVD(tn.reshape(cz_new, [rz[k]*N[k],rz[k+1]]))
                cz_new = uz[:,:min(kickrank,uz.shape[1])] # truncate to kickrank
                if k < d-1: # extend cz_new with random elements
                    cz_new = tn.cat((cz_new,tn.randn((cz_new.shape[0],kick2),  dtype = dtype, device = device)),1)
                
                qz,_ = QR(cz_new)
                rz[k+1] = qz.shape[1]
                z_cores[k] = tn.reshape(qz,[rz[k],N[k],rz[k+1]])

            if k < d-1:
                if not last:
                    left_res = local_product(Phiz[k+1],Phis[k],a.cores[k],tn.reshape(u@v.t(),[rx[k],N[k],rx[k+1]]),[rx[k],N[k],rx[k+1]])
                    left_b = tn.einsum('br,bmB,BR->rmR',Phis_b[k],b.cores[k]*nrmsc,Phiz_b[k+1])
                    uk = left_b - left_res # rx_k x N_k x rz_k+1
                    u, Rmat = QR(tn.cat((u,tn.reshape(uk,[u.shape[0],-1])),1))
                    r_add = uk.shape[2]
                    v = tn.cat((v,tn.zeros([rx[k+1],r_add],  dtype = dtype, device = device)), 1)
                    v = v @ Rmat.t()
                 
                r = u.shape[1]
                # print(u.shape,v.shape,x_cores[k+1].shape)
                v = tn.einsum('ji,jkl->ikl',v,x_cores[k+1])
                # remove norm correction
                nrmsc = nrmsc * normA[k] * normx[k] / normb[k]  

                norm_now = tn.linalg.norm(v)

                if norm_now>0:
                    v = v / norm_now
                else:
                    norm_now = 1.0
                normx[k] *= norm_now

                x_cores[k] = tn.reshape(u, [rx[k],N[k],r])
                x_cores[k+1] = tn.reshape(v, [r,N[k+1],rx[k+2]])
                rx[k+1] = r

                # next phis with norm correction
                Phis[k+1] = compute_phi_fwd_A(Phis[k], x_cores[k], a.cores[k], x_cores[k]) 
                Phis_b[k+1] = compute_phi_fwd_rhs(Phis_b[k], b.cores[k],x_cores[k])
                
                # ... and norms 
                norm = tn.linalg.norm(Phis[k+1])
                norm = norm if norm>0 else 1.0
                normA[k] = norm 
                Phis[k+1] /= norm
                norm = tn.linalg.norm(Phis_b[k+1])
                norm = norm if norm>0 else 1.0
                normb[k] = norm 
                Phis_b[k+1] /= norm
                
                # norm correction
                nrmsc = nrmsc * normb[k] / ( normA[k] * normx[k] )


                # next phiz
                if not last:
                    Phiz[k+1] = compute_phi_fwd_A(Phiz[k], z_cores[k], a.cores[k], x_cores[k]) / normA[k]
                    Phiz_b[k+1] = compute_phi_fwd_rhs(Phiz_b[k], b.cores[k],z_cores[k]) / normb[k]
            else:
                x_cores[k] = tn.reshape(u@tn.diag(s[:r]) @ v[:r,:].t(),[rx[k],N[k],rx[k+1]])

        if verbose:
            print('Solution rank is',rx)
            print('Maxres ',max_res)
            tme_sweep = datetime.datetime.now()-tme_sweep
            print('Time ',tme_sweep)
              
                
        if last:
            break

        if max_res < eps:
            last = True

    if verbose:
        time_total = datetime.datetime.now() - time_total
        print()
        print('Finished after' ,swp,' sweeps and ',time_total)
        print()
    normx = np.exp(np.sum(np.log(normx))/d)

    for k in range(d):
        x_cores[k] *= normx

    

    return x_cores



def compute_phi_bck_A(Phi_now,core_left,core_A,core_right):
    """
    Compute the phi backwards for the form dot(left,A @ right)

    Args:
        Phi_now (torch.tensor): The current phi. Has shape r1_k+1 x R_k+1 x r2_k+1
        core_left (torch.tensor): the core on the left. Has shape r1_k x N_k x r1_k+1 
        core_A (torch.tensor): the core of the matrix. Has shape  R_k x N_k x N_k x R_k
        core_right (torch.tensor): the core to the right. Has shape r2_k x N_k x r2_k+1 

    Returns:
        torch.tensor: The following phi (backward). Has shape r1_k x R_k x r2_k
    """
    
    # Phip = tn.einsum('ijk,klm->ijlm',core_right,Phi_now)
    # Phipp = tn.einsum('ijkl,abjk->ilba',Phip,core_A)
    # Phi = tn.einsum('ijkl,akj->ila',Phipp,core_left)
    Phi = oe.contract('LSR,lML,sMS,rMR->lsr',Phi_now,core_left,core_A,core_right)
    return Phi

def compute_phi_fwd_A(Phi_now, core_left, core_A, core_right):
    """
    Compute the phi forward for the form dot(left,A @ right)

    Args:
        Phi_now (torch.tensor): The current phi. Has shape r1_k x R_k x r2_k
        core_left (torch.tensor): the core on the left. Has shape r1_k x N_k x r1_k+1 
        core_A (torch.tensor): the core of the matrix. Has shape  R_k x N_k x N_k x R_k
        core_right (torch.tensor): the core to the right. Has shape r2_k x N_k x r2_k+1 

    Returns:
        torch.tensor: The following phi (backward). Has shape r1_k+1 x R_k+1 x r2_k+1
    """
    # Psip = tn.einsum('ijk,kbc->ijbc', Phi_now, core_left)  # shape is rk-1 x Rk-1 x Nk x rk 
    # Psipp = tn.einsum('ijkl,aijd->klad', core_A, Psip)  # shape is nk x Rk x rk-1 x rk
    # Phi_next= tn.einsum('ijk,jbid->kbd',core_right,Psipp) # shape is rk x  Rk x rk
    # tme1 = datetime.datetime.now()
   #  Phi_next = tn.einsum('lsr,lML,sMNS,rNR->LSR',Phi_now,core_left,core_A,core_right)
    # tme1 = datetime.datetime.now() - tme1 
    # tme2 = datetime.datetime.now()
    Phi_next = oe.contract('lsr,lML,sMS,rMR->LSR',Phi_now,core_left,core_A,core_right)
    # tme2 = datetime.datetime.now() - tme2 
    # print('\n>>>>>>>>>>>>>>>>>>>>>>>>>>Time1 ',tme1,' time 2', tme2) 
    return Phi_next

def compute_phi_bck_rhs(Phi_now,core_b,core):
    """
    

    Args:
        Phi_now (torch.tensor): The current phi. Has shape rb_k+1 x r_k+1
        core_b (torch.tensor): The current core of the rhs. Has shape rb_k x N_k x rb_k+1
        core (torch.tensor): The current core. Has shape r_k x N_k x r_k+1

    Returns:
        torch.tensor: The backward phi corresponding to the rhs. Has shape rb_k x r_k
    """
    #Phit = tn.einsum('ij,abj->iba',Phi_now,core_b)
    #Phi = tn.einsum('ijk,kjc->ic',core,Phit)
    Phi = oe.contract('BR,bnB,rnR->br',Phi_now,core_b,core)
    return Phi

def compute_phi_fwd_rhs(Phi_now,core_rhs,core):
    """
    

    Args:
        Phi_now (torch.tensor): The current phi. Has shape  rb_k x r_k
        core_b (torch.tensor): The current core of the rhs. Has shape rb_k x N_k+1 x rb_k+1
        core (torch.tensor): The current core. Has shape r_k x N_k x r_k+1

    Returns:
        torch.tensor: The forward computer phi for the rhs. Has shape rb_k+1 x r_k+1
    """
    # tmp = tn.einsum('ij,jbc->ibc',Phi_now,core_rhs) # shape rk-1 x Nk x rbk
    # Phi_next = tn.einsum('ijk,ijc->kc',core,tmp) 
    Phi_next = oe.contract('br,bnB,rnR->BR',Phi_now,core_rhs,core)
    return Phi_next
