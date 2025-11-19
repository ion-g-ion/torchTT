import torch as tn
import math 
import torchtt
import datetime
from torchtt._decomposition import QR, SVD, lr_orthogonal, rl_orthogonal, rank_chop
from ._extras import randn
import opt_einsum as oe 
from .errors import InvalidArguments
import abc
import numpy as np

class AMENApproximateExpression(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        required_methods = ["update_phi", "bfun", "dtype", "device", "result_shape"]
       
        # Check if the subclass has the required methods
        has_methods = all(callable(getattr(subclass, method, None)) for method in required_methods)

        if has_methods:
            return True  
        return NotImplemented  

    @abc.abstractmethod
    def update_phi(self, path: str, file_name: str):
        """Update the phi"""
        raise NotImplementedError

    @abc.abstractmethod
    def bfun(self, path: str, file_name: str):
        """Bfun"""
        raise NotImplementedError
    
    @property
    @abc.abstractmethod
    def dtype(self):
        """ Get the dtype. """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def device(self):
        """ Get the device. """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def result_shape(self):
        """ Get the dtype. """
        raise NotImplementedError
    
class AMENApproximateMV(AMENApproximateExpression):
    
    def __init__(self):
        pass
    
def amen_approx_interface(operator, approx_ttm = False, nswp=20,eps=1e-12, y0 = None, kickrank=4, kickrank2=0, verbose=0):
    
    
    dtype = operator.dtype
    device = operator.device
    
    m, n = operator.result_shape
    d = len(m)
    is_ttm = m is not None

    initial_qr = True
    
    if y0 is None:
        ry = [1] + [2]*(d-1) + [1]
        if is_ttm:
            cores_y = randn([(mm, nn) for mm, nn in zip(m, n)], R = ry, device=device, dtype=dtype).cores
        else:
            cores_y = randn(n, R = ry, device=device, dtype=dtype).cores
    else:
        if n != y0.N and (is_ttm and y0.M != m):
            raise InvalidArguments("The initial tensor shape does not match the result. ")
        cores_y = y0.to(device=device, dtype=dtype).cores
        ry = y0.R.copy()
        
    if kickrank or kickrank2:
        rz = [1] + [kickrank+kickrank2]*(d-1) + [1]
        if is_ttm:
            cores_z = randn([(mm, nn) for mm, nn in zip(m, n)], R = rz, device=device, dtype=dtype).cores
        else:
            cores_z = randn(n, R = rz, device=device, dtype=dtype).cores
        phi_zy = [tn.ones((1,1), device=device, dtype=dtype)] + [None]*(d-1) + [tn.ones((1,1), device=device, dtype=dtype)]

    nrms = [0.0]*d
    
    # Initial orthogonalization
    for k in range(d-1):
        if initial_qr:
            core = tn.reshape(cores_y[k], [-1, ry[k+1]])
            core, R = tn.linalg.qr(core)
            nrm = tn.linalg.norm(core, p='fro')
            if nrm>0:
                R /= nrm 
            core_next = tn.einsum("rs,s...->r...", R, cores_y[k+1])
            ry[k+1] = core.shape[1]
            cores_y[k] = tn.reshape(core, tuple(cores_y[k].shape[:-1]) + (ry[k+1],))
            cores_y[k+1] = core_next
            
        # >>>>> update newt phiy left to right, get nrms[k]
            
        # also the z cpmponent
        if kickrank or kickrank2:
            core = tn.reshape(cores_z[k], [-1, rz[k+1]])
            core, R = tn.linalg.qr(core)
            nrm = tn.linalg.norm(core, p="fro")
            if nrm > 0:
                R /= nrm
            core_next = tn.einsum("rs,s...->r...", R, cores_z[k+1])
            rz[k+1] = core.shape[1]
            cores_z[k] = tn.reshape(core, tuple(cores_z[k].shape[:-1]) + (rz[k+1],))
            cores_z[k+1] = core_next

            # >>>> update newxt phizx with norms left to right 
            # >>>> update newxt phizy with norms left to right
            
    k = d-1
    direction = -1
    sweep = 1
    max_dx = 0
    
    while sweep <= nswp:
        
        # obtain the core 
        # >>>>> 
        core = None 
        nrms[k] = tn.linalg.norm(core, p="fro")
        if nrms[k] > 0:
            core = core/nrms[k]
        else:
            nrms[k] = 1.0
        dx = tn.linalg.norm(core.flatten()-cores_y[k].flatten())
        max_dx = max(dx, max_dx)
        
        if direction > 0 and k < d-1:
            core = tn.reshape(cores_y[k], [-1, ry[k+1]])
            U,S,V = tn.linalg.svd(cores_y, full_matrices=False)
            rnew = rank_chop(S.cpu().cumpy(), eps*tn.linalg.norm(S.cpu())/math.sqrt(d))
            U = U[:, :rnew]
            V = S[:rnew, None] * V[:rnew, :]
            
            if kickrank or kickrank2:
                core = U @ V
            
                # >>> use bfun to obtain crz 
                core_z = None
                
                core_yz = oe.einsum("y...Y,yz,YZ->z...Z", tn.reshape(core, [ry[k], m[k], n[k], ry[k+1]] if is_ttm else [ry[k], n[k], ry[k+1]]), phi_zy[k], phi_zy[k+1])
                
                core_z = core_z/nrms[k] - core_yz
                nrm_z = tn.linalg.norm(core_z)
                
                if kickrank2 > 0:
                    core_z, _,  _ = tn.linalg.svd(core_z.reshape([-1, core_z.shape[-1]]))
                    core_z = core_z[:, :min(core_z.shape[1], kickrank)]
                    core_z = tn.cat((core_z, tn.randn([core_z.shape[0], kickrank2], dtype=core_z.dtype, device=core_z.device)), 1)
                    
            cores_y[k] = tn.reshape(U, [ry[k], m[k], n[k], rnew] if is_ttm else [ry[k], n[k], rnew])
            cores_y[k+1] = tn.einsum("nr,r...->n...", V, cores_y[k+1])
            ry[k+1] = rnew
            
            # >>>> update next phy yax left to right 
            
            if kickrank or kickrank2:
                
                core_z, Rmat = QR(tn.reshape(core_z, [-1, rz[k+1]]))
                rz[k+1] = core_z.shape[1]
                cores_z[k] = tn.reshape(core_z, [-1, m[k], n[k], rz[k+1]] if is_ttm else [-1, n[k], rz[k+1]])
        
                # phizax{i+1} = compute_next_Phi(phizax{i}, z{i}, A{i}, x{i}, 'lr', nrms(i));
                # phizy{i+1} = compute_next_Phi(phizy{i}, z{i}, [], y{i}, 'lr');

        
        
        
        
        elif direction < 0 and k > 0:
            core = tn.reshape(cores_y[k], [ry[k], -1])
            U,S,V = tn.linalg.svd(core, full_matrices=False)
            rnew = rank_chop(S.cpu().cumpy(), eps*tn.linalg.norm(S.cpu())/math.sqrt(d))
            V = V[:rnew, :]
            U = U[:,:rnew] * S[None, :rnew]
        
            if kickrank or kickrank2:
                core_y = U @ V
                
                # use bfun to update core_z
                core_z = None 
                
                
                core_tmp = oe.einsum("y...Y,yz->z...Y", tn.reshape(core_y, [ry[k], m[k], n[k], ry[k+1]] if is_ttm else [ry[k], n[k], ry[k+1]]), phi_zy[k])
                core_yz = oe.einsum("z...Y,YZ->z...Z", core_tmp, phi_zy[k+1])
                    
                core_z = core_z/nrms[k] - core_yz
                nrm_z = tn.linalg.norm(core_z)

                if kickrank2 > 0:
                    _, _, core_z = SVD(tn.reshape(core_z, [core_z.shape[0], -1]))
                    core_z = core_z[:min(core_z.shape[0], kickrank), :]
                    core_z = tn.cat((core_z, tn.randn(kickrank2, core_z.shape[-1])), 0)
                    core_z = tn.reshape(core_z, [-1, m[k], n[k], rz[k+1]] if is_ttm else [-1, n[k], rz[k+1]])
        

                # use bfun to obtain core_yz of shape zmnY
                core_yz = None
                
                core_yz = core_yz / nrms[k] - core_tmp 
                
                V = tn.cat([V.reshape([V.shape[0], -1]), core_yz.reshape([core_yz.shape[0], -1])], 0)
                [V, Rmat] = QR(V.t())
                
                U = tn.cat([U, tn.zeros([ry[k], rz[k]], dtype=dtype, device=device)], 1)
                U = U @ Rmat.t()
                rnew = V.shape[1]
            
            cores_y[k-1] = tn.einsum("...Y,Yr->...r", cores_y[k-1], U) 
            cores_y[k] = tn.reshape(V.t(), [rnew, m[k], n[k], ry[k+1]] if is_ttm else [rnew, n[k], ry[k+1]])
            ry[k] = rnew
            
            # Update phizax
            
            if kickrank or kickrank2:
                core_z, Rmat = QR(core_z.reshape([rz[k], -1]).t())
                rz[k] = core_z.shape[1]
                cores_z[k] = tn.reshape(core_z.t(), [rz[k], m[k], n[k], rz[k+1]] if is_ttm else [rz[k], n[k], rz[k+1]])

                # update phiz, phis
                
        if verbose:
            print()
            
        if (direction > 0 and k == d-1) or (direction<0 and k==0):
            if verbose:
                pass 
            
            if (max_dx < eps or sweep == nswp) and direction > 0:
                break
            else:
                if direction > 0:
                    swp += 1
            
            max_dx = 0.0
            direction *= -1
        else:
            k += direction
            
            
        if verbose:
            time_total = datetime.datetime.now() - time_total
            print()
            print('Finished after', swp+1, ' sweeps and ', time_total)
            print()
        nrms = np.exp(np.sum(np.log(nrms))/d)

        for k in range(d):
            cores_y[k] = cores_y[k] * nrms

        y = torchtt.TT(cores_y)

        return y
