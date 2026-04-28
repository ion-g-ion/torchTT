"""
Implements the cross approximation methods (DMRG).

"""
import torch as tn
import numpy as np
import torchtt
import datetime
from torchtt._decomposition import QR, SVD, rank_chop, lr_orthogonal, rl_orthogonal
from torchtt._iterative_solvers import BiCGSTAB_reset, gmres_restart
import opt_einsum as oe
import sys
from ._dmrg import _function_interpolate_dmrg, _maxvol
from ._amen_approx import amen_approx, AmenCallbacks


def _LU(M):
    """
    Perform an LU decomposition and returns L, U and a permutation vector P. 

    Args:
        M (torch.tensor): [description]

    Returns:
        tuple[torch.tensor,torch.tensor,torch.tensor]: L, U, P
    """
    LU, P = tn.linalg.lu_factor(M)
    P, L, U = tn.lu_unpack(LU, P)
    P = tn.reshape(tn.arange(P.shape[1],dtype=P.dtype,device=P.device),[1,-1]) @ P

    return L, U, tn.squeeze(P).to(tn.int64)


def _max_matrix(M):

    values, indices = M.flatten().topk(1)
    try:
        indices = [tn.unravel_index(i, M.shape) for i in indices]
    except:
        indices = [np.unravel_index(i, M.shape) for i in indices]

    return values, indices


def _maxvol(M):
    """
    Maxvol

    Args:
        M (torch.tensor): input matrix.

    Returns:
        torch.tensor: indices of tha maxvol submatrix.
    """

    if M.shape[1] >= M.shape[0]:
        # more cols than row -> return all the row indices
        idx = tn.tensor(range(M.shape[0]), dtype=tn.int64)
        return idx
    else:
        L, U, P = _LU(M)
        idx = P[:M.shape[1]]

    Msub = M[idx, :]

    Mat = tn.linalg.solve(Msub.T, M.T).t()

    for i in range(100):
        val_max, idx_max = _max_matrix(tn.abs(Mat))
        idx_max = idx_max[0]
        if val_max <= 1+5e-2:
            idx = tn.sort(idx)[0]
            return idx
        Mat += tn.outer(Mat[:, idx_max[1]], Mat[idx[idx_max[1]]] -
                        Mat[idx_max[0], :])/Mat[idx_max[0], idx_max[1]]
        idx[idx_max[1]] = idx_max[0]
    return idx


def function_interpolate(function, x, eps=1e-9, start_tens=None, nswp=20, kick=2, dtype=tn.float64, rmax=sys.maxsize, method='dmrg', verbose=False):
    if method == 'dmrg':
        return _function_interpolate_dmrg(function, x, eps, start_tens, nswp, kick, dtype, rmax, verbose)
    elif method == 'amen':
        return _function_interpolate_amen(function, x, eps, start_tens, nswp, kick, dtype, rmax, verbose)
    else:
        raise ValueError("Method must be 'dmrg' or 'amen'.")

class AmenCrossCallbacks(AmenCallbacks):
    def __init__(self, function, eval_mv, x, N, dtype, device):
        self.function = function
        self.eval_mv = eval_mv
        self.x = x
        self.N = N
        self.d = len(N)
        self.dtype = dtype
        self.device = device
        self.n_eval = 0

    def _eval_function(self, I_left, I_curr, I_right, k):
        I_left = I_left.to(device=self.device)
        I_curr = I_curr.to(device=self.device)
        I_right = I_right.to(device=self.device)
        rank_l = I_left.shape[0] if I_left.shape[1] > 0 else 1
        rank_r = I_right.shape[1] if I_right.shape[0] > 0 else 1
        nk = I_curr.shape[0]

        I1 = tn.reshape(tn.kron(tn.kron(tn.ones(rank_l, dtype=tn.int64, device=self.device), I_curr), tn.ones(rank_r, dtype=tn.int64, device=self.device)), [-1, 1])

        if I_left.shape[1] > 0:
            I3 = I_left[tn.kron(tn.kron(tn.arange(rank_l, dtype=tn.int64, device=self.device), tn.ones(nk, dtype=tn.int64, device=self.device)), tn.ones(rank_r, dtype=tn.int64, device=self.device)), :]
        else:
            I3 = tn.zeros((rank_l * nk * rank_r, 0), dtype=tn.int64, device=self.device)

        if I_right.shape[0] > 0:
            I4 = I_right[:, tn.kron(tn.kron(tn.ones(rank_l, dtype=tn.int64, device=self.device), tn.ones(nk, dtype=tn.int64, device=self.device)), tn.arange(rank_r, dtype=tn.int64, device=self.device))].t()
        else:
            I4 = tn.zeros((rank_l * nk * rank_r, 0), dtype=tn.int64, device=self.device)

        eval_index = tn.concat((I3, I1, I4), 1).to(dtype=tn.int64)

        if self.eval_mv:
            ev = tn.zeros((eval_index.shape[0], 0), dtype=self.dtype, device=self.device)
            for j in range(len(self.x)):
                core = self.x[j].cores[0][0, eval_index[:, 0], :]
                for i in range(1, self.d):
                    core = tn.einsum('ij,jil->il', core, self.x[j].cores[i][:, eval_index[:, i], :])
                core = tn.reshape(core[..., 0], [-1, 1])
                ev = tn.hstack((ev, core))
            res = tn.reshape(self.function(ev), [rank_l, nk, rank_r])
            self.n_eval += eval_index.shape[0]
        else:
            core = self.x.cores[0][0, eval_index[:, 0], :]
            for i in range(1, self.d):
                core = tn.einsum('ij,jil->il', core, self.x.cores[i][:, eval_index[:, i], :])
            core = core[..., 0]
            res = tn.reshape(self.function(core), [rank_l, nk, rank_r])
            self.n_eval += eval_index.shape[0]

        return res

    def compute_x_fwd(self, k, state_dict, x_cores, z_cores):
        I_left = state_dict['Jy_left'][k]
        I_right = state_dict['Jy_right'][k+1]
        I_curr = tn.arange(self.N[k], dtype=tn.int64)

        res = self._eval_function(I_left, I_curr, I_right, k)
        res = tn.linalg.solve(state_dict['Ps_left'][k], tn.reshape(res, [x_cores[k].shape[0], -1]))
        res = tn.linalg.solve(state_dict['Ps_right'][k+1].t(), tn.reshape(res, [-1, x_cores[k].shape[3]]).t()).t()
        res = tn.reshape(res, [x_cores[k].shape[0], 1, self.N[k], x_cores[k].shape[3]])

        norm_res = tn.linalg.norm(res)
        return res, norm_res

    def compute_x_bck(self, k, state_dict, x_cores, z_cores):
        I_left = state_dict['Jy_left'][k]
        I_right = state_dict['Jy_right'][k+1]
        I_curr = tn.arange(self.N[k], dtype=tn.int64)

        res = self._eval_function(I_left, I_curr, I_right, k)
        res = tn.linalg.solve(state_dict['Ps_left'][k], tn.reshape(res, [x_cores[k].shape[0], -1]))
        res = tn.linalg.solve(state_dict['Ps_right'][k+1].t(), tn.reshape(res, [-1, x_cores[k].shape[3]]).t()).t()
        res = tn.reshape(res, [x_cores[k].shape[0], 1, self.N[k], x_cores[k].shape[3]])

        norm_res = tn.linalg.norm(res)
        return res, norm_res

    def compute_z_bck(self, k, state_dict, x_cores, z_cores):
        I_left = state_dict['Jz_left'][k]
        I_right = state_dict['Jz_right'][k+1]
        I_curr = tn.arange(self.N[k], dtype=tn.int64)

        fz = self._eval_function(I_left, I_curr, I_right, k)
        fz = tn.linalg.solve(state_dict['Ps_z_left'][k], tn.reshape(fz, [z_cores[k].shape[0], -1]))
        fz = tn.linalg.solve(state_dict['Ps_z_right'][k+1].t(), tn.reshape(fz, [-1, z_cores[k].shape[3]]).t()).t()
        fz = tn.reshape(fz, [z_cores[k].shape[0], self.N[k], z_cores[k].shape[3]])

        cryz = tn.einsum('zl,lmn,nr->zmr', state_dict['phizy_left'][k], tn.reshape(x_cores[k], [x_cores[k].shape[0], self.N[k], x_cores[k].shape[3]]), state_dict['phizy_right'][k+1])

        return fz - cryz

    def compute_z_fwd(self, k, state_dict, x_cores, z_cores, u, v):
        I_left = state_dict['Jz_left'][k]
        I_right = state_dict['Jz_right'][k+1]
        I_curr = tn.arange(self.N[k], dtype=tn.int64)

        fz = self._eval_function(I_left, I_curr, I_right, k)
        fz = tn.linalg.solve(state_dict['Ps_z_left'][k], tn.reshape(fz, [z_cores[k].shape[0], -1]))
        fz = tn.linalg.solve(state_dict['Ps_z_right'][k+1].t(), tn.reshape(fz, [-1, z_cores[k].shape[3]]).t()).t()
        fz = tn.reshape(fz, [z_cores[k].shape[0], self.N[k], z_cores[k].shape[3]])

        rx_k = u.shape[0] // self.N[k]
        rx_k1 = v.shape[0]
        core_u = tn.reshape(u@v.t(), [rx_k, self.N[k], rx_k1])
        cryz = tn.einsum('zl,lmn,nr->zmr', state_dict['phizy_left'][k], core_u, state_dict['phizy_right'][k+1])
        
        cz_new = fz - cryz
        return cz_new

    def compute_enrichment(self, k, state_dict, x_cores, z_cores, u, v):
        I_left = state_dict['Jy_left'][k]
        I_right = state_dict['Jz_right'][k+1]
        I_curr = tn.arange(self.N[k], dtype=tn.int64)

        fz = self._eval_function(I_left, I_curr, I_right, k)
        fz = tn.linalg.solve(state_dict['Ps_left'][k], tn.reshape(fz, [x_cores[k].shape[0], -1]))
        fz = tn.linalg.solve(state_dict['Ps_z_right'][k+1].t(), tn.reshape(fz, [-1, z_cores[k].shape[3]]).t()).t()
        fs = tn.reshape(fz, [x_cores[k].shape[0], self.N[k], z_cores[k].shape[3]])

        rx_k = u.shape[0] // self.N[k]
        rx_k1 = v.shape[0]
        core_u = tn.reshape(u@v.t(), [rx_k, self.N[k], rx_k1])
        crys = tn.einsum('lmn,nr->lmr', core_u, state_dict['phizy_right'][k+1])

        return fs - crys

    def update_phis_bck(self, k, state_dict, x_cores, z_cores, swp, last):
        core = tn.einsum('ijkl,lm->ijkm', x_cores[k], state_dict['Ps_right'][k+1])
        core = tn.reshape(core, [x_cores[k].shape[0], -1]).t()
        idx = _maxvol(core)
        try:
            tmp = tn.unravel_index(idx[:x_cores[k].shape[0]], (self.N[k], x_cores[k].shape[3]))
        except:
            tmp = np.unravel_index(idx[:x_cores[k].shape[0]], (self.N[k], x_cores[k].shape[3]))

        idx_new = tn.tensor(np.vstack((tmp[0].reshape([1, -1]), state_dict['Jy_right'][k+1][:, tmp[1]])))
        state_dict['Jy_right'][k] = idx_new
        Ps_new = core[idx[:x_cores[k].shape[0]], :].t()
        s = tn.linalg.svdvals(Ps_new)
        min_s = tn.clamp(tn.min(s), min=1e-16)
        norm_factor = 1.0 # Removed scaling
        # Ps_new = Ps_new * norm_factor
        state_dict['Ps_right'][k] = Ps_new
        
        if 'normx' in state_dict:
            normx_val = norm_factor
            state_dict['normx'][k-1] = normx_val



        if not last:
            core_z = tn.einsum('ijkl,lm->ijkm', z_cores[k], state_dict['Ps_z_right'][k+1])
            core_z = tn.reshape(core_z, [z_cores[k].shape[0], -1]).t()
            idx_z = _maxvol(core_z)
            try:
                tmp_z = tn.unravel_index(idx_z[:z_cores[k].shape[0]], (self.N[k], z_cores[k].shape[3]))
            except:
                tmp_z = np.unravel_index(idx_z[:z_cores[k].shape[0]], (self.N[k], z_cores[k].shape[3]))

            idx_new_z = tn.tensor(np.vstack((tmp_z[0].reshape([1, -1]), state_dict['Jz_right'][k+1][:, tmp_z[1]])))
            state_dict['Jz_right'][k] = idx_new_z
            state_dict['Ps_z_right'][k] = core_z[idx_z[:z_cores[k].shape[0]], :].t()

            core_y = tn.reshape(x_cores[k], [x_cores[k].shape[0], self.N[k], x_cores[k].shape[3]])
            cry = tn.einsum('lmr,rt->lmt', core_y, state_dict['phizy_right'][k+1])
            cry = tn.reshape(cry, [x_cores[k].shape[0], -1]).t()
            state_dict['phizy_right'][k] = tn.linalg.solve(state_dict['Ps_z_right'][k], cry[idx_z[:z_cores[k].shape[0]], :]).t()

    def update_phis_fwd(self, k, state_dict, x_cores, z_cores, swp, last):
        core = tn.einsum('ij,jklm->iklm', state_dict['Ps_left'][k], x_cores[k])
        core = tn.reshape(core, [-1, x_cores[k].shape[3]])
        idx = _maxvol(core)
        try:
            tmp = tn.unravel_index(idx[:x_cores[k].shape[3]], (x_cores[k].shape[0], self.N[k]))
        except:
            tmp = np.unravel_index(idx[:x_cores[k].shape[3]], (x_cores[k].shape[0], self.N[k]))

        idx_new = tn.tensor(np.hstack((state_dict['Jy_left'][k][tmp[0], :], tmp[1].reshape([-1, 1]))))
        state_dict['Jy_left'][k+1] = idx_new
        
        Ps_new = core[idx[:x_cores[k].shape[3]], :]
        s = tn.linalg.svdvals(Ps_new)
        min_s = tn.clamp(tn.min(s), min=1e-16)
        state_dict['Ps_left'][k+1] = Ps_new
        
        if 'normx' in state_dict:
            state_dict['normx'][k] = 1.0


        if not last:
            core_z = tn.einsum('ij,jklm->iklm', state_dict['Ps_z_left'][k], z_cores[k])
            core_z = tn.reshape(core_z, [-1, z_cores[k].shape[3]])
            idx_z = _maxvol(core_z)
            try:
                tmp_z = tn.unravel_index(idx_z[:z_cores[k].shape[3]], (z_cores[k].shape[0], self.N[k]))
            except:
                tmp_z = np.unravel_index(idx_z[:z_cores[k].shape[3]], (z_cores[k].shape[0], self.N[k]))

            idx_new_z = tn.tensor(np.hstack((state_dict['Jz_left'][k][tmp_z[0], :], tmp_z[1].reshape([-1, 1]))))
            state_dict['Jz_left'][k+1] = idx_new_z
            state_dict['Ps_z_left'][k+1] = core_z[idx_z[:z_cores[k].shape[3]], :]

            core_y = tn.reshape(x_cores[k], [x_cores[k].shape[0], self.N[k], x_cores[k].shape[3]])
            cry = tn.einsum('tl,lmr->tmr', state_dict['phizy_left'][k], core_y)
            cry = tn.reshape(cry, [-1, x_cores[k].shape[3]])
            state_dict['phizy_left'][k+1] = tn.linalg.solve(state_dict['Ps_z_left'][k+1], cry[idx_z[:z_cores[k].shape[3]], :])


def _function_interpolate_amen(function, x, eps=1e-9, start_tens=None, nswp=20, kick=2, dtype=tn.float64, rmax=sys.maxsize, verbose=False):
    if isinstance(x, list) or isinstance(x, tuple):
        eval_mv = True
        N = x[0].N
    else:
        eval_mv = False
        N = x.N
    device = x[0].cores[0].device if eval_mv else x.cores[0].device

    if not eval_mv and len(N) == 1:
        return torchtt.TT(function(x.full())).to(device)

    if eval_mv and len(N) == 1:
        return torchtt.TT(function(x[0].full())).to(device)

    d = len(N)

    if start_tens is None:
        rank_init = 2
        cores = torchtt.random(N, rank_init, dtype, device).cores
        rx = [1]+[rank_init]*(d-1)+[1]
    else:
        rx = start_tens.R.copy()
        cores = [c+0 for c in start_tens.cores]

    M_dummy = [1] * d
    N_list = list(N)

    Jy_left = [tn.zeros((1, 0), dtype=tn.int64)] + [None]*d
    Jy_right = [None]*d + [tn.zeros((0, 1), dtype=tn.int64)]
    Jz_left = [tn.zeros((1, 0), dtype=tn.int64)] + [None]*d
    Jz_right = [None]*d + [tn.zeros((0, 1), dtype=tn.int64)]

    phizy_left = [tn.ones((1, 1), dtype=dtype, device=device)] + [None]*d
    phizy_right = [None]*d + [tn.ones((1, 1), dtype=dtype, device=device)]

    state_dict = {
        'Jy_left': Jy_left,
        'Jy_right': Jy_right,
        'Jz_left': Jz_left,
        'Jz_right': Jz_right,
        'phizy_left': phizy_left,
        'phizy_right': phizy_right,
        'Ps_left': [tn.ones((1, 1), dtype=dtype, device=device)] + [None] * d,
        'Ps_right': [None] * d + [tn.ones((1, 1), dtype=dtype, device=device)],
        'Ps_z_left': [tn.ones((1, 1), dtype=dtype, device=device)] + [None] * d,
        'Ps_z_right': [None] * d + [tn.ones((1, 1), dtype=dtype, device=device)],
        'normx': np.ones((d-1)),
        'enable_x_bck': False
    }

    callbacks = AmenCrossCallbacks(function, eval_mv, x, N_list, dtype, device)

    rz = [1]+(d-1)*[kick]+[1]

    x_cores = [tn.reshape(c, [c.shape[0], 1, c.shape[1], c.shape[2]]) for c in cores]

    x_cores, rx = amen_approx(M_dummy, N_list, d, x_cores, rx, rz, state_dict, callbacks,
                              nswp=nswp, eps=eps, rmax=rmax, kickrank=kick, kick2=0, verbose=verbose)

    if verbose:
        print('number of function calls ', callbacks.n_eval)

    x_cores = [tn.reshape(c, [c.shape[0], c.shape[2], c.shape[3]]) for c in x_cores]
    return torchtt.TT(x_cores)
def dmrg_cross(function, N, eps=1e-9, nswp=10, x_start=None, kick=2, dtype=tn.float64, device=None, eval_vect=True, rmax=sys.maxsize, verbose=False):
    """
    Approximate a tensor in the TT format given that the individual entries are given using a function.
    The function is given as a function handle taking as arguments a matrix of integer indices.

    Example:

        .. code-block:: python

            func = lambda I: 1/(2+I[:,0]+I[:,1]+I[:,2]+I[:,3]).to(dtype=torch.float64)
            N = [20]*4
            x = torchtt.interpolate.dmrg_cross(func, N, eps = 1e-7)


    Args:
        function (Callable): function handle.
        N (list[int]): the shape of the tensor.
        eps (float, optional): the relative accuracy. Defaults to 1e-9.
        nswp (int, optional): number of iterations. Defaults to 20.
        x_start (torchtt.TT, optional): initial approximation of the output tensor (None coresponds to random initialization). Defaults to None.
        kick (int, optional): enrichment rank. Defaults to 2.
        dtype (torch.dtype, optional): the dtype of the result. Defaults to tn.float64.
        device (torch.device, optional): the device where the approximation will be stored. Defaults to None.
        eval_vect (bool, optional): not yet implemented. Defaults to True.
        rmax (int, optional): the maximum rank. Defaults to the maximum possible integer.
        verbose (bool, optional): display debug information to the console. Defaults to False.

    Returns:
        torchtt.TT: the result.

    """
    # store the computed values
    computed_vals = dict()

    d = len(N)

    # random init of the tensor
    if x_start == None:
        rank_init = 2
        cores = torchtt.random(N, rank_init, dtype, device).cores
        rank = [1]+[rank_init]*(d-1)+[1]
    else:
        rank = x_start.R.copy()
        cores = [c+0 for c in x_start.cores]
    # cores = (ones(N,dtype=dtype)).cores

    cores, rank = lr_orthogonal(cores, rank, False)

    Mats = []*(d+1)

    Ps = [tn.ones((1, 1), dtype=dtype, device=device)]+(d-1) * \
        [None] + [tn.ones((1, 1), dtype=dtype, device=device)]
    # ortho
    Rm = tn.ones((1, 1), dtype=dtype, device=device)
    Idx = [tn.zeros((1, 0), dtype=tn.int64)]+(d-1)*[None] + \
        [tn.zeros((0, 1), dtype=tn.int64)]
    for k in range(d-1, 0, -1):

        tmp = tn.einsum('ijk,kl->ijl', cores[k], Rm)
        tmp = tn.reshape(tmp, [rank[k], -1]).t()
        core, Rmat = QR(tmp)

        rnew = min(N[k]*rank[k+1], rank[k])
        Jk = _maxvol(core)
        # print(Jk)
        try:
            tmp = tn.unravel_index(Jk[:rnew], (rank[k+1], N[k]))
        except:
            tmp = np.unravel_index(Jk[:rnew], (rank[k+1], N[k]))
        # if k==d-1:
        #    idx_new = tn.tensor(tmp[1].reshape([1,-1]))
        # else:
        idx_new = tn.tensor(
            np.vstack((tmp[1].reshape([1, -1]), Idx[k+1][:, tmp[0]])))

        Idx[k] = idx_new+0

        Rm = core[Jk, :]

        core = tn.linalg.solve(Rm.T, core.T)
        # core = tn.linalg.solve(Rm,core.T)
        Rm = (Rm@Rmat).t()
        # core = core.t()
        cores[k] = tn.reshape(core, [rnew, N[k], rank[k+1]])
        core = tn.reshape(core, [-1, rank[k+1]]) @ Ps[k+1]
        core = tn.reshape(core, [rank[k], -1]).t()
        _, Ps[k] = QR(core)
    cores[0] = tn.einsum('ijk,kl->ijl', cores[0], Rm)

    # for p in Ps:
    #     print(p)
    # for i in Idx:
    #     print(i)
    # return
    n_eval = 0

    for swp in range(nswp):

        max_err = 0.0
        if verbose:
            print('Sweep %d: ' % (swp+1))
        # left to right
        for k in range(d-1):
            if verbose:
                print('\tLR supercore %d,%d' % (k+1, k+2))
            I1 = tn.reshape(tn.kron(tn.kron(tn.ones(rank[k], dtype=tn.int64), tn.arange(N[k], dtype=tn.int64)), tn.kron(
                tn.ones(N[k+1], dtype=tn.int64), tn.ones(rank[k+2], dtype=tn.int64))), [-1, 1])
            I2 = tn.reshape(tn.kron(tn.kron(tn.ones(rank[k], dtype=tn.int64), tn.ones(N[k], dtype=tn.int64)), tn.kron(
                tn.arange(N[k+1], dtype=tn.int64), tn.ones(rank[k+2], dtype=tn.int64))), [-1, 1])
            I3 = Idx[k][tn.kron(tn.kron(tn.arange(rank[k], dtype=tn.int64), tn.ones(N[k], dtype=tn.int64)), tn.kron(
                tn.ones(N[k+1], dtype=tn.int64), tn.ones(rank[k+2], dtype=tn.int64))), :]
            I4 = Idx[k+2][:, tn.kron(tn.kron(tn.ones(rank[k], dtype=tn.int64), tn.ones(N[k], dtype=tn.int64)),
                                     tn.kron(tn.ones(N[k+1], dtype=tn.int64), tn.arange(rank[k+2], dtype=tn.int64)))].t()

            eval_index = tn.concat((I3, I1, I2, I4), 1)

            eval_index = tn.reshape(eval_index, [-1, d]).to(dtype=tn.int64)

            if verbose:
                print('\t\tnumber evaluations', eval_index.shape[0])

            if eval_vect:
                supercore = tn.reshape(function(eval_index), [
                                       rank[k], N[k], N[k+1], rank[k+2]])
                n_eval += eval_index.shape[0]
            else:
                supercore = tn.zeros(eval_index.shape[0], dtype=dtype, device=device)
                for ind in range(eval_index.shape[0]):
                    supercore[ind] = function(*eval_index[ind,:])
                supercore = tn.reshape(supercore, [rank[k], N[k], N[k+1], rank[k+2]])
                n_eval += eval_index.shape[0]

            # multiply with P_k left and right
            supercore = tn.einsum('ij,jklm,mn->ikln',
                                  Ps[k], supercore.to(dtype=dtype), Ps[k+2])
            rank[k] = supercore.shape[0]
            rank[k+2] = supercore.shape[3]
            supercore = tn.reshape(
                supercore, [supercore.shape[0]*supercore.shape[1], -1])

            # split the super core with svd
            U, S, V = SVD(supercore)
            rnew = rank_chop(S.cpu().numpy(), tn.linalg.norm(
                S).cpu().numpy()*eps/np.sqrt(d-1))+1
            rnew = min(S.shape[0], rnew)
            rnew = min(rmax, rnew)
            U = U[:, :rnew]
            S = S[:rnew]
            V = V[:rnew, :]
            # print('kkt new',tn.linalg.norm(supercore-U@tn.diag(S)@V))
            # kick the rank
            V = tn.diag(S) @ V
            UK = tn.randn((U.shape[0], kick), dtype=dtype, device=device)
            U, Rtemp = QR(tn.cat((U, UK), 1))
            radd = U.shape[1] - rnew
            if radd > 0:
                V = tn.cat(
                    (V, tn.zeros((radd, V.shape[1]), dtype=dtype, device=device)), 0)
                V = Rtemp @ V
            # print('kkt new',tn.linalg.norm(supercore-U@V))
            # compute err (dx)
            super_prev = tn.einsum('ijk,kmn->ijmn', cores[k], cores[k+1])
            super_prev = tn.einsum(
                'ij,jklm,mn->ikln', Ps[k], super_prev, Ps[k+2])
            err = tn.linalg.norm(
                supercore.flatten()-super_prev.flatten())/tn.linalg.norm(supercore)
            max_err = max(max_err, err)
            # update the rank
            if verbose:
                print('\t\trank updated %d -> %d, local error %e' %
                      (rank[k+1], U.shape[1], err))
            rank[k+1] = U.shape[1]

            U = tn.linalg.solve(Ps[k], tn.reshape(U, [rank[k], -1]))
            V = tn.linalg.solve(
                Ps[k+2].t(), tn.reshape(V, [rank[k+1]*N[k+1], rank[k+2]]).t()).t()

            # U = tn.einsum('ij,jkl->ikl',tn.linalg.inv(Ps[k]),tn.reshape(U,[rank[k],N[k],-1]))
            # V = tn.einsum('ijk,kl->ijl',tn.reshape(V,[-1,N[k+1],rank[k+2]]),tn.linalg.inv(Ps[k+2]))

            V = tn.reshape(V, [rank[k+1], -1])
            U = tn.reshape(U, [-1, rank[k+1]])

            # split cores
            Qmat, Rmat = QR(U)
            idx = _maxvol(Qmat)
            Sub = Qmat[idx, :]
            core = tn.linalg.solve(Sub.T, Qmat.T).t()
            core_next = Sub@Rmat@V
            cores[k] = tn.reshape(core, [rank[k], N[k], rank[k+1]])
            cores[k+1] = tn.reshape(core_next, [rank[k+1], N[k+1], rank[k+2]])
            # calc Ps
            tmp = tn.einsum('ij,jkl->ikl', Ps[k], cores[k])
            _, Ps[k+1] = QR(tn.reshape(tmp, [rank[k]*N[k], rank[k+1]]))

            # calc Idx
            try:
                tmp = tn.unravel_index(idx[:rank[k+1]], (rank[k], N[k]))
            except:
                tmp = np.unravel_index(idx[:rank[k+1]], (rank[k], N[k]))
            idx_new = tn.tensor(
                np.hstack((Idx[k][tmp[0], :], tmp[1].reshape([-1, 1]))))
            Idx[k+1] = idx_new+0

        # right to left

        for k in range(d-2, -1, -1):
            if verbose:
                print('\tRL supercore %d,%d' % (k+1, k+2))
            I1 = tn.reshape(tn.kron(tn.kron(tn.ones(rank[k], dtype=tn.int64), tn.arange(N[k], dtype=tn.int64)), tn.kron(
                tn.ones(N[k+1], dtype=tn.int64), tn.ones(rank[k+2], dtype=tn.int64))), [-1, 1])
            I2 = tn.reshape(tn.kron(tn.kron(tn.ones(rank[k], dtype=tn.int64), tn.ones(N[k], dtype=tn.int64)), tn.kron(
                tn.arange(N[k+1], dtype=tn.int64), tn.ones(rank[k+2], dtype=tn.int64))), [-1, 1])
            I3 = Idx[k][tn.kron(tn.kron(tn.arange(rank[k], dtype=tn.int64), tn.ones(N[k], dtype=tn.int64)), tn.kron(
                tn.ones(N[k+1], dtype=tn.int64), tn.ones(rank[k+2], dtype=tn.int64))), :]
            I4 = Idx[k+2][:, tn.kron(tn.kron(tn.ones(rank[k], dtype=tn.int64), tn.ones(N[k], dtype=tn.int64)),
                                     tn.kron(tn.ones(N[k+1], dtype=tn.int64), tn.arange(rank[k+2], dtype=tn.int64)))].t()

            eval_index = tn.concat((I3, I1, I2, I4), 1)
            eval_index = tn.reshape(eval_index, [-1, d]).to(dtype=tn.int64)

            if verbose:
                print('\t\tnumber evaluations', eval_index.shape[0])

            if eval_vect:
                supercore = tn.reshape(function(eval_index).to(dtype=dtype), [
                                       rank[k], N[k], N[k+1], rank[k+2]])
                n_eval += eval_index.shape[0]
            else:
                supercore = tn.zeros(eval_index.shape[0], dtype=dtype, device=device)
                for ind in range(eval_index.shape[0]):
                    supercore[ind] = function(*eval_index[ind,:])
                supercore = tn.reshape(supercore, [rank[k], N[k], N[k+1], rank[k+2]])
                n_eval += eval_index.shape[0]

            # multiply with P_k left and right
            supercore = tn.einsum('ij,jklm,mn->ikln',
                                  Ps[k], supercore.to(dtype=dtype), Ps[k+2])
            rank[k] = supercore.shape[0]
            rank[k+2] = supercore.shape[3]
            supercore = tn.reshape(
                supercore, [supercore.shape[0]*supercore.shape[1], -1])

            # split the super core with svd
            U, S, V = SVD(supercore)
            rnew = rank_chop(S.cpu().numpy(), tn.linalg.norm(
                S).cpu().numpy()*eps/np.sqrt(d-1))+1
            rnew = min(S.shape[0], rnew)
            rnew = min(rmax, rnew)
            U = U[:, :rnew]
            S = S[:rnew]
            V = V[:rnew, :]

            # kick the rank
            U = U @ tn.diag(S)
            VK = tn.randn((kick, V.shape[1]), dtype=dtype, device=device)
            V, Rtemp = QR(tn.cat((V, VK), 0).t())
            radd = Rtemp.shape[1] - rnew
            if radd > 0:
                U = tn.cat(
                    (U, tn.zeros((U.shape[0], radd), dtype=dtype, device=device)), 1)
                U = U @ Rtemp.T
                V = V.t()

            # compute err (dx)
            super_prev = tn.einsum('ijk,kmn->ijmn', cores[k], cores[k+1])
            super_prev = tn.einsum(
                'ij,jklm,mn->ikln', Ps[k], super_prev, Ps[k+2])
            err = tn.linalg.norm(
                supercore.flatten()-super_prev.flatten())/tn.linalg.norm(supercore)
            max_err = max(max_err, err)
            # update the rank
            if verbose:
                print('\t\trank updated %d -> %d, local error %e' %
                      (rank[k+1], U.shape[1], err))
            rank[k+1] = U.shape[1]

            U = tn.linalg.solve(Ps[k], tn.reshape(U, [rank[k], -1]))
            V = tn.linalg.solve(
                Ps[k+2].t(), tn.reshape(V, [rank[k+1]*N[k+1], rank[k+2]]).t()).t()

            # U = tn.einsum('ij,jkl->ikl',tn.linalg.inv(Ps[k]),tn.reshape(U,[rank[k],N[k],-1]))
            # V = tn.einsum('ijk,kl->ijl',tn.reshape(V,[-1,N[k+1],rank[k+2]]),tn.linalg.inv(Ps[k+2]))

            V = tn.reshape(V, [rank[k+1], -1])
            U = tn.reshape(U, [-1, rank[k+1]])

            # split cores
            Qmat, Rmat = QR(V.T)
            idx = _maxvol(Qmat)
            Sub = Qmat[idx, :]
            core_next = tn.linalg.solve(Sub.T, Qmat.T)
            core = U@(Sub@Rmat).t()
            cores[k] = tn.reshape(core, [rank[k], N[k], -1])
            cores[k+1] = tn.reshape(core_next, [-1, N[k+1], rank[k+2]])

            # calc Ps
            tmp = tn.einsum('ijk,kl->ijl', cores[k+1], Ps[k+2])
            _, tmp = QR(tn.reshape(tmp, [rank[k+1], -1]).t())
            Ps[k+1] = tmp
            # calc Idx
            try:
                tmp = tn.unravel_index(idx[:rank[k+1]], (N[k+1], rank[k+2]))
            except:
                tmp = np.unravel_index(idx[:rank[k+1]], (N[k+1], rank[k+2]))
            idx_new = tn.tensor(
                np.vstack((tmp[0].reshape([1, -1]), Idx[k+2][:, tmp[1]])))
            Idx[k+1] = idx_new+0
        # xxx = TT(cores)
        # print('#            ',xxx[1,2,3,4])

        # exit condition

        if max_err < eps:
            if verbose:
                print('Max error %e < %e  ---->  DONE' % (max_err, eps))
            break
        else:
            if verbose:
                print('Max error %g' % (max_err))
    if verbose:
        print('number of function calls ', n_eval)
        print()

    return torchtt.TT(cores)
