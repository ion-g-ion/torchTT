"""
Implements the cross approximation methods (DMRG).

"""
import torch as tn
import numpy as np
import torchtt
import datetime
from torchtt._decomposition import QR, SVD, lr_orthogonal, rl_orthogonal
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
        idx = tn.arange(M.shape[0], dtype=tn.int64, device=M.device)
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


def _unravel_index(idx, shape, device):
    idx = idx.to(device=device, dtype=tn.int64) if tn.is_tensor(idx) else tn.as_tensor(idx, dtype=tn.int64, device=device)
    try:
        tmp = tn.unravel_index(idx, shape)
    except Exception:
        tmp = np.unravel_index(idx.detach().cpu().numpy(), shape)
        tmp = tuple(tn.as_tensor(t, dtype=tn.int64, device=device) for t in tmp)

    return tuple(t.to(device=device, dtype=tn.int64) if tn.is_tensor(t) else tn.as_tensor(t, dtype=tn.int64, device=device) for t in tmp)


def _rank_chop_torch(s, eps):
    if tn.linalg.norm(s) == 0.0:
        return 1

    eps = tn.as_tensor(eps, dtype=s.dtype, device=s.device)
    if eps <= 0.0:
        return s.numel()

    sc = tn.cumsum(tn.abs(tn.flip(s, dims=[0]))**2, dim=0)
    sc = tn.flip(sc, dims=[0])
    r = int(tn.argmax((sc < eps**2).to(tn.int64)).item())
    r = r if r > 0 else 1
    return s.numel() if sc[-1] > eps**2 else r


def _factorize_projection(mat):
    return tn.linalg.lu_factor(mat)


def _solve_projection(state_dict, key, idx, rhs, transpose=False):
    factor_key = key + '_lu'
    factors = state_dict.get(factor_key, None)
    if factors is None:
        factors = [None] * len(state_dict[key])
        state_dict[factor_key] = factors

    if factors[idx] is None:
        factors[idx] = _factorize_projection(state_dict[key][idx])

    lu, pivots = factors[idx]
    return tn.linalg.lu_solve(lu, pivots, rhs, adjoint=transpose)


def _build_one_core_eval_index(I_left, I_curr, I_right, rank_l, rank_r, device):
    nk = I_curr.shape[0]
    n_eval = rank_l * nk * rank_r

    left_rows = tn.arange(rank_l, dtype=tn.int64, device=device).repeat_interleave(nk * rank_r)
    curr_col = I_curr.repeat_interleave(rank_r).repeat(rank_l).reshape(-1, 1)
    right_cols = tn.arange(rank_r, dtype=tn.int64, device=device).repeat(rank_l * nk)

    if I_left.shape[1] > 0:
        I3 = I_left[left_rows, :]
    else:
        I3 = tn.zeros((n_eval, 0), dtype=tn.int64, device=device)

    if I_right.shape[0] > 0:
        I4 = I_right[:, right_cols].t()
    else:
        I4 = tn.zeros((n_eval, 0), dtype=tn.int64, device=device)

    return tn.cat((I3, curr_col, I4), 1).to(dtype=tn.int64)


def _build_two_core_eval_index(Idx_left, Idx_right, rank_l, n_left, n_right, rank_r, device):
    n_eval = rank_l * n_left * n_right * rank_r

    left_rows = tn.arange(rank_l, dtype=tn.int64, device=device).repeat_interleave(n_left * n_right * rank_r)
    i_left = tn.arange(n_left, dtype=tn.int64, device=device).repeat_interleave(n_right * rank_r).repeat(rank_l).reshape(-1, 1)
    i_right = tn.arange(n_right, dtype=tn.int64, device=device).repeat_interleave(rank_r).repeat(rank_l * n_left).reshape(-1, 1)
    right_cols = tn.arange(rank_r, dtype=tn.int64, device=device).repeat(rank_l * n_left * n_right)

    if Idx_left.shape[1] > 0:
        I3 = Idx_left[left_rows, :]
    else:
        I3 = tn.zeros((n_eval, 0), dtype=tn.int64, device=device)

    if Idx_right.shape[0] > 0:
        I4 = Idx_right[:, right_cols].t()
    else:
        I4 = tn.zeros((n_eval, 0), dtype=tn.int64, device=device)

    return tn.cat((I3, i_left, i_right, I4), 1).to(dtype=tn.int64)


def function_interpolate(function, x, eps=1e-9, start_tens=None, nswp=20, kick=2, kick2=0, dtype=tn.float64, rmax=sys.maxsize, method='dmrg', verbose=False):
    """
    Interpolate a function using tensor train cross approximation.

    Args:
        function (Callable): Function to interpolate.
        x (torchtt.TT or list[torchtt.TT]): The points at which to evaluate the function.
        eps (float, optional): The desired relative error. Defaults to 1e-9.
        start_tens (torchtt.TT, optional): Initial tensor train approximation. Defaults to None.
        nswp (int, optional): Number of sweeps. Defaults to 20.
        kick (int, optional): Rank enrichment. Defaults to 2.
        kick2 (int, optional): Secondary rank enrichment (meant for amen method). Defaults to 0.
        dtype (torch.dtype, optional): The datatype of the result. Defaults to tn.float64.
        rmax (int, optional): Maximum allowed rank. Defaults to sys.maxsize.
        method (str, optional): Method to use ('dmrg' or 'amen'). Defaults to 'dmrg'.
        verbose (bool, optional): If True, display information. Defaults to False.

    Raises:
        ValueError: If the method is not 'dmrg' or 'amen'.

    Returns:
        torchtt.TT: The interpolated tensor.
    """
    if method == 'dmrg':
        return _function_interpolate_dmrg(function, x, eps, start_tens, nswp, kick, dtype, rmax, verbose)
    elif method == 'amen':
        return _function_interpolate_amen(function, x, eps, start_tens, nswp, kick, kick2, dtype, rmax, verbose)
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

        eval_index = _build_one_core_eval_index(I_left, I_curr, I_right, rank_l, rank_r, self.device)

        if self.eval_mv:
            values = []
            for j in range(len(self.x)):
                core = self.x[j].cores[0][0, eval_index[:, 0], :]
                for i in range(1, self.d):
                    core = tn.einsum('ij,jil->il', core, self.x[j].cores[i][:, eval_index[:, i], :])
                core = tn.reshape(core[..., 0], [-1, 1])
                values.append(core)
            ev = tn.cat(values, dim=1)
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
        I_curr = tn.arange(self.N[k], dtype=tn.int64, device=self.device)

        rx_k = I_left.shape[0] if I_left.shape[1] > 0 else 1
        rx_k1 = I_right.shape[1] if I_right.shape[0] > 0 else 1

        res = self._eval_function(I_left, I_curr, I_right, k)
        res = _solve_projection(state_dict, 'Ps_left', k, tn.reshape(res, [rx_k, -1]))
        res = _solve_projection(state_dict, 'Ps_right', k+1, tn.reshape(res, [-1, rx_k1]).t(), transpose=True).t()
        res = tn.reshape(res, [rx_k, 1, self.N[k], rx_k1])

        norm_res = tn.linalg.norm(res)
        return res, norm_res

    def compute_x_bck(self, k, state_dict, x_cores, z_cores):
        I_left = state_dict['Jy_left'][k]
        I_right = state_dict['Jy_right'][k+1]
        I_curr = tn.arange(self.N[k], dtype=tn.int64, device=self.device)

        rx_k = I_left.shape[0] if I_left.shape[1] > 0 else 1
        rx_k1 = I_right.shape[1] if I_right.shape[0] > 0 else 1

        res = self._eval_function(I_left, I_curr, I_right, k)
        res = _solve_projection(state_dict, 'Ps_left', k, tn.reshape(res, [rx_k, -1]))
        res = _solve_projection(state_dict, 'Ps_right', k+1, tn.reshape(res, [-1, rx_k1]).t(), transpose=True).t()
        res = tn.reshape(res, [rx_k, 1, self.N[k], rx_k1])

        norm_res = tn.linalg.norm(res)
        return res, norm_res

    def compute_z_bck(self, k, state_dict, x_cores, z_cores):
        I_left = state_dict['Jz_left'][k]
        I_right = state_dict['Jz_right'][k+1]
        I_curr = tn.arange(self.N[k], dtype=tn.int64, device=self.device)

        rz_k = I_left.shape[0] if I_left.shape[1] > 0 else 1
        rz_k1 = I_right.shape[1] if I_right.shape[0] > 0 else 1

        fz = self._eval_function(I_left, I_curr, I_right, k)
        fz = _solve_projection(state_dict, 'Ps_z_left', k, tn.reshape(fz, [rz_k, -1]))
        fz = _solve_projection(state_dict, 'Ps_z_right', k+1, tn.reshape(fz, [-1, rz_k1]).t(), transpose=True).t()
        fz = tn.reshape(fz, [rz_k, self.N[k], rz_k1])

        cryz = oe.contract('zl,lmn,nr->zmr', state_dict['phizy_left'][k], tn.reshape(x_cores[k], [x_cores[k].shape[0], self.N[k], x_cores[k].shape[3]]), state_dict['phizy_right'][k+1])

        return fz - cryz

    def compute_z_fwd(self, k, state_dict, x_cores, z_cores, u, v):
        I_left = state_dict['Jz_left'][k]
        I_right = state_dict['Jz_right'][k+1]
        I_curr = tn.arange(self.N[k], dtype=tn.int64, device=self.device)

        rz_k = I_left.shape[0] if I_left.shape[1] > 0 else 1
        rz_k1 = I_right.shape[1] if I_right.shape[0] > 0 else 1

        fz = self._eval_function(I_left, I_curr, I_right, k)
        fz = _solve_projection(state_dict, 'Ps_z_left', k, tn.reshape(fz, [rz_k, -1]))
        fz = _solve_projection(state_dict, 'Ps_z_right', k+1, tn.reshape(fz, [-1, rz_k1]).t(), transpose=True).t()
        fz = tn.reshape(fz, [rz_k, self.N[k], rz_k1])

        rx_k = u.shape[0] // self.N[k]
        rx_k1 = v.shape[0]
        core_u = tn.reshape(u@v.t(), [rx_k, self.N[k], rx_k1])
        cryz = oe.contract('zl,lmn,nr->zmr', state_dict['phizy_left'][k], core_u, state_dict['phizy_right'][k+1])
        
        cz_new = fz - cryz
        return cz_new

    def compute_enrichment(self, k, state_dict, x_cores, z_cores, u, v):
        I_left = state_dict['Jy_left'][k]
        I_right = state_dict['Jz_right'][k+1]
        I_curr = tn.arange(self.N[k], dtype=tn.int64, device=self.device)

        rx_k = I_left.shape[0] if I_left.shape[1] > 0 else 1
        rz_k1 = I_right.shape[1] if I_right.shape[0] > 0 else 1

        fz = self._eval_function(I_left, I_curr, I_right, k)
        fz = _solve_projection(state_dict, 'Ps_left', k, tn.reshape(fz, [rx_k, -1]))
        fz = _solve_projection(state_dict, 'Ps_z_right', k+1, tn.reshape(fz, [-1, rz_k1]).t(), transpose=True).t()
        fs = tn.reshape(fz, [rx_k, self.N[k], rz_k1])

        rx_k = u.shape[0] // self.N[k]
        rx_k1 = v.shape[0]
        core_u = tn.reshape(u@v.t(), [rx_k, self.N[k], rx_k1])
        crys = oe.contract('lmn,nr->lmr', core_u, state_dict['phizy_right'][k+1])

        return fs - crys

    def update_phis_bck(self, k, state_dict, x_cores, z_cores, swp, last):
        core = tn.einsum('ijkl,lm->ijkm', x_cores[k], state_dict['Ps_right'][k+1])
        core = tn.reshape(core, [x_cores[k].shape[0], -1]).t()
        idx = _maxvol(core)
        tmp = _unravel_index(idx[:x_cores[k].shape[0]], (self.N[k], x_cores[k].shape[3]), self.device)

        idx_new = tn.vstack((tmp[0].reshape([1, -1]), state_dict['Jy_right'][k+1][:, tmp[1]]))
        state_dict['Jy_right'][k] = idx_new
        Ps_new = core[idx[:x_cores[k].shape[0]], :].t()
        norm_factor = 1.0 # Removed scaling
        # Ps_new = Ps_new * norm_factor
        state_dict['Ps_right'][k] = Ps_new
        state_dict['Ps_right_lu'][k] = _factorize_projection(Ps_new)
        
        if 'normx' in state_dict:
            normx_val = norm_factor
            state_dict['normx'][k-1] = normx_val



        if not last:
            core_z = tn.einsum('ijkl,lm->ijkm', z_cores[k], state_dict['Ps_z_right'][k+1])
            core_z = tn.reshape(core_z, [z_cores[k].shape[0], -1]).t()
            idx_z = _maxvol(core_z)
            tmp_z = _unravel_index(idx_z[:z_cores[k].shape[0]], (self.N[k], z_cores[k].shape[3]), self.device)

            idx_new_z = tn.vstack((tmp_z[0].reshape([1, -1]), state_dict['Jz_right'][k+1][:, tmp_z[1]]))
            state_dict['Jz_right'][k] = idx_new_z
            Ps_z_new = core_z[idx_z[:z_cores[k].shape[0]], :].t()
            state_dict['Ps_z_right'][k] = Ps_z_new
            state_dict['Ps_z_right_lu'][k] = _factorize_projection(Ps_z_new)

            core_y = tn.reshape(x_cores[k], [x_cores[k].shape[0], self.N[k], x_cores[k].shape[3]])
            cry = oe.contract('lmr,rt->lmt', core_y, state_dict['phizy_right'][k+1])
            cry = tn.reshape(cry, [x_cores[k].shape[0], -1]).t()
            state_dict['phizy_right'][k] = _solve_projection(state_dict, 'Ps_z_right', k, cry[idx_z[:z_cores[k].shape[0]], :]).t()

    def update_phis_fwd(self, k, state_dict, x_cores, z_cores, swp, last):
        core = tn.einsum('ij,jklm->iklm', state_dict['Ps_left'][k], x_cores[k])
        core = tn.reshape(core, [-1, x_cores[k].shape[3]])
        idx = _maxvol(core)
        tmp = _unravel_index(idx[:x_cores[k].shape[3]], (x_cores[k].shape[0], self.N[k]), self.device)

        idx_new = tn.hstack((state_dict['Jy_left'][k][tmp[0], :], tmp[1].reshape([-1, 1])))
        state_dict['Jy_left'][k+1] = idx_new
        
        Ps_new = core[idx[:x_cores[k].shape[3]], :]
        state_dict['Ps_left'][k+1] = Ps_new
        state_dict['Ps_left_lu'][k+1] = _factorize_projection(Ps_new)
        
        if 'normx' in state_dict:
            state_dict['normx'][k] = 1.0


        if not last:
            core_z = tn.einsum('ij,jklm->iklm', state_dict['Ps_z_left'][k], z_cores[k])
            core_z = tn.reshape(core_z, [-1, z_cores[k].shape[3]])
            idx_z = _maxvol(core_z)
            tmp_z = _unravel_index(idx_z[:z_cores[k].shape[3]], (z_cores[k].shape[0], self.N[k]), self.device)

            idx_new_z = tn.hstack((state_dict['Jz_left'][k][tmp_z[0], :], tmp_z[1].reshape([-1, 1])))
            state_dict['Jz_left'][k+1] = idx_new_z
            Ps_z_new = core_z[idx_z[:z_cores[k].shape[3]], :]
            state_dict['Ps_z_left'][k+1] = Ps_z_new
            state_dict['Ps_z_left_lu'][k+1] = _factorize_projection(Ps_z_new)

            core_y = tn.reshape(x_cores[k], [x_cores[k].shape[0], self.N[k], x_cores[k].shape[3]])
            cry = oe.contract('tl,lmr->tmr', state_dict['phizy_left'][k], core_y)
            cry = tn.reshape(cry, [-1, x_cores[k].shape[3]])
            state_dict['phizy_left'][k+1] = _solve_projection(state_dict, 'Ps_z_left', k+1, cry[idx_z[:z_cores[k].shape[3]], :])


def _function_interpolate_amen(function, x, eps=1e-9, start_tens=None, nswp=20, kick=2, kick2=0, dtype=tn.float64, rmax=sys.maxsize, verbose=False):
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

    Jy_left = [tn.zeros((1, 0), dtype=tn.int64, device=device)] + [None]*d
    Jy_right = [None]*d + [tn.zeros((0, 1), dtype=tn.int64, device=device)]
    Jz_left = [tn.zeros((1, 0), dtype=tn.int64, device=device)] + [None]*d
    Jz_right = [None]*d + [tn.zeros((0, 1), dtype=tn.int64, device=device)]

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
        'Ps_left_lu': [_factorize_projection(tn.ones((1, 1), dtype=dtype, device=device))] + [None] * d,
        'Ps_right_lu': [None] * d + [_factorize_projection(tn.ones((1, 1), dtype=dtype, device=device))],
        'Ps_z_left_lu': [_factorize_projection(tn.ones((1, 1), dtype=dtype, device=device))] + [None] * d,
        'Ps_z_right_lu': [None] * d + [_factorize_projection(tn.ones((1, 1), dtype=dtype, device=device))],
        'normx': np.ones((d-1)),
        'enable_x_bck': False
    }

    callbacks = AmenCrossCallbacks(function, eval_mv, x, N_list, dtype, device)

    rz = [1]+(d-1)*[kick]+[1]

    x_cores = [tn.reshape(c, [c.shape[0], 1, c.shape[1], c.shape[2]]) for c in cores]

    x_cores, rx = amen_approx(M_dummy, N_list, d, x_cores, rx, rz, state_dict, callbacks,
                              nswp=nswp, eps=eps, rmax=rmax, kickrank=kick, kick2=kick2, verbose=verbose)

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
    Idx = [tn.zeros((1, 0), dtype=tn.int64, device=device)]+(d-1)*[None] + \
        [tn.zeros((0, 1), dtype=tn.int64, device=device)]
    for k in range(d-1, 0, -1):

        tmp = tn.einsum('ijk,kl->ijl', cores[k], Rm)
        tmp = tn.reshape(tmp, [rank[k], -1]).t()
        core, Rmat = QR(tmp)

        rnew = min(N[k]*rank[k+1], rank[k])
        Jk = _maxvol(core)
        # print(Jk)
        tmp = _unravel_index(Jk[:rnew], (rank[k+1], N[k]), device)
        # if k==d-1:
        #    idx_new = tn.tensor(tmp[1].reshape([1,-1]))
        # else:
        idx_new = tn.vstack((tmp[1].reshape([1, -1]), Idx[k+1][:, tmp[0]]))

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
            eval_index = _build_two_core_eval_index(Idx[k], Idx[k+2], rank[k], N[k], N[k+1], rank[k+2], device)

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
            rnew = _rank_chop_torch(S, tn.linalg.norm(S)*eps/np.sqrt(d-1))+1
            rnew = min(S.shape[0], rnew)
            rnew = min(rmax, rnew)
            U = U[:, :rnew]
            S = S[:rnew]
            V = V[:rnew, :]
            # print('kkt new',tn.linalg.norm(supercore-U@tn.diag(S)@V))
            # kick the rank
            V = S[:, None] * V
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
            tmp = _unravel_index(idx[:rank[k+1]], (rank[k], N[k]), device)
            idx_new = tn.hstack((Idx[k][tmp[0], :], tmp[1].reshape([-1, 1])))
            Idx[k+1] = idx_new+0

        # right to left

        for k in range(d-2, -1, -1):
            if verbose:
                print('\tRL supercore %d,%d' % (k+1, k+2))
            eval_index = _build_two_core_eval_index(Idx[k], Idx[k+2], rank[k], N[k], N[k+1], rank[k+2], device)

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
            rnew = _rank_chop_torch(S, tn.linalg.norm(S)*eps/np.sqrt(d-1))+1
            rnew = min(S.shape[0], rnew)
            rnew = min(rmax, rnew)
            U = U[:, :rnew]
            S = S[:rnew]
            V = V[:rnew, :]

            # kick the rank
            U = U * S[None, :]
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
            tmp = _unravel_index(idx[:rank[k+1]], (N[k+1], rank[k+2]), device)
            idx_new = tn.vstack((tmp[0].reshape([1, -1]), Idx[k+2][:, tmp[1]]))
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
