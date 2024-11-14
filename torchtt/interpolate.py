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


def _LU(M):
    """
    Perform an LU decomposition and returns L, U and a permutation vector P. 

    Args:
        M (torch.tensor): [description]

    Returns:
        tuple[torch.tensor,torch.tensor,torch.tensor]: L, U, P
    """
    LU, P = tn.linalg.lu_factor(M)
    P, L, U = tn.lu_unpack(LU, P)  # P transpose or not transpose?
    P = P@tn.reshape(tn.arange(P.shape[1],
                     dtype=P.dtype, device=P.device), [-1, 1])
    # P = tn.reshape(tn.arange(P.shape[1],dtype=P.dtype,device=P.device),[1,-1]) @ P

    return L, U, tn.squeeze(P).to(tn.int64)


def _max_matrix(M):

    values, indices = M.flatten().topk(1)
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


def function_interpolate(function, x, eps=1e-9, start_tens=None, nswp=20, kick=2, dtype=tn.float64, rmax=sys.maxsize, verbose=False):
    """
    Appication of a nonlinear function on a tensor in the TT format (using DMRG). Two cases are distinguished:

    * Univariate interpoaltion:

    Let :math:`f:\\mathbb{R}\\rightarrow\\mathbb{R}` be a function and :math:`\\mathsf{x}\\in\\mathbb{R}^{N_1\\times\\cdots\\times N_d}` be a tensor with a known TT approximation.
    The goal is to determine the TT approximation of :math:`\\mathsf{y}_{i_1...i_d}=f(\\mathsf{x}_{i_1...i_d})` within a prescribed relative accuracy `eps`. 

    * Multivariate interpolation

    Let :math:`f:\\mathbb{R}\\rightarrow\\mathbb{R}` be a function and :math:`\\mathsf{x}^{(1)},...,\\mathsf{x}^{(d)}\\in\\mathbb{R}^{N_1\\times\\cdots\\times N_d}` be tensors with a known TT approximation. The goal is to determine the TT approximation of :math:`\\mathsf{y}_{i_1...i_d}=f(\\mathsf{x}_{i_1...i_d}^{(1)},...,\\mathsf{x}^{(d)})_{i_1...i_d}` within a prescribed relative accuracy `eps`.


    Example:

        * Univariate interpolation:

        .. code-block:: python

            func = lambda t: torch.log(t)
            y = tntt.interpolate.function_interpolate(func, x, 1e-9) # the tensor x is chosen such that y has an afforbable low rank structure

        * Multivariate interpolation:

        .. code-block:: python

            xs = tntt.meshgrid([tn.arange(0,n,dtype=torch.float64) for n in N])
            func = lambda x: 1/(2+tn.sum(x,1).to(dtype=torch.float64))
            z = tntt.interpolate.function_interpolate(func, xs)


    Args:
        function (Callable): function handle. If the argument `x` is a `torchtt.TT` instance, the the function handle has to be appliable elementwise on torch tensors.
                             If a list is passed as `x`, the function handle takes as argument a :math:`M\times d` torch.tensor and every of the :math:`M` lines corresponds to an evaluation of the function :math:`f` at a certain tensor entry. The function handle returns a torch tensor of length M.
        x (torchtt.TT or list[torchtt.TT]): the argument/arguments of the function.
        eps (float, optional): the relative accuracy. Defaults to 1e-9.
        start_tens (torchtt.TT, optional): initial approximation of the output tensor (None coresponds to random initialization). Defaults to None.
        nswp (int, optional): number of iterations. Defaults to 20.
        kick (int, optional): enrichment rank. Defaults to 2.
        dtype (torch.dtype, optional): the dtype of the result. Defaults to tn.float64.
        rmax (int, optional): the maximum rank. Defaults to the maximum possible integer.
        verbose (bool, optional): display debug information to the console. Defaults to False.

    Returns:
        torchtt.TT: the result.
    """

    if isinstance(x, list) or isinstance(x, tuple):
        eval_mv = True
        N = x[0].N
    else:
        eval_mv = False
        N = x.N
    device = None

    if not eval_mv and len(N) == 1:
        return torchtt.TT(function(x.full())).to(device)

    if eval_mv and len(N) == 1:
        return torchtt.TT(function(x[0].full())).to(device)

    d = len(N)

    # random init of the tensor
    if start_tens == None:
        rank_init = 2
        cores = torchtt.random(N, rank_init, dtype, device).cores
        rank = [1]+[rank_init]*(d-1)+[1]
    else:
        rank = start_tens.R.copy()
        cores = [c+0 for c in start_tens.cores]
    # cores = (ones(N,dtype=dtype)).cores

    cores, rank = rl_orthogonal(cores, rank, False)
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
        tmp = np.unravel_index(Jk[:rnew], (rank[k+1], N[k]))
        # if k==d-1:
        #    idx_new = tn.tensor(tmp[1].reshape([1,-1]))
        # else:
        idx_new = tn.tensor(
            np.vstack((tmp[1].reshape([1, -1]), Idx[k+1][:, tmp[0]])))

        Idx[k] = idx_new+0

        Rm = core[Jk, :]

        core = tn.linalg.solve(Rm.T, core.T)
        Rm = (Rm@Rmat).t()
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

            if eval_mv:
                ev = tn.zeros((eval_index.shape[0], 0), dtype=dtype)
                for j in range(d):
                    core = x[j].cores[0][0, eval_index[:, 0], :]
                    for i in range(1, d):
                        core = tn.einsum('ij,jil->il', core,
                                         x[j].cores[i][:, eval_index[:, i], :])
                    core = tn.reshape(core[..., 0], [-1, 1])
                    ev = tn.hstack((ev, core))
                supercore = tn.reshape(
                    function(ev), [rank[k], N[k], N[k+1], rank[k+2]])
                n_eval += core.shape[0]
            else:
                core = x.cores[0][0, eval_index[:, 0], :]
                for i in range(1, d):
                    core = tn.einsum('ij,jil->il', core,
                                     x.cores[i][:, eval_index[:, i], :])
                core = core[..., 0]
                supercore = tn.reshape(
                    function(core), [rank[k], N[k], N[k+1], rank[k+2]])
                n_eval += core.shape[0]

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
            radd = Rtemp.shape[1] - rnew
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

            if eval_mv:
                ev = tn.zeros((eval_index.shape[0], 0), dtype=dtype)
                for j in range(d):
                    core = x[j].cores[0][0, eval_index[:, 0], :]
                    for i in range(1, d):
                        core = tn.einsum('ij,jil->il', core,
                                         x[j].cores[i][:, eval_index[:, i], :])
                    core = tn.reshape(core[..., 0], [-1, 1])
                    ev = tn.hstack((ev, core))
                supercore = tn.reshape(
                    function(ev), [rank[k], N[k], N[k+1], rank[k+2]])
                n_eval += core.shape[0]
            else:
                core = x.cores[0][0, eval_index[:, 0], :]
                for i in range(1, d):
                    core = tn.einsum('ij,jil->il', core,
                                     x.cores[i][:, eval_index[:, i], :])
                core = core[..., 0]
                supercore = tn.reshape(
                    function(core), [rank[k], N[k], N[k+1], rank[k+2]])
                n_eval += core.shape[0]

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
            # print('u before', U.shape)
            U = U @ tn.diag(S)
            VK = tn.randn((kick, V.shape[1]), dtype=dtype, device=device)
            # print('V enrich', V.shape)
            V, Rtemp = QR(tn.cat((V, VK), 0).t())
            radd = Rtemp.shape[1] - rnew
            # print('V after QR',V.shape,Rtemp.shape,radd)
            if radd > 0:
                U = tn.cat(
                    (U, tn.zeros((U.shape[0], radd), dtype=dtype, device=device)), 1)
                U = U @ Rtemp.T
                V = V.t()

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
            U = U @ tn.diag(S)
            VK = tn.randn((kick, V.shape[1]), dtype=dtype, device=device)
            V, Rtemp = QR(tn.cat((V, VK), 0).t())
            radd = V.shape[1] - rnew
            if radd > 0:
                U = tn.cat(
                    (U, tn.zeros((U.shape[0], radd), dtype=dtype, device=device)), 1)
                U = U @ Rtemp.T
                V = V.t()

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
