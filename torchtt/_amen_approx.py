"""
AMEN approximation skeleton for tensor computations.
"""
import torch as tn
import torchtt
from torchtt._decomposition import QR, SVD, rl_orthogonal, rank_chop
import numpy as np
import datetime

class AmenCallbacks:
    """
    Interface for providing task-specific operations to the generic AMEN approximation skeleton.
    """
    def compute_z_bck(self, k, state_dict, x_cores, z_cores):
        raise NotImplementedError

    def update_phis_bck(self, k, state_dict, x_cores, z_cores, swp, last):
        raise NotImplementedError

    def compute_x_fwd(self, k, state_dict, x_cores, z_cores):
        raise NotImplementedError

    def compute_z_fwd(self, k, state_dict, x_cores, z_cores, u, v):
        raise NotImplementedError

    def compute_enrichment(self, k, state_dict, x_cores, z_cores, u, v):
        raise NotImplementedError

    def update_phis_fwd(self, k, state_dict, x_cores, z_cores, swp, last):
        raise NotImplementedError

    def pre_orthogonalize_bck(self, k, state_dict, x_cores, z_cores, swp, last):
        """Optional hook to execute before right-to-left core orthogonalization"""
        pass

    def compute_norm_bck(self, k, state_dict, x_cores, z_cores, swp, last):
        """Optional hook to extract current_norm during right-to-left sweep"""
        return None

    def pre_orthogonalize_fwd(self, k, state_dict, x_cores, z_cores, swp, last, v):
        """Optional hook to execute before left-to-right core orthogonalization"""
        pass


def amen_approx(M, N, d, x_cores, rx, rz, state_dict, callbacks: AmenCallbacks,
                nswp=22, eps=1e-10, rmax=1024, kickrank=4, kick2=0, verbose=False):
    """
    General Alternating Minimal Energy (AMEN) approximation algorithm.

    Args:
        M (list[int]): row mode dimensions.
        N (list[int]): column mode dimensions.
        d (int): number of dimensions.
        x_cores (list[torch.tensor]): initial approximation cores.
        rx (list[int]): initial approximation ranks.
        rz (list[int]): initial enrichment ranks.
        state_dict (dict): state variables like Phis, norms, and evaluation grids.
        callbacks (AmenCallbacks): callback interface for task-specific steps.
        nswp (int): number of sweeps.
        eps (float): tolerance.
        rmax (list[int] or int): maximum rank allowed.
        kickrank (int): rank enlargement for error approximation.
        kick2 (int): additional random enrichment for Z.
        verbose (bool): output debug info.
    """
    dtype = x_cores[0].dtype
    device = x_cores[0].device

    if isinstance(rmax, int):
        rmax = [1] + (d-1) * [rmax] + [1]

    z_tt = torchtt.random([(m, n) for m, n in zip(M, N)], rz, dtype, device=device)
    z_cores = [tn.reshape(c, [c.shape[0], -1, c.shape[-1]]) for c in z_tt.cores]
    z_cores, rz = rl_orthogonal(z_cores, rz, False)
    z_cores = [tn.reshape(c, [c.shape[0], m, -1, c.shape[-1]]) for c, m in zip(z_cores, M)]

    last = False
    normx = np.ones((d-1))
    state_dict['normx'] = normx

    if verbose:
        print('Starting AMEn approximation with:\n\tepsilon: %g\n\tsweeps: %d\n' % (eps, nswp))

    for swp in range(nswp):
        if verbose:
            print('\nStarting sweep %d %s...' % (swp+1, "(last one) " if last else ""))
            tme_sweep = datetime.datetime.now()

        # ---------------- Right-to-Left Sweep ----------------
        for k in range(d-1, 0, -1):
            if not last:
                if swp > 0:
                    cz_new = callbacks.compute_z_bck(k, state_dict, x_cores, z_cores)
                    _, _, vz = SVD(tn.reshape(cz_new, [cz_new.shape[0], -1]))
                    cz_new = vz[:min(kickrank, vz.shape[0]), :].t()
                    if k < d-1:
                        cz_new = tn.cat((cz_new, tn.randn((cz_new.shape[0], kick2), dtype=dtype, device=device)), 1)
                else:
                    cz_new = tn.reshape(z_cores[k], [rz[k], -1]).t()

                qz, _ = QR(cz_new)
                rz[k] = qz.shape[1]
                z_cores[k] = tn.reshape(qz.t(), [rz[k], M[k], N[k], rz[k+1]])

            # pre_orthogonalize hook
            callbacks.pre_orthogonalize_bck(k, state_dict, x_cores, z_cores, swp, last)

            compute_x_bck = getattr(callbacks, 'compute_x_bck', None)
            use_x_bck = callable(compute_x_bck) and state_dict.get('enable_x_bck', True)
            if use_x_bck:
                solution_now, _ = callbacks.compute_x_bck(k, state_dict, x_cores, z_cores)
                core = tn.reshape(solution_now, [rx[k], M[k]*N[k]*rx[k+1]]).t()
                Qmat, Rmat = QR(core)
            else:
                core = tn.reshape(x_cores[k], [rx[k], M[k]*N[k]*rx[k+1]]).t()
                Qmat, Rmat = QR(core)

            core_prev = tn.einsum('ijlk,km->ijlm', x_cores[k-1], Rmat.T)
            rx[k] = Qmat.shape[1]

            x_cores[k] = tn.reshape(Qmat.t(), [rx[k], M[k], N[k], rx[k+1]])
            x_cores[k-1] = core_prev[:]

            current_norm = callbacks.compute_norm_bck(k, state_dict, x_cores, z_cores, swp, last)
            
            if current_norm is not None:
                normx_val = current_norm.item() if isinstance(current_norm, tn.Tensor) else current_norm
                normx[k-1] = normx[k-1] * normx_val
                state_dict['normx'] = normx

            callbacks.update_phis_bck(k, state_dict, x_cores, z_cores, swp, last)

        max_dx = 0

        # ---------------- Left-to-Right Sweep ----------------
        for k in range(d):
            if verbose:
                print('\tCore', k)
            previous_solution = x_cores[k]

            solution_now, norm_solution = callbacks.compute_x_fwd(k, state_dict, x_cores, z_cores)
            if solution_now.shape != previous_solution.shape:
                solution_now = tn.reshape(solution_now, previous_solution.shape)

            dx = tn.linalg.norm(solution_now - previous_solution) / tn.linalg.norm(solution_now)
            if verbose:
                print('\t\tdx = %g' % (dx))

            max_dx = max(dx, max_dx)

            solution_now = tn.reshape(solution_now, [rx[k]*M[k]*N[k], rx[k+1]])

            if k < d-1:
                u, s, v = SVD(solution_now)
                r = rank_chop(s.cpu().numpy(), (norm_solution.cpu() * eps / (d**(0.5 if last else 1.5))).numpy())
                r = min([r, tn.numel(s), rmax[k+1]])
            else:
                u, v = QR(solution_now)
                r = u.shape[1]
                s = tn.ones(r, dtype=dtype, device=device)

            u = u[:, :r]
            v = v[:, :r] @ tn.diag(s[:r])

            if not last:
                cz_new = callbacks.compute_z_fwd(k, state_dict, x_cores, z_cores, u, v)
                uz, _, _ = SVD(tn.reshape(cz_new, [rz[k]*M[k]*N[k], rz[k+1]]))
                cz_new = uz[:, :min(kickrank, uz.shape[1])]
                if k < d-1:
                    cz_new = tn.cat((cz_new, tn.randn((cz_new.shape[0], kick2), dtype=dtype, device=device)), 1)

                qz, _ = QR(cz_new)
                rz[k+1] = qz.shape[1]
                z_cores[k] = tn.reshape(qz, [rz[k], M[k], N[k], rz[k+1]])

            if k < d-1:
                callbacks.pre_orthogonalize_fwd(k, state_dict, x_cores, z_cores, swp, last, v)

                if not last:
                    uk = callbacks.compute_enrichment(k, state_dict, x_cores, z_cores, u, v)
                    u, Rmat = QR(tn.cat((u, tn.reshape(uk, [u.shape[0], -1])), 1))
                    r_add = uk.shape[-1]
                    v = tn.cat((v, tn.zeros([rx[k+1], r_add], dtype=dtype, device=device)), 1)
                    v = v @ Rmat.t()

                r = u.shape[1]
                v = tn.einsum('ji,jklm->iklm', v, x_cores[k+1])



                x_cores[k] = tn.reshape(u, [rx[k], M[k], N[k], r])
                x_cores[k+1] = tn.reshape(v, [r, M[k+1], N[k+1], rx[k+2]])
                rx[k+1] = r

                callbacks.update_phis_fwd(k, state_dict, x_cores, z_cores, swp, last)

            else:
                x_cores[k] = tn.reshape(u @ tn.diag(s[:r]) @ v[:r, :].t(), [rx[k], M[k], N[k], rx[k+1]])

        if verbose:
            print('Solution rank is', rx)
            print('Maxdx ', max_dx)
            tme_sweep = datetime.datetime.now() - tme_sweep
            print('Time ', tme_sweep)

        if last:
            break

        if max_dx < eps:
            last = True

    # Apply cumulative norm factor back to cores
    # normx_scalar = np.exp(np.sum(np.log(normx))/d)
    # for k in range(d):
    #     x_cores[k] = x_cores[k] * normx_scalar

    return x_cores, rx
