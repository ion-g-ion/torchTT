#pragma once
#include "define.h"
#include "matvecs.h"

/**
 * @brief Restarted GMRES for the local systems of the AMEn solver, started from zero.
 *
 * The Krylov basis is stored in the rows of one preallocated tensor and every new vector is
 * orthogonalised with two passes of classical Gram-Schmidt (CGS2), i.e. with matrix-vector products.
 * An iteration therefore launches a fixed number of kernels and copies only the new column of the
 * Hessenberg matrix to the host, where the Givens rotations and the stopping test are done.
 *
 * @param[out] solution the solution, same shape and dtype as rhs.
 * @param[out] flag 1 if the tolerance was reached, 0 otherwise.
 * @param[out] nit the total number of iterations.
 * @param[in] Op the operator. The iterations use its dtype.
 * @param[in] rhs the right-hand side.
 * @param[in] max_iters the number of iterations before a restart.
 * @param[in] tol the absolute tolerance for the norm of the residual.
 * @param[in] resets the maximum number of restarts.
 */
inline void gmres(at::Tensor &solution, int &flag, int &nit, const AMENsolveMV &Op, const at::Tensor &rhs, int64_t max_iters, double tol, int64_t resets){
    auto b = rhs.reshape({-1}).to(Op.dtype());
    int64_t m = max_iters;
    auto x = at::zeros_like(b);
    auto V = at::empty({m+1, b.sizes()[0]}, b.options());

    // Hessenberg matrix (column major, columns of length m+1), rotations and rhs of the least squares problem
    std::vector<double> H((m+1)*m), cs(m), sn(m), g(m+1);

    flag = 0;
    nit = 0;
    for(int64_t rs = 0; rs < resets; ++rs){
        auto res = rs == 0 ? b : b - Op.matvec(x);
        double beta = torch::norm(res).item<double>();
        if(beta == 0){
            flag = 1;
            break;
        }
        V.select(0, 0).copy_(res / beta);
        std::fill(g.begin(), g.end(), 0.0);
        g[0] = beta;

        int64_t k;
        for(k = 0; k < m; ++k){
            auto w = Op.matvec(V.select(0, k));
            auto Vk = V.narrow(0, 0, k+1);
            auto h = at::mv(Vk, w);
            w.addmv_(Vk.t(), h, 1, -1);
            auto h2 = at::mv(Vk, w);
            w.addmv_(Vk.t(), h2, 1, -1);
            h += h2;
            auto norm_w = torch::norm(w);
            V.select(0, k+1).copy_(w / norm_w);

            // the only synchronisation of the iteration
            auto col_h = at::cat({h, norm_w.view({1})}).to(at::kCPU, at::kDouble);
            double *col = H.data() + k*(m+1);
            std::copy(col_h.data_ptr<double>(), col_h.data_ptr<double>() + k + 2, col);

            for(int64_t i = 0; i < k; ++i){
                double tmp = cs[i]*col[i] + sn[i]*col[i+1];
                col[i+1] = -sn[i]*col[i] + cs[i]*col[i+1];
                col[i] = tmp;
            }
            double den = std::hypot(col[k], col[k+1]);
            bool breakdown = col[k+1] == 0;
            cs[k] = col[k]/den;
            sn[k] = col[k+1]/den;
            col[k] = den;
            col[k+1] = 0.0;
            g[k+1] = -sn[k]*g[k];
            g[k] = cs[k]*g[k];
            ++nit;
            if(std::abs(g[k+1]) <= tol || breakdown){
                flag = 1;
                ++k;
                break;
            }
        }

        // solve the triangular system H[:k,:k] y = g[:k] and update x
        std::vector<double> y(k);
        for(int64_t i = k-1; i >= 0; --i){
            double tmp = g[i];
            for(int64_t j = i+1; j < k; ++j)
                tmp -= H[j*(m+1)+i]*y[j];
            y[i] = tmp/H[i*(m+1)+i];
        }
        auto y_t = at::from_blob(y.data(), {k}, at::kDouble).to(b.options());
        x.addmv_(V.narrow(0, 0, k).t(), y_t);

        if(flag == 1)
            break;
    }

    solution = x.to(rhs.scalar_type()).reshape(rhs.sizes());
}
