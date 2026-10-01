#include "define.h"
#include "ortho.h"
#include <cmath>
#include "matvecs.h"
#include "gmres.h"
/**
 * @brief Compute thelocal matvec product in the AMEn: lsr,smnS,LSR,rnR->lmL
 * 
 * @param[in] Phi_right The right interface
 * @param[in] Phi_left The left interface
 * @param[in] coreA The corre of the TT operator
 * @param[in] core The core vector
 * @return at::Tensor 
 */
at::Tensor local_product(at::Tensor &Phi_right, at::Tensor &Phi_left, at::Tensor &coreA, at::Tensor &core){
    
    
    // rnR,lsr->nRls
    auto tmp1 = at::tensordot(core, Phi_left, {0}, {2});
    // nRls,smnS->RlmS
    auto tmp2 = at::tensordot(tmp1, coreA, {0,3},{2,0});
    // RlmS,LSR->lmL 
    return at::tensordot(tmp2, Phi_right, {0,3}, {2,1});

} 

/**
 * @brief Compute the phi backwards for the form dot(left,A @ right)
 * 
 * @param[in] Phi_now the current phi. Has shape r1_k+1 x R_k+1 x r2_k+1
 * @param[in] core_left the core on the left. Has shape r1_k x N_k x r1_k+1 
 * @param[in] core_A the core of the matrix. Has shape  R_k x N_k x N_k x R_k
 * @param[in] core_right the core to the right. Has shape r2_k x N_k x r2_k+1 
 * @return at::Tensor the following phi (backward). Has shape r1_k x R_k x r2_k
 */
at::Tensor compute_phi_bck_A(at::Tensor &Phi_now, at::Tensor &core_left, at::Tensor &core_A, at::Tensor &core_right){    
    at::Tensor Phi;
    // 5           GEMM          lML,LSR->lMSR                    sMNS,rNR,lMSR->lsr
    // 6           TDOT        lMSR,sMNS->lRsN                         rNR,lRsN->lsr
    // 5           TDOT          lRsN,rNR->lsr                              lsr->lsr
    Phi = at::tensordot(core_left, Phi_now, {2}, {0});
    Phi = at::tensordot(Phi, core_A, {1,2}, {1,3}); // lRsN
    return at::tensordot(Phi, core_right, {1,3}, {2,1});
}

/**
 * @brief Compute the phi forward for the form dot(left,A @ right)
 * 
 * @param[in] Phi_now the current phi. Has shape r1_k x R_k x r2_k 
 * @param[in] core_left the core on the left. Has shape r1_k x N_k x r1_k+1 
 * @param[in] core_A the core of the matrix. Has shape  R_k x N_k x N_k x R_k
 * @param[in] core_right the core to the right. Has shape r2_k x N_k x r2_k+1 
 * @return at::Tensor the following phi (backward). Has shape r1_k+1 x R_k+1 x r2_k+1 
 */
at::Tensor compute_phi_fwd_A(at::Tensor &Phi_now, at::Tensor &core_left, at::Tensor &core_A, at::Tensor &core_right){
    at::Tensor Phi_next;
//    5           GEMM          lML,lsr->MLsr                    sMNS,rNR,MLsr->LSR
//    6           TDOT        MLsr,sMNS->LrNS                         rNR,LrNS->LSR
//    5           TDOT          LrNS,rNR->LSR                              LSR->LSR
    Phi_next = at::tensordot(core_left, Phi_now, {0}, {0}); // MLsr
    Phi_next = at::tensordot(Phi_next, core_A, {0,2}, {1,0}); // LrNS
    Phi_next = at::tensordot(Phi_next, core_right, {1,2}, {0,1});
    return Phi_next;
}

/**
 * @brief Compute the backward phi `BR,bnB,rnR->br`
 * 
 * @param[in] Phi_now the current Phi. Has shape rb_k+1 x r_k+1
 * @param[in] core_b the core of the rhs. Has shape rb_k x N_k x rb_k+1
 * @param[in] core the current core. Has shape r_k x N_k x r_k+1
 * @return at::Tensor Has shape rb_k x r_k
 */
at::Tensor compute_phi_bck_rhs(at::Tensor &Phi_now, at::Tensor &core_b, at::Tensor &core){
    at::Tensor Phi;
    Phi = at::tensordot(core_b, Phi_now, {2}, {0});
    Phi = at::tensordot(Phi, core, {1,2}, {1,2});
    return Phi;
}

/**
 * @brief Compute the forward phi `br,bnB,rnR->BR`
 * 
 * @param[in] Phi_now the current Phi. Has shape  rb_k x r_k
 * @param[in] core_rhs the core of the rhs. Has shape rb_k x N_k+1 x rb_k+1
 * @param[in] core the current core. Has shape r_k x N_k x r_k+1
 * @return at::Tensor Has shape rb_k+1 x r_k+1
 */
at::Tensor compute_phi_fwd_rhs(at::Tensor &Phi_now, at::Tensor &core_rhs, at::Tensor &core){

    at::Tensor Phi_next = at::tensordot(Phi_now, core_rhs, {0}, {0});
    Phi_next = at::tensordot(Phi_next, core, {0,1}, {0,1});
    return Phi_next;
}

/**
 * @brief Copy scalar tensors to the host as doubles with a single device synchronisation.
 *
 * @param[in] scalars the 0-dimensional tensors.
 * @return std::vector<double> the values.
 */
std::vector<double> to_host(std::initializer_list<at::Tensor> scalars){
    at::Tensor values = at::stack(std::vector<at::Tensor>(scalars)).to(at::kCPU, at::kDouble);
    return std::vector<double>(values.data_ptr<double>(), values.data_ptr<double>() + values.numel());
}

/**
 * @brief AMEn solve implementation in C++.
 * 
 * @param[in] A_cores 
 * @param[in] b_cores 
 * @param[in] x0_cores 
 * @param[in] N 
 * @param[in] rA 
 * @param[in] rb 
 * @param[in] r_x0 
 * @param[in] nswp 
 * @param[in] eps 
 * @param[in] rmax 
 * @param[in] max_full 
 * @param[in] kickrank 
 * @param[in] kick2 
 * @param[in] local_iterations 
 * @param[in] resets 
 * @param[in] verbose 
 * @param[in] preconditioner
 * @param[in] use_single_precision run GMRES in single precision. The residuals are computed in the precision of the input.
 * @return std::vector<at::Tensor> TT cores of the solution
 */
std::vector<at::Tensor> amen_solve(
                        std::vector<at::Tensor> &A_cores, 
                        std::vector<at::Tensor> &b_cores,
                        std::vector<at::Tensor> &x0_cores,
                        std::vector<uint64_t> N,
                        std::vector<uint64_t> rA,
                        std::vector<uint64_t> rb,
                        std::vector<uint64_t> r_x0,
                        uint64_t nswp,
                        double eps,
                        uint64_t rmax,
                        uint64_t max_full,
                        uint64_t kickrank,
                        uint64_t kick2, 
                        uint64_t local_iterations,
                        uint64_t resets,
                        bool verbose,
                        int preconditioner,
                        bool use_single_precision)
{

    torch::NoGradGuard no_grad;

    TORCH_CHECK(!A_cores[0].is_complex() && !b_cores[0].is_complex(), "The C++ AMEn solver does not support complex tensors.");

    if(verbose)
    {
        std::cout << "Starting AMEn solve with:";
        std::cout << "\n\tepsilon              : " << eps;
        std::cout << "\n\tsweeps               : " << nswp;
        std::cout << "\n\tlocal iterations     : " << local_iterations;
        std::cout << "\n\tresets               : " << resets;
        char prec_char = (preconditioner == 0 ? 'N' : ( preconditioner == 1 ? 'C' : 'R'));
        std::cout << "\n\tlocal preconditioner : " << prec_char;
        std::cout << std::endl << std::endl;
    } 
    auto options = A_cores[0].options();
    at::ScalarType dtype = A_cores[0].scalar_type();
    at::ScalarType dtype_gmres = (use_single_precision && dtype == at::kDouble) ? at::kFloat : dtype;
    uint64_t d = N.size();
    std::vector<at::Tensor> x_cores;
    
    if(x0_cores.size() == 0){
        for(int i = 0; i < d; i++)
            x_cores.push_back(torch::ones({1,N[i],1}, options));
    }
    else
        x_cores = x0_cores;
    std::vector<uint64_t> rx = r_x0;

    std::vector<uint64_t> rz(d+1);
    rz[0] = 1;
    rz[d] = 1;
    for(int i=1;i<d;i++) 
        rz[i] = kickrank+kick2;
    std::vector<at::Tensor> z_cores(d);
    for(int i=0;i<d;i++) 
        z_cores[i] = torch::randn({rz[i], N[i], rz[i + 1]}, options);

    rl_orthogonal_this(z_cores, N, rz);

    // the interface matrices
    std::vector<at::Tensor> Phiz(d+1);
    std::vector<at::Tensor> Phiz_b(d+1); 
    std::vector<at::Tensor> Phis(d+1);
    std::vector<at::Tensor> Phis_b(d+1);
    Phiz[0] = at::ones({1,1,1}, options);
    Phiz_b[0] = at::ones({1,1}, options);
    Phis[0] = at::ones({1,1,1}, options);
    Phis_b[0] = at::ones({1,1}, options);
    Phiz[d] = at::ones({1,1,1}, options);
    Phiz_b[d] = at::ones({1,1}, options);
    Phis[d] = at::ones({1,1,1}, options);
    Phis_b[d] = at::ones({1,1}, options);

    double *normA = new double[d-1];
    double *normb = new double[d-1];
    double *normx = new double[d-1];
    for(int k=0;k<d-1;++k){
        normA[k] = 1.0;
        normb[k] = 1.0;
        normx[k] = 1.0;
    }
    double nrmsc = 1.0;
    double damp = 2.0;

    bool last = false;

    std::chrono::time_point<std::chrono::high_resolution_clock> tme_swp, tme_total;
    if(verbose)
        tme_total = std::chrono::high_resolution_clock::now();
    int swp;
    for(swp=0;swp<nswp;swp++){

        if(verbose){
            if(last)
                std::cout<<std::endl<<"Starting sweep " << swp+1 << " (last one)" << std::endl;
            else
                std::cout<<std::endl<<"Starting sweep " << swp+1 <<  std::endl;
            tme_swp = std::chrono::high_resolution_clock::now();
        }

        for(int k=d-1;k>0;k--){
            if(!last){
                at::Tensor cz_new;
                if(swp>0){
                    at::Tensor czA = local_product(Phiz[k+1], Phiz[k], A_cores[k], x_cores[k]);
                    at::Tensor czy = at::tensordot(Phiz_b[k], b_cores[k], {0}, {0});
                    czy = at::tensordot(czy, Phiz_b[k+1], {2}, {0});
                    czy *= nrmsc;
                    czy -= czA;
                    std::tuple <at::Tensor, at::Tensor, at::Tensor> USV = svd_thin(czy.reshape({czy.sizes()[0],-1}));
                    uint64_t temp = kickrank < std::get<2>(USV).sizes()[0] ? kickrank :  std::get<2>(USV).sizes()[0];
                    cz_new = std::get<2>(USV).index({ torch::indexing::Slice(0, temp), torch::indexing::Ellipsis}).t();
                    if(k < d-1)
                        cz_new = at::cat({cz_new,torch::randn({cz_new.sizes()[0],  kick2}, options)}, 1);
                }
                else{
                    cz_new = z_cores[k].reshape({rz[k],-1}).t();
                }

                at::Tensor Qz;
                std::tie(Qz, std::ignore) = at::linalg_qr(cz_new);
                rz[k] = Qz.sizes()[1];
                z_cores[k] = (Qz.t()).reshape({rz[k], N[k], rz[k+1]});
            }


            
            if(swp>0)
                nrmsc = nrmsc * normA[k-1] * normx[k-1] / normb[k-1];

            auto core = x_cores[k].reshape({rx[k],N[k]*rx[k+1]}).t();

            std::tuple<at::Tensor, at::Tensor> QR = at::linalg_qr(core);
            
            auto core_prev = at::tensordot(x_cores[k-1], std::get<1>(QR).t(), {2}, {0});
            rx[k] = std::get<0>(QR).sizes()[1];

            x_cores[k] = (std::get<0>(QR).t()).reshape({rx[k], N[k], rx[k+1]}).clone();

            Phis[k] = compute_phi_bck_A(Phis[k+1],x_cores[k],A_cores[k],x_cores[k]);
            Phis_b[k] = compute_phi_bck_rhs(Phis_b[k+1],b_cores[k],x_cores[k]);

            // the three norms with one synchronisation
            std::vector<double> norms = to_host({torch::norm(core_prev), torch::norm(Phis[k]), torch::norm(Phis_b[k])});

            double current_norm = norms[0] > 0 ? norms[0] : 1.0;
            x_cores[k-1] = core_prev / current_norm;
            normx[k-1] = normx[k-1] * current_norm;

            normA[k-1] = norms[1] > 0 ? norms[1] : 1.0;
            Phis[k] = Phis[k] / normA[k-1];

            normb[k-1] = norms[2] > 0 ? norms[2] : 1.0;
            Phis_b[k] = Phis_b[k] / normb[k-1];
            
            // norm correction
            nrmsc = nrmsc * normb[k-1] / (normA[k-1] * normx[k-1]);

            // compute phis_z
            if(!last){
                Phiz[k] = compute_phi_bck_A(Phiz[k+1], z_cores[k], A_cores[k], x_cores[k]) / normA[k-1];
                Phiz_b[k] = compute_phi_bck_rhs(Phiz_b[k+1], b_cores[k], z_cores[k]) / normb[k-1];
            }
        }
        double max_res = 0;

        for(int k = 0; k<d ; k++){
            if(verbose)
                std::cout<<"\tCore "<<k<<std::endl;

            at::Tensor previous_solution = x_cores[k].reshape({-1,1});

            // assemble rhs
            at::Tensor rhs = at::tensordot(Phis_b[k], b_cores[k] * nrmsc, {0}, {0});
            rhs = at::tensordot(rhs, Phis_b[k+1], {2}, {0}).reshape({-1,1});

            double norm_rhs = torch::norm(rhs).item<double>();
            bool zero_rhs = norm_rhs == 0;
            norm_rhs = norm_rhs > 0 ? norm_rhs : 1.0;

            // residuals 
            double real_tol = (eps/std::sqrt(d))/damp;

            // direct local solver or iterative
            bool use_full = rx[k]*N[k]*rx[k+1] < max_full;
            at::Tensor solution_now;
            double res_old, res_new;

            at::Tensor B;
            // the local operator in the precision of the input, used for the residuals
            std::optional<AMENsolveMV> Op;

            if(use_full){
                if(verbose) 
                    std::cout << "\t\tChoosing direct solver (local size " << rx[k]*N[k]*rx[k+1] << ")..." << std::endl;
                auto Bp = at::tensordot(A_cores[k], Phis[k+1], {3}, {1}); // smnS,LSR->smnLR
                Bp = at::tensordot(Phis[k], Bp, {1}, {0}); // lsr,smnLR->lrmnLR
                B = Bp.permute({0,2,4,1,3,5}).reshape({rx[k]*N[k]*rx[k+1], rx[k]*N[k]*rx[k+1]});

                solution_now = at::linalg_solve(B, rhs);

                std::vector<double> res = to_host({torch::norm(at::linalg_matmul(B, previous_solution) - rhs), torch::norm(at::linalg_matmul(B, solution_now) - rhs)});
                res_old = res[0] / norm_rhs;
                res_new = res[1] / norm_rhs;
            }
            else{
                std::chrono::time_point<std::chrono::high_resolution_clock> tme_local;
                if(verbose) {
                    std::cout << "\t\tChoosing iterative solver (local size " << rx[k]*N[k]*rx[k+1] << ")..." <<std::endl;
                    tme_local = std::chrono::high_resolution_clock::now();
                }
                bool same_dtype = dtype_gmres == dtype;
                Op.emplace(Phis[k], Phis[k+1], A_cores[k], rx[k], N[k], rx[k+1], same_dtype ? preconditioner : NO_PREC, dtype);

                auto drhs = rhs - Op->matvec(previous_solution, false);

                int flag;
                int nit;

                if(zero_rhs){
                    solution_now = at::zeros_like(previous_solution);
                    flag = 1;
                    nit = 0;
                }
                else{
                    // GMRES solves for the correction, the tolerance is the absolute residual wanted for the local system
                    double tol = real_tol * norm_rhs;
                    if(same_dtype){
                        gmres(solution_now, flag, nit, *Op, drhs, local_iterations, tol, resets);
                        if(preconditioner!=NO_PREC)
                            solution_now = Op->apply_prec(solution_now);
                    }
                    else{
                        AMENsolveMV Op_gmres(Phis[k], Phis[k+1], A_cores[k], rx[k], N[k], rx[k+1], preconditioner, dtype_gmres);
                        gmres(solution_now, flag, nit, Op_gmres, drhs, local_iterations, tol, resets);
                        if(preconditioner!=NO_PREC)
                            solution_now = Op_gmres.apply_prec(solution_now.to(dtype_gmres)).to(dtype);
                    }
                    solution_now = solution_now + previous_solution;
                }

                // drhs is the residual of the previous solution
                std::vector<double> res = to_host({torch::norm(drhs), torch::norm(Op->matvec(solution_now, false)-rhs)});
                res_old = res[0]/norm_rhs;
                res_new = res[1]/norm_rhs;

                if(verbose){
                    std::cout<<"\t\tFinished with flag " << flag << " after " << nit << " iterations with relres " << res_new << " (from " << real_tol << ")" << std::endl;
                    auto duration = (double)(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()-tme_local)).count() /1000.0 ;
                    std::cout<<"\t\tTime needed " << duration << " ms" << std::endl;
                }
            }
            if(verbose && res_old/res_new < damp && res_new > real_tol)
                std::cout << "WARNING: residual increase. res_old " << res_old << ", res_new " << res_new << ", " << real_tol << std::endl;

            if(verbose){
                double norm_solution = torch::norm(solution_now).item<double>();
                auto dx = torch::norm(solution_now - previous_solution).item<double>() / (norm_solution > 0 ? norm_solution : 1.0);
                std::cout << "\t\tdx = " << dx << ", res_now = " << res_new << ", res_old = " << res_old << std::endl;
            }

            max_res = max_res < res_old ? res_old : max_res;

            solution_now = solution_now.reshape({rx[k]*N[k], rx[k+1]});

            at::Tensor u,s,v;
            int64_t r;

            if(k<d-1){
                std::tie(u,s,v) = svd_thin(solution_now);

                // On the final sweep the residual enrichment is disabled, so any
                // accuracy given away here can no longer be recovered. Budget the
                // truncation against what the local solve just achieved instead of
                // against the global tolerance. This does less rank reduction on
                // that sweep, so the returned ranks may be higher than before.
                double trunc_budget = last ? res_new*damp
                                           : (res_new > real_tol*damp ? res_new : real_tol*damp);

                // Going down from the full rank, find the first rank whose truncation exceeds the budget.
                // The residuals of a group of consecutive ranks are computed together, with one synchronisation.
                int64_t r_full = u.sizes()[1];
                int64_t group = std::max<int64_t>(1, std::min<int64_t>(4, (1 << 22) / solution_now.numel()));
                bool exceeded = false;
                r = r_full;
                while(r > 0 && !exceeded){
                    int64_t c = std::min(group, r);
                    // candidate j has rank r-j: the weights are s with the trailing values set to zero
                    auto ranks = at::arange(r, r - c, -1, s.options().dtype(at::kLong)).unsqueeze(1);
                    auto weights = s.unsqueeze(0) * (at::arange(r_full, ranks.options()).unsqueeze(0) < ranks).to(s.scalar_type());
                    auto candidates = at::matmul(u.unsqueeze(0) * weights.unsqueeze(1), v).reshape({c, -1});
                    auto products = use_full ? at::mm(candidates, B.t()) : Op->matvec(candidates, false);
                    at::Tensor res = at::linalg_vector_norm(products - rhs.reshape({1,-1}), 2, {1}).to(at::kCPU, at::kDouble);
                    const double *res_ptr = res.data_ptr<double>();
                    for(int64_t j = 0; j < c && !exceeded; ++j)
                        if(res_ptr[j] / norm_rhs > trunc_budget){
                            r -= j;
                            exceeded = true;
                        }
                    if(!exceeded)
                        r -= c;
                }
                ++r;

                r = std::min<int64_t>({r, r_full, (int64_t)rmax});

            }
            else{
                std::tie(u,v) = at::linalg_qr(solution_now);
                r = u.sizes()[1];
                s = torch::ones(r, options);
            }


            u = u.index({torch::indexing::Ellipsis, torch::indexing::Slice(0,r)});
            auto tmp1 = torch::diag(s.index({torch::indexing::Slice(0,r)}));
            auto tmp2 = v.index({torch::indexing::Slice(0,r), torch::indexing::Ellipsis});
            v = at::linalg_matmul(tmp1, tmp2).t();
            
            if(!last){
                at::Tensor czA, czy;
                at::Tensor tmp = at::linalg_matmul(u, v.t()).reshape({rx[k], N[k], rx[k+1]});
                czA = local_product(Phiz[k+1], Phiz[k], A_cores[k], tmp);
                czy = at::tensordot(Phiz_b[k], nrmsc * b_cores[k], {0}, {0});
                czy = at::tensordot(czy, Phiz_b[k+1], {2}, {0});
                tmp = (czy - czA).reshape({rz[k]*N[k], rz[k+1]});
                
                
                at::Tensor uz;
                std::tie(uz, std::ignore, std::ignore) = svd_thin(tmp);
                auto rtmp = kickrank < uz.sizes()[1] ? kickrank : uz.sizes()[1];
                tmp = uz.index({torch::indexing::Ellipsis, torch::indexing::Slice(0,rtmp,1)});
                if(k < d-1)
                    tmp = at::cat({tmp,torch::randn({tmp.sizes()[0],  kick2}, options)}, 1);
                
                at::Tensor qz;
                std::tie(qz, std::ignore) = at::linalg_qr(tmp);

                rz[k+1] = qz.sizes()[1];

                z_cores[k] = qz.reshape({rz[k], N[k], rz[k+1]}).clone();
            }
            if(k<d-1){
                if(!last){
                    at::Tensor tmp = at::linalg_matmul(u, v.t()).reshape({rx[k], N[k], rx[k+1]});
                    at::Tensor left_res = local_product(Phiz[k+1], Phis[k], A_cores[k], tmp);
                    at::Tensor left_b = at::tensordot(Phis_b[k], b_cores[k]*nrmsc, {0}, {0});
                    left_b = at::tensordot(left_b, Phiz_b[k+1], {2}, {0});

                    at::Tensor uk = (left_b - left_res).reshape({u.sizes()[0],-1});
                    uk = at::cat({u, uk},1);
                    auto r_add = left_res.sizes()[2];
                    at::Tensor Rmat;
                    std::tie(u, Rmat) = at::linalg_qr(uk);

                    at::Tensor toadd = torch::zeros({rx[k+1], r_add}, options);
                    v = at::cat({v, toadd}, 1);
                    
                    v = at::linalg_matmul(v, Rmat.t());
                }

                r = u.sizes()[1];
                v = at::tensordot(v, x_cores[k+1], {0}, {0});

                nrmsc = nrmsc * normA[k] * normx[k] / normb[k];

                x_cores[k] = u.reshape({rx[k], N[k], r}).clone();

                Phis[k+1] = compute_phi_fwd_A(Phis[k], x_cores[k], A_cores[k], x_cores[k]);
                Phis_b[k+1] = compute_phi_fwd_rhs(Phis_b[k], b_cores[k],x_cores[k]);

                // ... and norms, with one synchronisation
                std::vector<double> norms = to_host({torch::norm(v), torch::norm(Phis[k+1]), torch::norm(Phis_b[k+1])});

                double norm_now = norms[0] > 0 ? norms[0] : 1.0;
                normx[k] = normx[k] * norm_now;
                x_cores[k+1] = (v / norm_now).reshape({r, N[k+1], rx[k+2]});
                rx[k+1] = r;

                normA[k] = norms[1] > 0 ? norms[1] : 1.0;
                Phis[k+1] = Phis[k+1] / normA[k];
                normb[k] = norms[2] > 0 ? norms[2] : 1.0;
                Phis_b[k+1] = Phis_b[k+1] / normb[k];
                
                // norm correction
                nrmsc = nrmsc * normb[k] / ( normA[k] * normx[k] );


                // next phiz
                if(!last){
                    Phiz[k+1] = compute_phi_fwd_A(Phiz[k], z_cores[k], A_cores[k], x_cores[k]) / normA[k];
                    Phiz_b[k+1] = compute_phi_fwd_rhs(Phiz_b[k], b_cores[k],z_cores[k]) / normb[k];
                }
            }   
            else{
                auto usv = at::linalg_matmul(u * s.index({torch::indexing::Slice(0,r,1)}), v.index({torch::indexing::Slice(0,r,1), torch::indexing::Ellipsis}).t());
                x_cores[k] = usv.reshape({rx[k],N[k],rx[k+1]});
            }



        }
        if(verbose){
            std::cout << "Solution rank [ ";
            for(auto rr: rx) std::cout << rr << " ";
            std::cout << "]" << std::endl;
            std::cout << "Maxres " << max_res << std::endl;
            auto diff_time = std::chrono::high_resolution_clock::now() - tme_swp;
            std::cout << "Time " << (double)(std::chrono::duration_cast<std::chrono::microseconds>(diff_time)).count()/1000.0 << " ms"<<std::endl<<std::flush;
        }

        if(last)
            break;

        if(max_res<eps)
            last = true;
    }

    if(verbose){
        auto diff_time = std::chrono::high_resolution_clock::now() - tme_total;
        std::cout << std::endl << "Finished after " << (swp < nswp ? swp+1 : swp) <<" sweeps and "<< (double)(std::chrono::duration_cast<std::chrono::microseconds>(diff_time)).count()/1000000.0 << " seconds"  << std::endl << std::endl;
    }

    double norm_x = 0.0;
    for(int i=0;i<d-1;i++)
        norm_x += std::log(normx[i]);

    norm_x = std::exp(norm_x/d);

    for(int i = 0;i<d;i++)
        x_cores[i] = x_cores[i] * norm_x;

    // release memory
    delete [] normA;
    delete [] normb;
    delete [] normx;

    return x_cores;
}
