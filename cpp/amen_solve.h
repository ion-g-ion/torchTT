#include "define.h"
#include "ortho.h"

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
    return at::tensordot(tmp2, Phi_right, {3}, {1});

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
                        int preconditioner)
{

    //at::TensorBase::device dtype = A_cores[0].dtype;
    auto options = A_cores[0].options();
    uint64_t d = N.size();
    std::vector<at::Tensor> x_cores = x0_cores;
    std::vector<uint64_t> rx = r_x0;

    std::vector<uint64_t> rz(d+1);
    rz[0] = 1;
    rz[d] = 1;
    for(int i=1;i<d;i++) 
        rz[i] = kickrank+kick2;
    std::vector<at::Tensor> z_cores(d);
    for(int i=0;i<d;i++) 
        z_cores[i] = at::randn({rz[i], N[i], rz[i + 1]}, options);

    rl_orthogonal_this(z_cores, N, rz);
    //std::cout<<"\n\n\nINSIDE "<<b_cores[0]<<"\n\n\n";

    // the interface matrices
    std::vector<at::Tensor> Phiz(d+1);
    std::vector<at::Tensor> Phiz_b(d+1); 
    std::vector<at::Tensor> Phis(d+1);
    std::vector<at::Tensor> Phis_b(d+1);
    Phiz[0] = at::ones({1,1,1}, options);
    Phiz_b[0] = at::ones({1,1}, options);
    Phis[0] = at::ones({1,1,1}, options);
    Phis_b[0] = at::ones({1,1}, options);

    double *normA = new double[d-1];
    double *normb = new double[d-1];
    double *normx = new double[d-1];
    double nrmsc = 1.0;

    bool last = false;

    std::chrono::time_point<std::chrono::high_resolution_clock> tme_swp;
    for(int swp=0;swp<nswp;swp++){

        if(verbose){
            if(last)
                std::cout<<"Starting sweep " << swp << " (last one)" << std::endl;
            else
                std::cout<<"Starting sweep " << swp <<  std::endl;
            tme_swp = std::chrono::high_resolution_clock::now();
        }

        for(int k=d-1;k>0;k--){
            if(!last){
                if(swp>0){
                    auto czA = local_product(Phiz[k+1], Phiz[k], A_cores[k], x_cores[k]);
                    // >>>>>>>>>>>>>>>>>>>
                }
            }
        }
    }

    // release memory
    delete [] normA;
    delete [] normb;
    delete [] normx;

    return A_cores;
}