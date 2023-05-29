#include "define.h"
#include "ortho.h"
#include <cmath>

/**
 * @brief Fast matvec `y = Ax` using DMRG optimization
 * 
 * @param A_cores the cores of the TT matrix `A`
 * @param x_cores the cores of `x`
 * @param y0_cores initial guess for `y`. If an emptyvector is provided the guess is random
 * @param M shape of `y`
 * @param N shape of `x`
 * @param rx rank of `x`
 * @param ry0 rank of the initial guess of `y`
 * @param nswp number of sweeps
 * @param eps relative accuracy
 * @param rmax maximum rank
 * @param kickrank kickrank
 * @param r_enlage how much we enlarge the rank compared to the previous step
 * @param verb show debug info.
 * @return std::vector<at::Tensor> 
 */
std::vector<at::Tensor> dmrg_mv(
                        std::vector<at::Tensor> A_cores, 
                        std::vector<at::Tensor> x_cores, 
                        std::vector<at::Tensor> y0_cores, 
                        std::vector<int64_t> M, 
                        std::vector<int64_t> N, 
                        std::vector<int64_t> rx, 
                        std::vector<int64_t> ry0, 
                        int64_t nswp, double eps, 
                        int64_t rmax, 
                        int64_t kickrank, 
                        bool verb)
{
    auto options = A_cores[0].options();
    int64_t d = N.size();

    std::vector<at::Tensor> y_cores(d);
    std::vector<int64_t> ry(d+1);
    
   
    if(y0_cores.size() == 0){
        
        for(int i = 1; i < d; ++i)
            ry[i] = 2;
        ry[0] = 1;
        ry[d] = 1;
        for(int i=0; i < d; ++i)
            y_cores[i] = torch::randn({ry[i], M[i], ry[i + 1]}, options);
    }
    else{
        y_cores = y0_cores;
        ry = ry0;
    }

    std::vector<at::Tensor> Phis(d+1);

    Phis[0] = at::ones({1,1,1}, options);
    Phis[d] = at::ones({1,1,1}, options);

    std::vector<double> delta_cores(d-1);
    std::vector<double> delta_cores_prev(d-1);
    std::vector<int64_t> r_enlarge(d);

    for(int i = 0; i < d-1; ++i)
    {
        r_enlarge[i] = 2;
        delta_cores[i] = 1.0;
        delta_cores_prev[i] = 1.0;
    }
    r_enlarge[d-1] = 2;

    bool last = false;

    std::chrono::time_point<std::chrono::high_resolution_clock> tme_swp, tme_total;

    if(verb)
        tme_total = std::chrono::high_resolution_clock::now();

    int swp;
    for(swp = 0; swp < nswp; ++swp)
    {
        if(verb)
            std::cout << "Sweep " << swp << std::endl;

        for(int k = d-1; k > 0; --k)
        {
            auto core = y_cores[k].permute({1,2,0}).reshape({M[k]*ry[k+1], ry[k]});
            at::Tensor Q,R;
            std::tie(Q,R) = at::linalg_qr(core.contiguous());
            int64_t rnew = core.sizes()[0] < core.sizes()[1] ? core.sizes()[0] : core.sizes()[1];
            y_cores[k] = Q.t().reshape({rnew, M[k], -1}).contiguous();
            ry[k] = rnew;
            auto core_next = at::tensordot(y_cores[k-1].reshape({-1, y_cores[k-1].sizes()[2]}), R, {1}, {1});
            y_cores[k-1] = core_next.reshape({-1, M[k-1], rnew});


            auto Phi = at::tensordot(Phis[k+1], at::conj(x_cores[k]), {2}, {2});
            Phi  = at::tensordot(at::conj(A_cores[k]), Phi, {2,3}, {3,1});
            Phi = at::tensordot(y_cores[k], Phi, {1,2}, {1,2});

            //auto Phi = at::einsum("ijk,mnk->ijmn",{Phis[k+1],at::conj(x_cores[k])});
            //Phi = at::einsum("ijkl,mlnk->ijmn",{at::conj(A_cores[k]),Phi});
            //Phi = at::einsum("ijkl,mjk->mil",{Phi,y_cores[k]}); 

            Phis[k] = Phi.contiguous().clone();
            
        }

        for(int k = 0; k <d-1; ++k)
        {
            if(verb)
                std::cout << "\tcore " << k << std::endl;

            auto W_prev = at::tensordot(y_cores[k], y_cores[k+1], {2}, {0});

            at::Tensor W;
            if(!last)
            {
                auto W1 = at::tensordot(Phis[k], at::conj(x_cores[k]), {2}, {0});
                W1 = at::tensordot(at::conj(A_cores[k]), W1, {0, 2}, {1, 2}); // Mk x rAk x rk-1 x rxk
                // jlmn   mjln
                
                
                auto W2 = at::tensordot(Phis[k+2], at::conj(x_cores[k+1]), {2}, {2}); // rk+1 x rAk+1 x rxk x Nk+1
                // ijmn    njmi
                W2 = at::tensordot(at::conj(A_cores[k+1]), W2, {2,3}, {3,1}); 

                W = at::tensordot(W1, W2, {1,3}, {0, 3});
                W = W.permute({1,0,2,3});

                // auto W1 = at::einsum("ijk,klm->ijlm",{Phis[k],at::conj(x_cores[k])});
                // W1 = at::einsum("ijkl,mikn->mjln",{at::conj(A_cores[k]),W1}); 
                //   
                // auto W2 = at::einsum("ijk,mnk->njmi",{Phis[k+2],at::conj(x_cores[k+1])});
                // W2 = at::einsum("ijkl,klmn->ijmn",{at::conj(A_cores[k+1]),W2});
                //   
                // W = at::einsum("ijkl,kmln->ijmn",{W1,W2});

            }
            else
                W = at::conj(W_prev);
            
            double b = torch::norm(W).cpu().item<double>();
            if ( b >0 )
            {
                double a = torch::norm(W-at::conj(W_prev)).cpu().item<double>();
                delta_cores[k] = a/b;
            }
            else
                delta_cores[k] = 0;

            if( delta_cores[k] / delta_cores_prev[k] >= 1 && delta_cores[k]>eps)
                r_enlarge[k] += 1;

            if( delta_cores[k]/delta_cores_prev[k] < 0.1 && delta_cores[k]<eps)
                r_enlarge[k] =  r_enlarge[k]-1 > 1 ? r_enlarge[k]-1 : 1;
            
            at::Tensor U,S,V;

            std::tie(U, S, V) = at::linalg_svd(W.reshape({W.sizes()[0]*W.sizes()[1], -1}), false);

            int64_t r_new = rank_chop(S.cpu(), b*eps/(std::pow((double)d, last ? 0.5 : 1.5)) ); //<<<<<<<<<<<<
            
            //  r_new = rank_chop(S.cpu().numpy(),(b.cpu()*eps/(d**(0.5 if last else 1.5))).numpy())

            if(!last)
                r_new += r_enlarge[k];

            r_new = std::min(std::min(r_new, S.sizes()[0]), rmax);
            r_new = r_new > 1 ? r_new : 1;

            at::Tensor W1 = U.index({torch::indexing::Ellipsis, torch::indexing::Slice(0,r_new,1)});
            at::Tensor W2 = V.index({torch::indexing::Slice(0,r_new,1), torch::indexing::Ellipsis}).t() * S.index({torch::indexing::Slice(0, r_new, 1)});

            if( swp < nswp - 1)
            {
                at::Tensor Rmat;
                auto tmp_tens = at::cat({W1, torch::randn({W1.sizes()[0], kickrank}, options)}, 1); 
                std::tie(W1, Rmat) = at::linalg_qr(tmp_tens);
                W2 = at::cat({W2, torch::zeros({W2.sizes()[0], kickrank}, options)}, 1);
                W2 = at::tensordot(Rmat, W2, {1},{1}); 
                r_new = W1.sizes()[1];
            }
            else 
                W2 = W2.t();

            if(verb)
            {
                std::cout << "\tcore " << k << ": delta " << delta_cores[k] << " rank " << ry[k+1] << " -> " << r_new << std::endl;
            }
            ry[k+1] = r_new;

            y_cores[k] = at::conj(W1.reshape({ry[k], M[k], r_new}));
            y_cores[k+1] = at::conj(W2.reshape({r_new, M[k+1], ry[k+2]}));

            // auto Wc = at::conj(at::tensordot(y_cores[k]), y_cores[k+1], {2}, {0}));

            auto Phi_next = at::tensordot(Phis[k], at::conj(x_cores[k]), {2}, {0});
            Phi_next = at::tensordot(Phi_next, at::conj(A_cores[k]), {1,2}, {0, 2}); // result ilmn
            Phi_next = at::tensordot(y_cores[k], Phi_next, {0,1}, {0,2});
            Phi_next = Phi_next.permute({0,2,1});
           
            // auto Phi_next = at::einsum("ijk,kmn->ijmn",{Phis[k],at::conj(x_cores[k])}); // # shape rk-1 x rAk-1 x Nk x rxk
            // Phi_next = at::einsum("ijkl,jmkn->imnl",{Phi_next,at::conj(A_cores[k])}); //# shape  rk-1 x Mk x rAk x rxk
            // Phi_next = at::einsum("ijm,ijkl->mkl",{y_cores[k],Phi_next});

            Phis[k+1] = Phi_next.contiguous().clone();
        }

        if(last)
            break;

        auto max_delta = std::max_element(delta_cores.begin(), delta_cores.end());

        if(*max_delta < eps)
            last = true;

        delta_cores_prev = delta_cores;
    }


    if(verb){
        auto diff_time = std::chrono::high_resolution_clock::now() - tme_total;
        std::cout << std::endl << "Finished after " << (swp < nswp ? swp+1 : swp) <<" sweeps and "<< (double)(std::chrono::duration_cast<std::chrono::microseconds>(diff_time)).count()/1000000.0 << " seconds"  << std::endl << std::endl;
    }


    return y_cores;
} 