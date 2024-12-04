#ifndef ORTHO
#define ORTHO
#include "define.h"

void perform_QR(at::Tensor &Q, at::Tensor &R, at::Tensor &M){
    at::linalg_qr_out(Q,R,M);
}

/**
 * @brief chop the rank up to a prescribed accuracy.
 *
 * @param s the singular values vactor.
 * @param eps the relative accuracy.
 * @return int
 */
int rank_chop(torch::Tensor s, double eps)
{
    int n = s.sizes()[0];
    int r = n - 1;
    if (torch::norm(s).item<double>() == 0.0)
        return 1;

    if (eps <= 0.0)
        return r;

    double *ss = (double *)s.data_ptr();

    while (r > 0)
    {
        double sum = 0.0;

        for (int k = r; k < n; k++)
            sum += ss[k] * ss[k];

        if (sum >= eps * eps)
            break;

        r--;
    }
    r++;
    r = r > 0 ? r : 1;

    return r;
}

void rl_orthogonal_this(std::vector<at::Tensor> &cores, std::vector<uint64_t> &shape, std::vector<uint64_t> &rank){

    uint64_t d = shape.size();


    at::Tensor core_now;


    for(int i=d-1;i>0;i--){
        core_now = cores[i].reshape({cores[i].sizes()[0],  cores[i].sizes()[1]* cores[i].sizes()[2]}).t();

        // perform QR
        // perform_QR(Q,R,core_now);
        std::tuple <at::Tensor, at::Tensor> QR = at::linalg_qr(core_now);


        uint64_t r_new; // = core_now.sizes()[0] < core_now.sizes()[1] ? core_now.sizes()[0] : core_now.sizes()[1];
        r_new = std::get<1>(QR).sizes()[0];

        cores[i] = std::get<0>(QR).t().reshape({r_new,shape[i],-1});
        rank[i] = r_new;

        cores[i-1] = (cores[i-1].reshape({-1,cores[i-1].sizes()[2]}).matmul(std::get<1>(QR).t())).reshape({cores[i-1].sizes()[0],shape[i-1],-1});

    }

    
}



void lr_orthogonal(std::vector<at::Tensor> &cores, std::vector<uint64_t> &shape, std::vector<uint64_t> &rank){
    int d = shape.size();

    at::Tensor core_now;
    


    for(int i=0;i<d-1;i++){
        core_now = cores[i].reshape({cores[i].sizes()[0]*cores[i].sizes()[1], cores[i].sizes()[2]});

        // perform QR
        //perform_QR(Q,R,core_now);
        std::tuple <at::Tensor, at::Tensor> QR = at::linalg_qr(core_now);
       
        rank[i+1] = std::get<0>(QR).sizes()[1];

        cores[i] = std::get<0>(QR).reshape({rank[i], shape[i], -1});
        
        cores[i+1] = (std::get<1>(QR).matmul(cores[i+1].reshape({cores[i+1].sizes()[0],-1}))).reshape({cores[i].sizes()[2], shape[i+1],-1});

    }



}

#endif