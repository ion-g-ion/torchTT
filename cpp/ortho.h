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
    // read the singular values as doubles whatever the dtype/device of s
    at::Tensor sd = s.to(torch::kCPU, torch::kFloat64).contiguous();
    int n = sd.sizes()[0];
    const double *ss = sd.data_ptr<double>();

    double total = 0.0;
    for (int k = 0; k < n; k++)
        total += ss[k] * ss[k];
    if (total == 0.0)
        return 1;

    if (eps <= 0.0)
        return n;

    // drop singular values from the tail while their accumulated energy stays below eps^2
    double tail = 0.0;
    int r = n;
    while (r > 1)
    {
        tail += ss[r - 1] * ss[r - 1];
        if (tail >= eps * eps)
            break;
        r--;
    }

    return r;
}

void rl_orthogonal_this(std::vector<at::Tensor> &cores, std::vector<uint64_t> &shape, std::vector<uint64_t> &rank){

    uint64_t d = shape.size();


    at::Tensor core_now;


    for(int i=d-1;i>0;i--){
        core_now = cores[i].reshape({cores[i].sizes()[0],  cores[i].sizes()[1]* cores[i].sizes()[2]}).t();

        // perform QR
        std::tuple <at::Tensor, at::Tensor> QR = at::linalg_qr(core_now);


        uint64_t r_new;
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
        std::tuple <at::Tensor, at::Tensor> QR = at::linalg_qr(core_now);
       
        rank[i+1] = std::get<0>(QR).sizes()[1];

        cores[i] = std::get<0>(QR).reshape({rank[i], shape[i], -1});
        
        cores[i+1] = (std::get<1>(QR).matmul(cores[i+1].reshape({cores[i+1].sizes()[0],-1}))).reshape({cores[i].sizes()[2], shape[i+1],-1});

    }



}

#endif
