#include "ortho.h"

void round_this(std::vector<at::Tensor> &cores, std::vector<uint64_t> &shape, std::vector<uint64_t> &rank, double epsilon, uint64_t rmax){
    uint64_t d = cores.size();

    if(d<=1){
        return ;
    }

    lr_orthogonal(cores, shape, rank);
    double eps = epsilon / std::sqrt((double)d-1.0);



    at::Tensor core_now, core_next;

    core_now = cores[d-1].reshape({rank[d-1],shape[d-1]*rank[d]});
      
    for(int i=d-1;i>0;i--){
        
        core_next = cores[i-1].reshape({rank[i-1]*shape[i-1], rank[i]});

        std::tuple <at::Tensor, at::Tensor, at::Tensor> USV = at::linalg_svd(core_now, false);

        int rc = rank_chop(std::get<1>(USV), eps * torch::norm(std::get<1>(USV)).item<double>());

        int rnew = rmax < rc ?  rmax : rc;

        rank[i] = rnew;

        auto US = std::get<0>(USV).index({torch::indexing::Ellipsis, torch::indexing::Slice(0, rnew)}).matmul(torch::diag(std::get<1>(USV).index({torch::indexing::Slice(0, rnew)})));
        core_next = core_next.matmul(US);

        cores[i-1] = core_next.reshape({rank[i-1], shape[i-1], rank[i]});
        cores[i] = std::get<2>(USV).index({ torch::indexing::Slice(0, rnew), torch::indexing::Ellipsis}).reshape({rank[i], shape[i], rank[i+1]});

        core_now = core_next.reshape({rank[i-1],-1});


    }

}