#include "define.h"

torch::Tensor full(std::vector<torch::Tensor> cores)
{
    torch::Tensor t = cores[0].index({0, torch::indexing::Ellipsis});
    for (int i = 1; i < cores.size(); i++)
    {

        t = torch::tensordot(t, cores[i], {i}, {0});
    }
    return t.index({torch::indexing::Ellipsis, 0});
}
