#include "define.h"

template <typename T> std::tuple<T,T> givens_rotation(T v1, T v2){
    T den = std::sqrt(v1*v1+v2*v2);
    return std::make_tuple(v1/den, v2/den);
}

template <typename T> void apply_givens_rotation_cpu(T *h, double *cs, T *sn, uint64_t k, T &cs_k, T &sn_k){

    for(int i = 0; i < k-1; ++i){
        T temp   =  cs[i]* h[i] + sn[i] * h[i+1];
        h[i+1] = -sn[i] * h[i] + cs[i] * h[i+1];
        h[i]   = temp;
    }
    std::tie(cs_k, sn_k) = givens_rotation(h[k-1], h[k]);

    h[k-1] = cs_k * h[k-1] + sn_k * h[k];
    h[k] = 0.0;
}

template <typename T>
void gmres(at::Tensor &solution, int &flag, int &nit, AMENsolveMV<T> &Op, at::Tensor &rhs,  at::Tensor &previous_solution, uint64_t size, uint64_t iters, T eps_local, uint64_t resets ){

}

template <typename T>
void gmres_single(at::Tensor &solution, int &flag, int &nit, AMENsolveMV<T> &Op, at::Tensor &rhs,  at::Tensor &previous_solution, uint64_t size, uint64_t iters, T eps_local, uint64_t resets ){

    bool converged = false;

    at::Tensor r = rhs - Op.matvec(previous_solution);

    T b_norm = torch::norm(rhs).item<T>();
    T error = torch::norm(r).item<T>() / b_norm;

    T * sn = new T[iters];
    T * cs = new T[iters];
    T * e1 = new T[iters+1];
    e1[0] = 1.0;

    


    for(int k = 0; k<iters; k++){


    }

    // free memory
    delete [] sn;
    delete [] cs;
    delete [] e1;



}

