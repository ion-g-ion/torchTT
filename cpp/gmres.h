#include "define.h"
#include <omp.h>
#include <functional>



/**
 * @brief givensrotation.
 * 
 * @tparam T typename (double or float).
 * @param[in] v1 the first value.
 * @param[in] v2 the second value.
 * @return std::tuple<T,T> 
 */
template <typename T> std::tuple<T,T> givens_rotation(T v1, T v2){
    T den = std::sqrt(v1*v1+v2*v2);
    return std::make_tuple(v1/den, v2/den);
}

template <typename T> void apply_givens_rotation_cpu(T *h, T *cs, T *sn, uint64_t k, T &cs_k, T &sn_k){
  
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
void gmres_single(at::Tensor &solution, int &flag, int &nit, AMENsolveMV<T> &Op, at::Tensor &rhs,  at::Tensor &x0, uint64_t size, uint64_t iters, T threshold){

    bool converged = false;

    at::Tensor r = rhs - Op.matvec(x0);

    T b_norm = torch::norm(rhs).item<T>();
    T error = torch::norm(r).item<T>() / b_norm;

    T * sn = new T[iters];
    T * cs = new T[iters];
    T * e1 = new T[iters+1];
    for (int i=0; i<iters;++i){
        sn[i] = 0;
        cs[i] = 0;
        e1[i+1] = 0;
    }
    e1[0] = 1.0;

    T r_norm = torch::norm(r).item<T>();

    if(r_norm<=0){
        flag = 1;
        nit = 0;
        solution = x0.clone();
        // free memory
        delete [] sn;
        delete [] cs;
        delete [] e1;
        return;
    }

    std::vector<at::Tensor> Q;
    Q.push_back(r.squeeze() / r_norm);

    at::Tensor H, beta;
    if(std::is_same<T,float>::value){
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        beta = torch::zeros(iters+1, options);
        H = torch::zeros({iters+1, iters}, options);
    }
    else{
        auto options = torch::TensorOptions().dtype(torch::kFloat64);
        beta = torch::zeros(iters+1, options);
        H = torch::zeros({iters+1, iters}, options); 
    }

    auto betaA = beta.accessor<T,1>();
    auto HA = H.accessor<T,2>();
    betaA[0] = r_norm;

    int k;
    for(k = 0; k<iters; k++){
        
       // std::cout <<"\n";
      //  auto ts = std::chrono::high_resolution_clock::now();
        at::Tensor q = Op.matvec(Q[k]);
     //   auto diff_time = std::chrono::high_resolution_clock::now() - ts;
      //  std::cout << " MV   " << (double)(std::chrono::duration_cast<std::chrono::microseconds>(diff_time)).count()/1000 << std::endl;

      //  ts = std::chrono::high_resolution_clock::now();
        
       // #pragma omp parallel for num_threads(32)
        for(int i=0;i<k+1;i++){
            HA[i][k] = at::dot(q.squeeze(), Q[i]).item<T>();
            q -= (HA[i][k] * Q[i]).reshape({-1,1});
        }
      //  diff_time = std::chrono::high_resolution_clock::now() - ts;
      //  std::cout << " PROJ " << (double)(std::chrono::duration_cast<std::chrono::microseconds>(diff_time)).count()/1000 << std::endl;

     //   ts = std::chrono::high_resolution_clock::now();
        T h = torch::norm(q).item<T>();

        q /= h;

        HA[k+1][k] = h;
        Q.push_back(q.clone().squeeze());

        T c,s;
        at::Tensor htemp = H.index({torch::indexing::Slice(0,k+2), k}).contiguous();
        apply_givens_rotation_cpu(htemp.data_ptr<T>(), cs, sn, k+1, c, s);
        H.index_put_({torch::indexing::Slice(0,k+2), k}, htemp);
        cs[k] = c;
        sn[k] = s;

        betaA[k+1] = -sn[k]*betaA[k];
        betaA[k] = cs[k]*betaA[k];
        error = std::abs(betaA[k+1])/b_norm;
       // diff_time = std::chrono::high_resolution_clock::now() - ts;
     // std::cout << " REST " << (double)(std::chrono::duration_cast<std::chrono::microseconds>(diff_time)).count()/1000 << std::endl;
        if(error<=threshold)
        {
            flag = 1;
            break;
        }
    }
    k = k<iters ? k : iters-1;
    at::Tensor y = at::linalg_solve(H.index({torch::indexing::Slice(0,k+1), torch::indexing::Slice(0,k+1)}), beta.index({torch::indexing::Slice(0, k+1)}).reshape({-1,1}));
    
    solution = x0.clone().squeeze();
    for(int i=0;i<k+1;++i)
        solution += Q[i] * y.index({i,0}).item<T>();  

    nit = k;
    // free memory
    delete [] sn;
    delete [] cs;
    delete [] e1;

}

template <typename T>
void gmres(at::Tensor &solution, int &flag, int &nit, AMENsolveMV<T> &Op, at::Tensor &rhs,  at::Tensor &x0, uint64_t size, uint64_t max_iters, T threshold, uint64_t resets ){
    nit = 0;
    flag = 0;

    auto xs = x0;
    for(int r =0;r<resets;r++){
        int nowit;
        gmres_single<T>(solution, flag, nowit, Op, rhs, xs, size, max_iters, threshold);
        nit+=nowit;
        if(flag==1){
            break;
        }
        xs = solution.clone();
    }
}

/*
void gmres_double_cpu(double *solution, 
                      int &flag, 
                      int &nit, 
                      std::function<void(double*,double*)> matvec,
                      double *rns, 
                      int64_t size, 
                      int64_t max_iters, 
                      double threshold, 
                      int64_t resets)
{

    nit = 0;
    flag = 0;

    int64_t inc1 = 1;
    char transN = 'N';

    double *sn = new double[iters];
    double *cs = new double[iters];
    double *e1 = new double[iters+1];

    double *Q = new double[size*(iters+1)];
    double *q = new double[size];
    double *H = new double[iters*(iters+1)];
    double *beta = new double[iters+1];
    double *work1 = new double [iters+1];
    
    for(uint64_t r=0; r<resets; r++)
    {
        int k;
        // fill with 0
        std::fill_n(beta, iters+1, 0);
        std::fill_n(H, (iters+1)*iters, 0);

        for(k = 0; k<iters; k++)
        {
        
            // matvec 
            matvec(Q+k*size, Q+(k+1)*size);

            // 
            for(int i=0;i<k+1;i++){
                H[i+(iters+1)*k] = BLAS::dot(&size, Q+(k+1)*size, &inc1, Q+i*size, &inc1);
                BLAS::axpy(&size, H+i+(iters+1)*k, Q+i*size, &inc1, Q+(k+1)*size, &inc1);
            }

            T h = 1/BLAS::nrm2(&size, Q+(k+1)*size, &inc1);

            BLAS::scal(&size, &h, Q+(k+1)*size, &inc1);

            H[k+1+(iters+1)*k] = h;

            // >>>
            T c,s;
            apply_givens_rotation_cpu(H+k*(iters+1), cs, sn, k+1, c, s);
            cs[k] = c;
            sn[k] = s;

            betaA[k+1] = -sn[k]*betaA[k];
            betaA[k] = cs[k]*betaA[k];
            error = std::abs(betaA[k+1])/b_norm;

            if(error<=threshold)
            {
                flag = 1;
                break;
            }
        }

        k = k<iters ? k : iters-1;

    at::Tensor y = at::linalg_solve(H.index({torch::indexing::Slice(0,k+1), torch::indexing::Slice(0,k+1)}), beta.index({torch::indexing::Slice(0, k+1)}).reshape({-1,1}));
    
    solution = x0.clone().squeeze();
    for(int i=0;i<k+1;++i)
        solution += Q[i] * y.index({i,0}).item<T>();  

        nit += k+1;

        if(flag==1){
            break;
        }
    }

    delete [] sn;
    delete [] cs;
    delete [] e1;
    delete [] Q;
    delete [] q;
    delete [] H;
}*/