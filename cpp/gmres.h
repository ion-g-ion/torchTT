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

    nit = k+1;
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


void gmres_double_cpu(double *solution, 
                      int &flag, 
                      int &nit, 
                      std::function<void(double*,double*)> matvec,
                      double *rhs, 
                      int64_t size, 
                      int64_t max_iters, 
                      double threshold, 
                      int64_t resets,
                      bool debug)
{

    nit = 0;
    flag = 0;

    int64_t inc1 = 1;
    char transN = 'N';
    double alpha1 = 1.0;
    double alpham1 = -1.0;

    double *sn = new double[max_iters];
    double *cs = new double[max_iters];

    double *Q = nullptr;
    //double *q = new double[size];
    double *H = new double[max_iters*(max_iters+1)];
    double *beta = new double[max_iters+1];
    double *work1 = new double [max_iters+1];
    
    int64_t *piv_tmp = new int64_t[size];

    double b_norm;
    double error;

    b_norm = BLAS::nrm2(&size, rhs, &inc1);

    if(b_norm <= 0)
    {   
        double alpha0 = 0.0;
        BLAS::scal(&size, &alpha0, solution, &inc1);
        nit = 1;
        flag = 1;
    }
    else
    {

        if(Q == nullptr)
            Q = new double[size*(max_iters+1)]; 

        for(uint64_t r=0; r<resets; r++)
        {
            int k;
            // compute residual
            matvec(solution, Q);
            BLAS::scal<double>(&size, &alpham1, Q, &inc1);
            BLAS::axpy(&size, &alpha1, rhs, &inc1, Q, &inc1);
    
            auto r_norm = BLAS::nrm2(&size, Q, &inc1);
    
            if( ! r_norm>0 )
            {
                flag = 1;
                nit = 0;
                break;
            }

            double tmp = 1/r_norm;
            BLAS::scal(&size, &tmp, Q, &inc1);
    
            //if(Q == nullptr)
            //    Q = new double[size*(max_iters+1)]; 

            // fill with 0
            std::fill_n(beta, max_iters+1, 0);
            std::fill_n(cs, max_iters+1, 0);
            std::fill_n(sn, max_iters+1, 0);
            std::fill_n(H, (max_iters+1)*max_iters, 0);


            error = r_norm / b_norm;
            beta[0] = r_norm;
    
            for(k = 0; k<max_iters; k++)
            {
            
                // matvec 
                matvec(Q+k*size, Q+(k+1)*size);
    
                // 
                for(int i=0;i<k+1;i++){
                    H[i+(max_iters+1)*k] = BLAS::dot(&size, Q+(k+1)*size, &inc1, Q+i*size, &inc1);
                    double s = -H[i+k*(max_iters+1)];
                    BLAS::axpy(&size, &s, Q+i*size, &inc1, Q+(k+1)*size, &inc1);
                }
    
                double h = BLAS::nrm2(&size, Q+(k+1)*size, &inc1);
    
                double oh = 1/h;
                BLAS::scal(&size, &oh, Q+(k+1)*size, &inc1);
    
                H[k+1+(max_iters+1)*k] = h;
    
                // >>>
                double c,s;
                apply_givens_rotation_cpu(H+k*(max_iters+1), cs, sn, k+1, c, s);
                cs[k] = c;
                sn[k] = s;
    
                beta[k+1] = -sn[k]*beta[k];
                beta[k] = cs[k]*beta[k];
                error = std::abs(beta[k+1])/b_norm;

                if(debug)
                    std::cout << "Iteration " << k << " error " << error << std::endl; 
                if(error<=threshold)
                {
                    flag = 1;
                    break;
                }
            }
    
            k = k<max_iters ? k : max_iters-1;

            
            int64_t info = LAPACK::gesv(k+1, 1, H, max_iters+1, piv_tmp, beta, k+1);

            //if(info != 0)
            //    throw std::runtime_error("Error in GMRES, Hy=beta is singular.");

            for(int i = 0; i<k+1; ++i)
                BLAS::axpy(&size, beta+i, Q+i*size, &inc1, solution, &inc1);

            nit += k+1;
    
            if(flag==1){
                break;
            }
        }
    }
    delete [] sn;
    delete [] cs;
    if(Q != nullptr)
        delete [] Q;
    delete [] work1;
    delete [] H;
    delete [] piv_tmp;
}
