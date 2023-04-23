#include "BLAS.h"
#include <chrono>
#include <iostream>
#include<functional>
#include <math.h>
#include <omp.h>

#define N 20000

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



void mv(double *in, double *out){

#pragma omp parallel for
    for(int i=1;i<N-1;++i)
        out[i] = (-2*in[i] + in[i+1] + in[i-1])*(N-1)*(N-1);
    out[0] = -in[0]*N*N;
    out[N-1] = -in[N-1]*N*N;
}


int main(){
    uint64_t n = 1024;

    double *rhs = new double[N];
    double *tmp = new double[N];
    double *solution = new double[N];

    rhs[0] = 0;
    rhs[N-1] = 0;
    for(int i=1;i<N-1;++i)
    {
        solution[i] = 0;
        rhs[i] = 1;
    }

    std::chrono::time_point<std::chrono::system_clock> start, end;
 
    start = std::chrono::system_clock::now();
    
    int nit, flag;
    gmres_double_cpu(solution, 
                      flag, 
                      nit, 
                      mv,
                      rhs, 
                      N, 
                      100, 
                      1e-8, 
                      20,
                      false);

    end = std::chrono::system_clock::now();
 
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
 
    mv(solution, tmp);

    int64_t sz = N;
    int64_t inc1 = 1;
    double m1 = -1;

    BLAS::axpy(&sz, &m1, rhs, &inc1, tmp, &inc1);
    double nrm = BLAS::nrm2(&sz, tmp, &inc1) / N;

    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << "Flag " << flag << " number of iterations " << nit <<std::endl;
    std::cout << "Residual " << nrm << std::endl;

    //for(int i = 0 ; i < N ; i++) std::cout << solution[i] << " ";

    delete [] rhs;
    delete [] tmp;
    delete [] solution;

    return 0;
}