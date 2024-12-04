#include "define.h"



template <typename T> 
class AMENsolveMV_cpu{

private:
    T * Phi_left;
    T * Phi_right;
    T * coreA;
    T * J;
    int prec;
    T * work1;
    T * work2;
    int64_t r,R,n,s,S,l,L;

public:
    ~AMENsolveMV_cpu(){
        delete [] Phi_left;
        delete [] Phi_right;
        delete [] coreA;
        delete [] work1;
        delete [] work2;

        if(!prec)
            delete [] J;
    }


    AMENsolveMV_cpu(at::Tensor &Phi_left, at::Tensor &Phi_right, at::Tensor & coreA, int prec)
    {
        int64_t inc1 = 1;
        int64_t size;

        s = coreA.sizes()[0];
        n = coreA.sizes()[1];
        S = coreA.sizes()[3];
        l = Phi_left.sizes()[0];
        r = Phi_left.sizes()[2];
        L = Phi_right.sizes()[0];
        R = Phi_right.sizes()[2];

        if(this->prec == C_PREC){
            auto Jl = at::tensordot(at::diagonal(Phi_left,0,0,2), coreA, {0}, {0});
            auto Jr = at::diagonal(Phi_right, 0, 0, 2);
            auto J = at::linalg_inv(at::tensordot(Jl,Jr,{3},{0}).permute({0,3,1,2}));
            // TODO: switch to column major

            size = r*R*n*n;
            this->J = new T[size];
            BLAS::copy(&size, J.data_ptr<T>(), &inc1, this->J, &inc1);
        }
        else if(this->prec == R_PREC){
            auto Jl = at::tensordot(at::diagonal(Phi_left,0,0,2), coreA, {0},{0}); // sd,smnS->dmnS
            auto Jt = at::tensordot(Jl, Phi_right, {3}, {1}); // dmnS,LSR->dmnLR
            Jt = Jt.permute({0, 1, 3, 2, 4});
            auto sh = Jt.sizes();
            auto Jt2 = Jt.reshape({-1, Jt.sizes()[1]*Jt.sizes()[2], Jt.sizes()[3]*Jt.sizes()[4]});
            auto J = at::linalg_inv(Jt2).reshape(sh);
            // TODO : switch to column major

            size = r*n*L*n*R;
            this->J = new T[size];
            BLAS::copy(&size, J.data_ptr<T>(), &inc1, this->J, &inc1);
            
        }
        
        // TODO: switch to column major !!!

        auto Pl = Phi_left.contiguous();
        auto Pr = Phi_right.contiguous();

        Pl = Pl.permute({0,2,1}).contiguous();
        Pr = Pr.permute({2,1,0}).contiguous();

        this->coreA = new T[s*n*n*S];
        this->Phi_left = new T[l*r*s];
        this->Phi_right = new T[R*S*L];
        size = std::max(r*n*S*L,s*n*r*L);
        this->work1 = new T[size];
        this->work2 = new T[size];

        // copy everything
        
        size = s*n*n*S;
        BLAS::copy(&size, coreA.contiguous().data_ptr<T>(), &inc1, this->coreA, &inc1);
        size = l*r*s;
        BLAS::copy(&size, Pl.data_ptr<T>(), &inc1, this->Phi_left, &inc1);
        size = R*S*L;
        BLAS::copy(&size, Pr.data_ptr<T>(), &inc1, this->Phi_right, &inc1);


    }

    void matvec(T *in, T *out)
    {
        //w = tn.einsum('lsr,smnS,LSR,rnR->lmL',self.Phi_left,self.coreA,self.Phi_right,x)
        //w = tn.einsum('rsl,smnS,RSL,rnR->lmL',self.Phi_left,self.coreA,self.Phi_right,x)
        char tN = 'N';
        char tC = 'C';
        T alpha1 = 1.0;
        T alpha0 = 0.0;
        int64_t M, N, K;

        M = r*n;
        N = S*L;
        K = R;
        BLAS::gemm(&tN, &tN, &M, &N, &K, &alpha1, in, &M, this->Phi_right, &K, &alpha0, this->work1, &M); // work1 is now rnSL
        // >>>>> transpose !!!!!

        // work is now LrnS
        M = L*r;
        N = s*n;
        K = n*S;
        BLAS::gemm(&tN, &tC, &M, &N, &K, &alpha1, this->work1, &M, this->coreA, &N, &alpha0, this->work2, &M);
        // work2 is now Lrsm

        // >>>>>> transpose !!!

        //work is now rsmL
        M = l;
        N = n*L;
        K = r*s;
        BLAS::gemm(&tN, &tN, &M, &N, &K, &alpha1, this->work2, &M, this->Phi_left, &K, &alpha0, out, &M);
    }

}; 

template <typename T> class AMENsolveMV{

private:
    at::Tensor Phi_left;
    at::Tensor Phi_right;
    at::Tensor coreA;
    at::Tensor J;
    int prec;
    at::IntArrayRef shape;
    at::TensorOptions options;

    T * Phi_left_ptr;
    T * Phi_right_ptr;
    T * coreA_ptr;
    T * J_ptr;
    T * work1_ptr;
    T * work2_ptr;
    int64_t r,R,n,s,S,l,L;
public:
    AMENsolveMV(){
       ;
    }
    
    void setter(at::Tensor &Phi_left, at::Tensor &Phi_right, at::Tensor & coreA, at::IntArrayRef shape, int prec, at::TensorOptions options){
        this->prec = prec;
        this->options = options;
        this->shape = shape;

        this->Phi_left = Phi_left; //torch::from_blob(Phi_left.contiguous().data_ptr<T>(), Phi_left.sizes(), options);
        this->Phi_right = Phi_right; //torch::from_blob(Phi_right.contiguous().data_ptr<T>(), Phi_right.sizes(), options);
        this->coreA = coreA; // torch::from_blob(coreA.contiguous().data_ptr<T>(), coreA.sizes(), options);
        if(this->prec == C_PREC){
            auto Jl = at::tensordot(at::diagonal(Phi_left,0,0,2), coreA, {0}, {0});
            auto Jr = at::diagonal(Phi_right, 0, 0, 2);
            this->J = at::linalg_inv(at::tensordot(Jl,Jr,{3},{0}).permute({0,3,1,2}));
        }
        else if(this->prec == R_PREC){
            auto Jl = at::tensordot(at::diagonal(Phi_left,0,0,2), coreA, {0},{0}); // sd,smnS->dmnS
            auto Jt = at::tensordot(Jl, Phi_right, {3}, {1}); // dmnS,LSR->dmnLR
            Jt = Jt.permute({0, 1, 3, 2, 4});
            auto sh = Jt.sizes();
            auto Jt2 = Jt.reshape({-1, Jt.sizes()[1]*Jt.sizes()[2], Jt.sizes()[3]*Jt.sizes()[4]});
            this->J = at::linalg_inv(Jt2).reshape(sh);
            //std::cout << "READY 1" <<std::endl;
        }
    }

    at::Tensor apply_prec(at::Tensor sol){
        at::Tensor ret;
        if(this->prec == C_PREC) {
        uint64_t s0,s1,s2;
            s0 = sol.sizes()[0];
            s1 = sol.sizes()[1]; 
            s2 = sol.sizes()[2];

            at::Tensor tmp = sol.permute({0,2,1}).reshape({s0, s2, s1, 1});
            ret = at::linalg_matmul(this->J, tmp).permute({0,2,1,3}).reshape({s0, s1, s2}); 
        }
        else if(this->prec == R_PREC){
            ret = at::einsum("rnR,rmLnR->rmL", {sol, this->J});

        }

        return ret;
    }

    at::Tensor matvec(at::Tensor &x, bool use_prec = true){
        at::Tensor tmp;

        if(!use_prec || this->prec == NO_PREC){
            tmp = x.reshape(this->shape);            
        }
        else
        {
            tmp = apply_prec(x.reshape(this->shape));
        }
// w = tn.einsum('lsr,smnS,LSR,rnR->lmL',self.Phi_left,self.coreA,self.Phi_right,x)
        auto w = at::tensordot(tmp, this->Phi_left, {0}, {2});
        auto w2 = at::tensordot(w, this->coreA, {0,3}, {2,0});
        auto w3 = at::tensordot(w2, this->Phi_right, {0,3}, {2,1});
        return w3.reshape({this->shape[0]*this->shape[1]*this->shape[2],1});

    }

    void matvec_cpu(T *in, T *out){

        
    }
};