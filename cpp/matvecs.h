#include "define.h"

/*
template <typename T> 
class AMENsolveMV_cpu{

private:
    T * Phi_left;
    T * Phi_right;
    T * coreA;
    T * J;
    int prec;
    T * work;

public:
    ~AMENsolveMV_cpu(){
        delete [] Phi_left;
        delete [] Phi_right;
        delete [] coreA;
        delete [] work;
        if(J!=nullptr)
            delete [] J;
    }


}; */

template <typename T> class AMENsolveMV{

private:
    at::Tensor Phi_left;
    at::Tensor Phi_right;
    at::Tensor coreA;
    at::Tensor J;
    int prec;
    at::IntArrayRef shape;
    at::TensorOptions options;
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

        at::Tensor w = at::tensordot(tmp, this->Phi_left, {0}, {2});
        at::Tensor w2 = at::tensordot(w, this->coreA, {0,3}, {2,0});
        at::Tensor w3 = at::tensordot(w2, this->Phi_right, {0,3}, {2,1});
        return w3.reshape({this->shape[0]*this->shape[1]*this->shape[2],1});

    }

    void matvec_cpu(T *in, T *out){

        
    }
};