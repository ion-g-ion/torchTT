#include "define.h"

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

        this->Phi_left = torch::from_blob(Phi_left.data_ptr<T>(), Phi_left.sizes(), options);
        this->Phi_right = torch::from_blob(Phi_right.data_ptr<T>(), Phi_right.sizes(), options);
        this->coreA = torch::from_blob(coreA.data_ptr<T>(), coreA.sizes(), options);
        if(this->prec == C_PREC){
            auto Jl = at::tensordot(at::diagonal(Phi_left,0,0,2), coreA, {0}, {0});
            auto Jr = at::diagonal(Phi_right, 0, 0, 2);
            this->J = at::linalg_inv(at::tensordot(Jl,Jr,{3},{0}).permute({0,3,1,2}));
        }
    }

    at::Tensor apply_prec(at::Tensor sol){
        return sol;
    }

    at::Tensor matvec(at::Tensor &x, bool apply_prec){
        if(!apply_prec || this->prec == NO_PREC){
            at::Tensor w = at::tensordot(x.reshape(this->shape), this->Phi_left, {0}, {2});
            at::Tensor w2 = at::tensordot(w, this->coreA, {0,3}, {2,0});
            at::Tensor w3 = at::tensordot(w2, this->Phi_right, {0,3}, {2,1});
            return w3;
        }
        else{

        }

    }

};