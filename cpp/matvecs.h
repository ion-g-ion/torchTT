#pragma once
#include "define.h"

/**
 * @brief Local operator of the AMEn solver: y[l,m,L] = Phi_left[l,s,r] coreA[s,m,n,S] Phi_right[L,S,R] x[r,n,R].
 *
 * The operands are converted to the working dtype and rearranged once in the constructor, so that
 * every product is three matrix products and no operand is copied.
 * The product can also be applied to a batch of vectors.
 */
class AMENsolveMV{

private:
    at::Tensor PL;  // Phi_left as (l s) x r
    at::Tensor CA;  // coreA as (m S) x (s n)
    at::Tensor PR;  // Phi_right as L x (S R)
    at::Tensor J;   // inverted blocks of the preconditioner
    int prec;
    at::ScalarType dt;
    int64_t r, n, R, s, S;

public:
    /**
     * @brief Construct the local operator.
     *
     * @param[in] Phi_left the left interface. Has shape r x s x r.
     * @param[in] Phi_right the right interface. Has shape R x S x R.
     * @param[in] coreA the core of the matrix. Has shape s x n x n x S.
     * @param[in] r the left rank of the local unknown.
     * @param[in] n the mode size of the local unknown.
     * @param[in] R the right rank of the local unknown.
     * @param[in] prec the preconditioner (NO_PREC, C_PREC or R_PREC).
     * @param[in] dtype the dtype used for the products.
     */
    AMENsolveMV(const at::Tensor &Phi_left, const at::Tensor &Phi_right, const at::Tensor &coreA, int64_t r, int64_t n, int64_t R, int prec, at::ScalarType dtype)
        : prec(prec), dt(dtype), r(r), n(n), R(R)
    {
        s = coreA.sizes()[0];
        S = coreA.sizes()[3];
        auto Phi_l = Phi_left.to(dtype);
        auto Phi_r = Phi_right.to(dtype);
        auto A = coreA.to(dtype);

        PL = Phi_l.reshape({r*s, r});
        CA = A.permute({1,3,0,2}).reshape({n*S, s*n});
        PR = Phi_r.reshape({R, S*R});

        if(prec == C_PREC){
            auto Jl = at::tensordot(at::diagonal(Phi_l,0,0,2), A, {0}, {0});
            auto Jr = at::diagonal(Phi_r, 0, 0, 2);
            J = at::linalg_inv(at::tensordot(Jl,Jr,{3},{0}).permute({0,3,1,2}));
        }
        else if(prec == R_PREC){
            auto Jl = at::tensordot(at::diagonal(Phi_l,0,0,2), A, {0},{0}); // sd,smnS->dmnS
            auto Jt = at::tensordot(Jl, Phi_r, {3}, {1}); // dmnS,LSR->dmnLR
            Jt = Jt.permute({0, 1, 3, 2, 4});
            std::vector<int64_t> sh(Jt.sizes().begin(), Jt.sizes().end());
            auto Jt2 = Jt.reshape({-1, Jt.sizes()[1]*Jt.sizes()[2], Jt.sizes()[3]*Jt.sizes()[4]});
            J = at::linalg_inv(Jt2).reshape(sh);
        }
    }

    /**
     * @brief The dtype used for the products.
     */
    at::ScalarType dtype() const {
        return dt;
    }

    /**
     * @brief Apply the preconditioner.
     *
     * @param[in] x the vector, with r*n*R entries.
     * @return at::Tensor the result, same shape as x.
     */
    at::Tensor apply_prec(const at::Tensor &x) const {
        auto sol = x.reshape({r, n, R});
        at::Tensor ret;
        if(prec == C_PREC) {
            at::Tensor tmp = sol.permute({0,2,1}).reshape({r, R, n, 1});
            ret = at::linalg_matmul(J, tmp).permute({0,2,1,3}).reshape({r, n, R});
        }
        else if(prec == R_PREC){
            ret = at::einsum("rnR,rmLnR->rmL", {sol, J});
        }
        else
            ret = sol;

        return ret.reshape(x.sizes());
    }

    /**
     * @brief Apply the operator to one vector or to a batch of vectors.
     *
     * @param[in] x a multiple of r*n*R entries, the vectors are consecutive.
     * @param[in] use_prec apply the preconditioner before the operator (single vector only).
     * @return at::Tensor the result, same shape as x.
     */
    at::Tensor matvec(const at::Tensor &x, bool use_prec = true) const {
        at::Tensor xx = (use_prec && prec != NO_PREC) ? apply_prec(x) : x;
        int64_t b = x.numel() / (r*n*R);

        // lsr,brnR->blsnR
        auto w = b == 1 ? at::mm(PL, xx.reshape({r, n*R})) : at::matmul(PL, xx.reshape({b, r, n*R}));
        // mSsn,blsnR->blmSR (the matrix is shared by the batch, no copy)
        auto w2 = at::matmul(CA, w.view({b*r, s*n, R}));
        // blmSR,LSR->blmL
        auto w3 = at::mm(w2.view({b*r*n, S*R}), PR.t());
        return w3.view(x.sizes());
    }
};
