#pragma once

#include <vector>
#include <iostream>


extern "C"{
    int dgemm_(char *, char *, int64_t *, int64_t *, int64_t *, double *, double *, int64_t *, double *, int64_t *,  double *, double *, int64_t *);
    int sgemm_(char *, char *, int64_t *, int64_t *, int64_t *, float *, float *, int64_t *, float *, int64_t *,  float *, float *, int64_t *);
    double dnrm2_(int64_t *, double *, int64_t *);
    float snrm2_(int64_t *, float *, int64_t *);
    int daxpy_(int64_t*, double*, double*, int64_t*, double*, int64_t*);
    int saxpy_(int64_t*, float*, float*, int64_t*, float*, int64_t*);
    void dscal_(int64_t*, double*, double*, int64_t*);
    void sscal_(int64_t*, float*, float*, int64_t*);
    double ddot_(int64_t*, double*, int64_t*, double*, int64_t*);
    float sdot_(int64_t*, float*, int64_t*, float*, int64_t*);

    void dcopy_(int64_t*, double*, int64_t*, double*, int64_t*);
    void scopy_(int64_t*, float*, int64_t*, float*, int64_t*);

    void dgesv_(int64_t *, int64_t *, double *, int64_t *, int64_t *, double *, int64_t *, int64_t *);
    void sgesv_(int64_t *, int64_t *, float *, int64_t *, int64_t *, float *, int64_t *, int64_t *);

    void domatcopy_(int64_t *m, int64_t *n, double *alpha, double *a, int64_t *lda, double *b, int64_t *ldb);
    void somatcopy_(int64_t *m, int64_t *n, float *alpha, float *a, int64_t *lda, float *b, int64_t *ldb);
} 

namespace BLAS{
    /**
     * @brief Computes a matrix-matrix product and adds the result to a matrix.
     *
     * This function computes a matrix-matrix product of the form C = alpha * op(A) * op(B) + beta * C,
     * where op(X) = X if trans == 'N', and op(X) = X^T if trans == 'T' or 'C'.
     *
     * @tparam T The data type of the matrices (float or double).
     * @param transA Specifies whether to transpose matrix A ('T') or not ('N').
     * @param transB Specifies whether to transpose matrix B ('T') or not ('N').
     * @param m The number of rows of matrix op(A) and matrix C.
     * @param n The number of columns of matrix op(B) and matrix C.
     * @param k The number of columns of matrix op(A) and rows of matrix op(B).
     * @param alpha The scalar alpha.
     * @param A The m x k matrix A (if transA == 'N') or k x m matrix A^T (if transA == 'T').
     * @param LDA The leading dimension of matrix A. LDA >= max(1,m) if transA == 'N' and LDA >= max(1,k) otherwise.
     * @param B The k x n matrix B (if transB == 'N') or n x k matrix B^T (if transB == 'T').
     * @param LDB The leading dimension of matrix B. LDB >= max(1,k) if transB == 'N' and LDB >= max(1,n) otherwise.
     * @param beta The scalar beta.
     * @param C The m x n matrix C.
     * @param LDC The leading dimension of matrix C. LDC >= max(1,m).
     */
    template <typename T> 
    void gemm(char *transA, char *transB, int64_t *m, int64_t *n, int64_t *k, T *alpha, T *A, int64_t *LDA, T *B, int64_t *LDB, T *beta, T *C, int64_t *LDC );

   /**
     * @brief Computes the Euclidean norm of a vector.
     *
     * This function computes the Euclidean norm of a vector x, defined as ||x||_2 = sqrt(x^T * x).
     *
     * @tparam T The data type of the vector elements (float or double).
     * @param n The number of elements in the vector.
     * @param x The vector of length n.
     * @param incx The stride between consecutive elements of the vector. incx > 0.
     * @return The Euclidean norm of the vector.
     */
    template <typename T> 
    T nrm2(int64_t *n, T *x, int64_t *incx);

    /**
     * @brief Computes a vector-scalar product and adds the result to a vector.
     *
     * This function computes a vector-scalar product, defined as y = alpha * x + y, where alpha is a scalar
     * and x and y are vectors of the same length. The operation is performed in-place, so the result
     * overwrites the input vector y.
     *
     * @tparam T The data type of the vector elements (float or double).
     * @param n The number of elements in the vectors.
     * @param alpha The scalar value by which to multiply the vector x.
     * @param x The vector of length n.
     * @param incx The stride between consecutive elements of the vector x. incx > 0.
     * @param y The vector of length n to which the result is added.
     * @param incy The stride between consecutive elements of the vector y. incy > 0.
     */
    template <typename T>
    void axpy(int64_t* N, T* alpha, T* X, int64_t* incX, T* Y, int64_t* incY);

    /**
     * @brief Scales a vector by a scalar value.
     * 
     * This function multiplies each element in a vector x by a scalar value alpha, overwriting the
     * original values in x.
     * 
     * @tparam T The data type of the elements in the vector.
     * @param n The length of the vector.
     * @param alpha The scalar value.
     * @param x The vector to scale.
     * @param incx The stride between consecutive elements in x.
     */
    template <typename T>
    void scal(int64_t* n, T* alpha, T* x, int64_t* incx);

    /**
     * @brief Computes the dot product of two vectors x and y.
     * 
     * This function computes the dot product of two vectors x and y, which is defined as:
     * 
     * dot(x, y) = sum(x_i * y_i) for i = 1 to n
     * 
     * where n is the length of the vectors and x_i and y_i are the ith elements of x and y, respectively.
     * 
     * @tparam T The data type of the elements in the vectors.
     * @param n The length of the vectors.
     * @param x The first vector.
     * @param incx The stride between consecutive elements in x.
     * @param y The second vector.
     * @param incy The stride between consecutive elements in y.
     * @return The dot product of x and y.
     */
    template <typename T> 
    T dot(int64_t *n, T *x, int64_t *incx, T *y, int64_t *incy);

    template <typename T>
    void copy(int64_t* n, T* x, int64_t* incx, T* y, int64_t* incy);

    template <typename T>
    int gesv(int64_t *, int64_t *, T *, int64_t *, int64_t *, T *, int64_t *, int64_t *);

    
    /// matrix multiplication
    //specialized for double
    template <>
    void gemm<double>(char *transA, char *transB, int64_t *m, int64_t *n, int64_t *k, double *alpha, double *A, int64_t *LDA, double *B, int64_t *LDB, double *beta, double *C, int64_t *LDC ){
        dgemm_(transA, transB, m, n, k, alpha, A, LDA, B, LDB, beta, C, LDC);
    }
    // specialized for float
    template <>
    void gemm<float>(char *transA, char *transB, int64_t *m, int64_t *n, int64_t *k, float *alpha, float *A, int64_t *LDA, float *B, int64_t *LDB, float *beta, float *C, int64_t *LDC ){
        sgemm_(transA, transB, m, n, k, alpha, A, LDA, B, LDB, beta, C, LDC);
    }

    /// norm 
    // specialized for double
    template <>
    double nrm2<double>(int64_t *n, double *x, int64_t *incx){
        return dnrm2_(n, x, incx);
    }
    // specialized for float
    template <>
    float nrm2<float>(int64_t *n, float *x, int64_t *incx){
        return snrm2_(n, x, incx);
    }

    /// Multiplication with scalar
    // specialized for double
    template <>
    void axpy<double>(int64_t* N, double* alpha, double* X, int64_t* incX, double* Y, int64_t* incY) {
        daxpy_(N, alpha, X, incX, Y, incY);
    }
    // specialized for float
    template <>
    void axpy<float>(int64_t* N, float* alpha, float* X, int64_t* incX, float* Y, int64_t* incY) {
        saxpy_(N, alpha, X, incX, Y, incY);
    }

    /// scale a vector
    // Specialization for double
    template <>
    void scal<double>(int64_t* n, double* alpha, double* x, int64_t* incx) {
        dscal_(n, alpha, x, incx);
    }
    // Specialization for float
    template <>
    void scal<float>(int64_t* n, float* alpha, float* x, int64_t* incx) {
        sscal_(n, alpha, x, incx);
    }

    /// dot product
    // specialized for double
    template <>
    double dot<double>(int64_t *n, double *x, int64_t *incx, double *y, int64_t *incy){
        return ddot_(n, x, incx, y, incy);
    }
    // specialized for float
    template <>
    float dot<float>(int64_t *n, float *x, int64_t *incx, float *y, int64_t *incy){
        return sdot_(n, x, incx, y, incy);
    }

    template <>
    void copy(int64_t* n, double* x, int64_t* incx, double* y, int64_t* incy){
        dcopy_(n, x, incx, y, incy);
    }

    template <>
    void copy(int64_t* n, float* x, int64_t* incx, float* y, int64_t* incy){
        scopy_(n, x, incx, y, incy);
    }


}

namespace LAPACK{

    template<typename T>
    int64_t gesv(int64_t n, int64_t nrhs,  T * A, int64_t lda, int64_t *ipiv, T *B, int64_t ldb);


    template <>
    int64_t gesv(int64_t n, int64_t nrhs,  double * A, int64_t lda, int64_t *ipiv, double *B, int64_t ldb){
        int64_t info;
        
        if(ipiv != nullptr)
            dgesv_(&n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
        else
        {
            int64_t *IPIV = new int64_t[n];
            dgesv_(&n, &nrhs, A, &lda, IPIV, B, &ldb, &info);
            delete [] IPIV;
        }

        return info;
    }

    template <>
    int64_t gesv(int64_t n, int64_t nrhs,  float * A, int64_t lda, int64_t *ipiv, float *B, int64_t ldb){
        int64_t info;
        
        if(ipiv != nullptr)
            sgesv_(&n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
        else
        {
            int64_t *IPIV = new int64_t[n];
            sgesv_(&n, &nrhs, A, &lda, IPIV, B, &ldb, &info);
            delete [] IPIV;
        }

        return info;
    }
}