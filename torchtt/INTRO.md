What is the Tensor-Train format?
-----------------

 The Tensor-Train (TT) format is a low-rank tensor decomposition format used to fight the curse of dimensionality. A d-dimensional tensor \(\mathsf{x} \in \mathbb{R} ^{n_1 \times n_2 \times \cdots \times n_d}\) can be expressed using algebraic operations between d smaller tensors:

 $$ \mathsf{x}_{i_1i_2...i_d} = \sum\limits_{s_0=1}^{r_0} \sum\limits_{s_1=1}^{r_1} \cdots \sum\limits_{s_{d-1}=1}^{r_{d-1}} \sum\limits_{s_d=1}^{r_d} \mathsf{g}^{(1)}_{s_0 i_1 s_1} \cdots \mathsf{g}^{(d)}_{s_{d-1} i_d s_d}, $$
where \(\mathbf{r} = (r_0,r_1,...,r_d), r_0 = r_d = 1\) is the TT rank and  \(\mathsf{g}^{(k)} \in \mathbb{R}^{r_{k-1} \times n_k \times r_k}\) are the TT cores.
The storage complexity is \(\mathcal{O}(nr^2d)\) instead of \(\mathcal{O}(n^d)\) if the rank remains bounded. Tensor operators \(\mathsf{A} \in \mathbb{R} ^{(m_1 \times m_2 \times \cdots \times m_d) \times (n_1 \times n_2 \times \cdots \times n_d)}\) can be similarly expressed in the TT format as:

 $$ \mathsf{A}_{i_1i_2...i_d,j_1j_2...j_d} = \sum\limits_{s_0=1}^{r_0} \sum\limits_{s_1=1}^{r_1} \cdots \sum\limits_{s_{d-1}=1}^{r_{d-1}} \sum\limits_{s_d=1}^{r_d} \mathsf{h}^{(1)}_{s_0 i_1 j_1 s_1} \cdots \mathsf{h}^{(d)}_{s_{d-1} i_d j_d s_d}, \\ j_k = 1,...,m_k, \: i_k=1,...,n_k, \; \; k=1,...,d.$$
 Tensor operators (also called tensor matrices in this library) generalize the concept of matrix-vector product to the multilinear case.

To create a `TT` object one can simply provide a tensor or the representation in terms of TT-cores. In the first case, the relative accuracy can also be provided such that 
$$ || \mathsf{x} - \mathsf{y} ||_F^2 < \epsilon || \mathsf{x}||_F^2, $$
where `y` is the `TT` tensor returned by the decomposition. In code, this translates to
```
import torchtt

# tens is a torch.Tensor 
# tens = ...

tt = torchtt.TT(tens, 1e-10)
```
The rank of the object `tt` can be inspected using the `print()` function or can accessed using `tt.R`. The tensor can be converted back to the full format using `tt.full()`.
The TT class implements tensors in the TT format as well as tensors operators in TT format. Once in the TT format, linear algebra operations (`+`, `-`, `*`, `@`, `/`) can be performed without resorting to the full format. The format and the operations is similat to the one implemented in `torch`.
As an example, we have the following code where 3 tensors in the TT format are involved in algebra operations:

```
import torchTT
import torch

# generate 2 random tensors and a tensor matrix
a = torchtt.randn([4,5,6,7],[1,2,3,4,1])
b = torchtt.randn([8,4,6,4],[1,2,5,2,1])
A = torchtt.randn([(4,8), (5,4) ,(6,6) (7,4)],[1,2,3,2,1])

x = a * ( A @ b )
x = x.round(1e-12)
y = x-2*a

# this is equivalent to 
yf = x.full() - 2*(a.full()*torch.einsum('ijklabcd,abcd->ijkl', A.full(), b.full()))
```
During the process, the `round()` function has been used. This has the role of further compressing tensors by reducing the rank. After successive linear algebra operations, the rank will overshoot and therefore it is required to perform rounding operations.

About the package
-----------------

The class `torchtt.TT` is used to create tensors in the TT format. Passing a `torch.Tensor` to the constructor computes a TT decomposition. The accuracy `eps` can be provided as an additional argument. In order to recover the original tensor (also called full tensor), the `torchtt.TT.full()` method can be used. Tensors can be further compressed using the `torchtt.TT.round()` method.

Once in the TT format, linear algebra operations can be performed between compressed tensors without going to the full format. The implemented operations are:
 
 * Sum and difference between TT objects. Two `torchtt.TT` instances can be summed using the `+` operator. The difference can be implemented using the `-` operator.
 * Elementwise product (also called Hadamard product is performed using) the `*` operator. The same operator also implements the scalar multiplication.
 * The operator `@` implements the generalization of the matrix product. It can also be used between a tensor operator and a tensor.
 * The operator `/` implements the elementwise division of two TT objects. The algorithm is AMEn.
 * The operator `**` implements the Kronecker product.

The package also includes more features such as solving multilinear systems, cross approximation and automatic differentiation (with the possibility to define TT layers for neural networks`torchtt.TT.full()). Working examples that can be used as a tutorial are to be found in [examples/](https://github.com/ion-g-ion/torchTT/tree/main/examples).

Utilities
-----------------

 * Example scripts (and ipy notebooks) can be found in the [examples/](https://github.com/ion-g-ion/torchTT/tree/main/examples) folder.
 * Tests can be found in the [tests/](https://github.com/ion-g-ion/torchTT/tree/main/tests) folder.

