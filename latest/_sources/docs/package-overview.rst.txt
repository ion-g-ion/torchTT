.. _about-tt-label:

Overview
=========

What is the Tensor-Train format?
--------------------------------


The Tensor-Train (TT) format :cite:p:`oseledets2011tensor` is a low-rank tensor decomposition format used to fight the curse of dimensionality. A d-dimensional tensor \(\mathsf{x} \in \mathbb{R} ^{n_1 \times n_2 \times \cdots \times n_d}\) can be expressed using algebraic operations between d smaller tensors:

.. math::
  \mathsf{x}_{i_1i_2...i_d} = \sum\limits_{s_0=1}^{r_0} \sum\limits_{s_1=1}^{r_1} \cdots \sum\limits_{s_{d-1}=1}^{r_{d-1}} \sum\limits_{s_d=1}^{r_d} \mathsf{g}^{(1)}_{s_0 i_1 s_1} \cdots \mathsf{g}^{(d)}_{s_{d-1} i_d s_d}, 

where :math:`\mathbf{r} = (r_0,r_1,...,r_d), r_0 = r_d = 1` is the TT rank and  :math:`\mathsf{g}^{(k)} \in \mathbb{R}^{r_{k-1} \times n_k \times r_k}` are the TT cores.
The storage complexity is :math:`\mathcal{O}(nr^2d)` instead of :math:`\mathcal{O}(n^d)` if the rank remains bounded. Tensor operators :math:`\mathsf{A} \in \mathbb{R} ^{(m_1 \times m_2 \times \cdots \times m_d) \times (n_1 \times n_2 \times \cdots \times n_d)}` can be similarly expressed in the TT format as:

.. math::
  \mathsf{A}_{i_1i_2...i_d,j_1j_2...j_d} = \sum\limits_{s_0=1}^{r_0} \sum\limits_{s_1=1}^{r_1} \cdots \sum\limits_{s_{d-1}=1}^{r_{d-1}} \sum\limits_{s_d=1}^{r_d} \mathsf{h}^{(1)}_{s_0 i_1 j_1 s_1} \cdots \mathsf{h}^{(d)}_{s_{d-1} i_d j_d s_d}, \\ j_k = 1,...,m_k, \: i_k=1,...,n_k, \; \; k=1,...,d.

Tensor operators (also called tensor matrices in this library) generalize the concept of matrix-vector product to the multilinear case.

To create a `TT` object one can simply provide a tensor or the representation in terms of TT-cores. In the first case, the relative accuracy can also be provided such that 

.. math::
  || \mathsf{x} - \mathsf{y} ||_F^2 < \epsilon || \mathsf{x}||_F^2, 

where `y` is the `TT` tensor returned by the decomposition. In code, this translates to

.. code-block:: python

  import torchtt

  # tens is a torch.Tensor 
  # tens = ...

  tt = torchtt.TT(tens, 1e-10)


The rank of the object ``tt`` can be inspected using the ``print()`` function or can accessed using ``tt.R``. The tensor can be converted back to the full format using ``tt.full()``.
The TT class implements tensors in the TT format as well as tensors operators in TT format. Once in the TT format, linear algebra operations (``+``, ``-``, ``*``, ``@``, ``/``) can be performed without resorting to the full format. The format and the operations is similar to the one implemented in ``torch``.
As an example, we have the following code where 3 tensors in the TT format are involved in algebra operations:

.. code-block:: python

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


During the process, the ``round()`` function has been used. This has the role of further compressing tensors by reducing the rank. After successive linear algebra operations, the rank will overshoot and therefore it is required to perform rounding operations.

.. _qtt-label:

Quantized TT (QTT)
------------------

The quantized tensor-train (QTT) format
:cite:p:`oseledets2010approximation,khoromskij2011quantics` is the TT format applied to a
tensor that has first been reshaped so that every mode has size 2. For example, a vector of length
:math:`8 = 2^3` is reshaped into a :math:`2 \times 2 \times 2` tensor. Entry 5 of the
vector (counting from 0) becomes entry :math:`(1, 0, 1)` of the tensor, which are the
binary digits of 5. In general, a mode of size :math:`2^L` becomes :math:`L` modes of
size 2, and for a TT matrix, a mode of size :math:`(2^L, 2^L)` becomes :math:`L` modes of
size :math:`(2, 2)`.

The reshaped tensor has more modes, but they are much smaller. For data with structure,
such as smooth functions sampled on a uniform grid, the TT ranks stay small. A vector of
length :math:`n` then needs only :math:`\mathcal{O}(r^2 \log_2 n)` numbers instead of
:math:`n`, where :math:`r` bounds the TT ranks. For unstructured data, such as random
vectors, QTT does not help.

In ``torchtt``, :meth:`torchtt.TT.to_qtt` converts a tensor to the QTT format and
:meth:`torchtt.TT.qtt_to_tens` converts it back:

.. code-block:: python

  import torch
  import torchtt

  x = torch.exp(torch.linspace(0, 1, 1024, dtype=torch.float64))  # 1 mode of size 1024
  x_qtt = torchtt.TT(x).to_qtt()   # 10 modes of size 2
  print(x_qtt.N)   # [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
  print(x_qtt.R)   # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

Operations in the TT format
---------------------------

The class ``torchtt.TT`` is used to create tensors in the TT format. Passing a `torch.Tensor` to the constructor computes a TT decomposition. The accuracy ``eps`` can be provided as an additional argument. In order to recover the original tensor (also called full tensor), the ``torchtt.TT.full()`` method can be used. Tensors can be further compressed using the ``torchtt.TT.round()`` method.

Once in the TT format, linear algebra operations can be performed between compressed tensors without going to the full format. The implemented operations are:
 - Sum and difference between TT objects. Two ``torchtt.TT`` instances can be summed using the ``+`` operator. The difference can be implemented using the ``-`` operator.
 - Elementwise product (also called Hadamard product is performed using) the ``*`` operator. The same operator also implements the scalar multiplication.
 - The operator ``@`` implements the generalization of the matrix product. It can also be used between a tensor operator and a tensor.
 - The operator ``/`` implements the elementwise division of two TT objects. It is computed with the alternating minimal energy (:term:`AMEn`) method :cite:p:`dolgov2014alternating`.
 - The operator ``**`` implements the Kronecker product.

The package also includes more features such as solving multilinear systems :cite:p:`dolgov2014alternating`, cross approximation :cite:p:`oseledets2010tt,savostyanov2011fast`, automatic differentiation (see :mod:`torchtt.grad` for what can be differentiated) and TT layers for neural networks (:mod:`torchtt.nn`). Tutorials are listed in :ref:`examples-label`.

.. _products-label:

Efficient products
------------------

The rank of the exact product of two TT objects is the product of their ranks, e.g. ``(A @ x).R[k] == A.R[k] * x.R[k]``, and the same holds for ``*``.
The result is therefore usually rounded, ``(A @ x).round(eps)``, but the rounding costs :math:`\mathcal{O}(dn(r_Ar_x)^3)`, where :math:`r_Ar_x` is the product rank.
When the result has a much smaller rank than the product rank, it is cheaper to compute the rounded result without forming the exact product.
The functions :func:`torchtt.matvec`, :func:`torchtt.matmat` and :func:`torchtt.hadamard` (also available as :meth:`torchtt.TT.matvec`, :meth:`torchtt.TT.matmat` and :meth:`torchtt.TT.hadamard`) compute the rounded products, and the ``method`` argument selects the algorithm:

.. code-block:: python

  import torchtt as tntt

  y = tntt.matvec(A, x, eps=1e-10)                                          # default method (DMRG)
  y = tntt.matvec(A, x, eps=1e-10, method='direct')                         # (A @ x).round(1e-10)
  y = tntt.matvec(A, x, eps=1e-10, method=tntt.methods.DMRG(nswp=40))      # DMRG with other parameters

The methods are the classes of :mod:`torchtt.methods`. A method is given by name to use its default parameters, or as an instance to change them.

.. list-table::
   :header-rows: 1

   * - Method
     - ``matvec``
     - ``matmat``
     - ``hadamard``
     - Iterative
     - Initial guess
   * - ``'direct'``
     - yes
     - yes (default)
     - yes
     - no
     - no
   * - ``'dmrg'``
     - yes (default)
     - no
     - yes (default)
     - yes
     - yes
   * - ``'amen'``
     - yes
     - yes
     - no
     - yes
     - yes
   * - ``'swap'``
     - yes
     - yes
     - yes
     - no
     - no

Which method to choose depends mostly on how much the result can be compressed, i.e. on the ratio between the product rank and the rank of the result:

- ``'direct'`` forms the exact product and rounds it with :meth:`torchtt.TT.round`. There are no iterations and the result has the smallest rank for the accuracy ``eps``. It is the fastest method while the product rank is moderate (up to a few hundred) and whenever the result cannot be compressed much (for random operands, the product rank is also the rank of the result). Its cost and memory grow with the cube and the square of the product rank, so it becomes slow for large ranks.
- ``'dmrg'`` (:term:`DMRG`, :cite:p:`oseledets2011dmrg`) optimizes two neighbouring cores of the result at a time. Its cost depends on the rank of the result and not on the product rank, so it is the method of choice when the product rank is large and the result has a much smaller rank. The matrix-vector product has a C++ implementation. The iteration starts from a random tensor (or from the ``initial`` guess), and the returned rank can be larger than needed by up to ``kickrank``.
- ``'amen'`` (:term:`AMEn`, :cite:p:`dolgov2014alternating`) updates one core at a time and enriches the result with an approximation of the residual. It serves the same cases as ``'dmrg'`` and is the only method of this kind for the product of TT matrices. For the matrix-vector product ``'dmrg'`` is usually faster, since the cost of AMEn grows faster with the ranks of the operands. Starting from rank 1, the rank grows by at most ``kickrank + kick2`` per sweep, so for results of large rank increase ``kickrank`` or ``nswp``, or pass an ``initial`` guess. Only real tensors are supported.
- ``'swap'`` :cite:p:`michailidis2025elementwise` moves the cores of one operand through the cores of the other with :math:`d(d-1)/2` swaps of neighbouring cores, each truncated with an SVD. It needs no iterations and no initial guess, and its result is deterministic. Every swap is truncated with the accuracy ``eps``, so the error of the result can exceed ``eps``. Like for ``'dmrg'``, the cost depends mostly on the ranks of the result, but it grows quadratically with the number of dimensions and quickly with the mode sizes, and it is usually slower than ``'dmrg'``. When the intermediate tensors have large ranks (e.g. for smooth functions in the :ref:`QTT format <qtt-label>`), the ranks of the result can be much larger than needed.

In short: use the default method when the operands have large ranks and the result is expected to compress well, and ``'direct'`` when the ranks are small or the result does not compress.
The example `efficient_linalg.py <https://github.com/ion-g-ion/torchTT/tree/main/examples/efficient_linalg.py>`_ compares the methods on a product that compresses well and on one that does not.
All the methods also run on the GPU.

Continuous functions and density estimation
-------------------------------------------

The :mod:`torchtt.functional` subpackage supplies B-spline and Gaussian bases
for continuous multivariate functions whose expansion coefficients are stored
in a TT. Contracting the cores with basis values evaluates the function;
contracting them with basis integrals integrates it. This extends TT compression
from discrete arrays to continuous function approximation.

The bases are also used by :class:`torchtt.nn.TTDensityLayer` for probability
density estimation. See :ref:`functional-label` for the basis API, links to the existing
tutorials, and the connection to the density layer.

Nonlinear Transformations for TTDensityLayer
--------------------------------------------

The ``torchtt.nn.TTDensityLayer`` models a conditional probability density function. To reduce the tensor train rank required for complex, curved distributions (like banana or C-shapes), the layer can apply a learnable change-of-variables before evaluating the TT cores. We implement several modular transformations:

**Affine Transform**

The simplest transformation is an affine map:

.. math::
    \mathbf{z} = R(\boldsymbol{\theta}) \text{diag}(e^{\mathbf{a}}) \mathbf{x} + \mathbf{b}

Here, :math:`R(\boldsymbol{\theta})` is a rotation matrix assembled from Givens rotations parameterized by angles :math:`\boldsymbol{\theta}`, :math:`\mathbf{a}` are log-scales, and :math:`\mathbf{b}` are offsets. 
The Jacobian determinant is :math:`| \det J | = \prod e^{a_i}`.

**Rank-1 Volume-Preserving Shear**

To straighten curved ridges, we apply a rank-1 nonlinear shear:

.. math::
    \mathbf{z} = \mathbf{x} + \mathbf{u} g(\mathbf{v}^\top \mathbf{x} + c)

To ensure the transformation is volume-preserving (i.e., :math:`| \det J | = 1`), we project :math:`\mathbf{u}` such that :math:`\mathbf{v}^\top \mathbf{u} = 0`. 
The scalar function :math:`g(t)` is a polynomial of degree :math:`D` with no constant term:

.. math::
    g(t) = \sum_{p=1}^D \alpha_p t^p

Optionally, the displacement can be squashed with a hyperbolic tangent to prevent numerical overflow: :math:`\tilde{g}(t) = c_{clip} \tanh(g(t) / c_{clip})`.

**Triangular Polynomial Shear**

A Knothe-Rosenblatt-style mapping that is also volume-preserving:

.. math::
    z_i = x_i + \sum_{j>i} q_{ij}(x_j)

The Jacobian is unit upper-triangular, so :math:`\det J = 1` identically. Each :math:`q_{ij}(t)` is a polynomial of degree :math:`D_{poly}` with no constant term:

.. math::
    q_{ij}(t) = \sum_{p=1}^{D_{poly}} \beta_{ij,p} t^p

**Sinh-Arcsinh Warp**

An elementwise bijection used to model asymmetric skewness and heavy tails:

.. math::
    z_i = \sinh\left(e^{s_i} \text{asinh}(x_i) + b_i\right)

where :math:`s_i` is the log-scale and :math:`b_i` is the offset.
