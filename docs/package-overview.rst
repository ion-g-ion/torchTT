.. _about-tt-label:

Overview
=========

What is the Tensor-Train format?
--------------------------------


The Tensor-Train (TT) format is a low-rank tensor decomposition format used to fight the curse of dimensionality. A d-dimensional tensor \(\mathsf{x} \in \mathbb{R} ^{n_1 \times n_2 \times \cdots \times n_d}\) can be expressed using algebraic operations between d smaller tensors:

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
The TT class implements tensors in the TT format as well as tensors operators in TT format. Once in the TT format, linear algebra operations (``+``, ``-``, ``*``, ``@``, ``/``) can be performed without resorting to the full format. The format and the operations is similat to the one implemented in ``torch``.
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

About the package
-----------------

The class ``torchtt.TT`` is used to create tensors in the TT format. Passing a `torch.Tensor` to the constructor computes a TT decomposition. The accuracy ``eps`` can be provided as an additional argument. In order to recover the original tensor (also called full tensor), the ``torchtt.TT.full()`` method can be used. Tensors can be further compressed using the ``torchtt.TT.round()`` method.

Once in the TT format, linear algebra operations can be performed between compressed tensors without going to the full format. The implemented operations are:
 - Sum and difference between TT objects. Two ``torchtt.TT`` instances can be summed using the ``+`` operator. The difference can be implemented using the ``-`` operator.
 - Elementwise product (also called Hadamard product is performed using) the ``*`` operator. The same operator also implements the scalar multiplication.
 - The operator ``@`` implements the generalization of the matrix product. It can also be used between a tensor operator and a tensor.
 - The operator ``/`` implements the elementwise division of two TT objects. The algorithm is AMEn.
 - The operator ``**`` implements the Kronecker product.

The package also includes more features such as solving multilinear systems, cross approximation and automatic differentiation (with the possibility to define TT layers for neural networks ``torchtt.TT.full()``). Working examples that can be used as a tutorial are to be found in `examples/ <https://github.com/ion-g-ion/torchTT/tree/main/examples>`_.
Following example scripts (as well as python notebooks) are also provied provided as part of the documentation:

 - `basic_tutorial.py <https://github.com/ion-g-ion/torchTT/tree/main/examples/basic_tutorial.py>`_ / `basic_tutorial.ipynb <https://github.com/ion-g-ion/torchTT/tree/main/examples/basic_tutorial.ipynb>`_: This contains a basic tutorial on decomposing full tensors in the TT format as well as performing rank rounding, slicing. `Try on Google Colab <https://colab.research.google.com/github/ion-g-ion/torchTT/blob/main/examples/basic_tutorial.ipynb>`_. 
 - `basic_linalg.py <https://github.com/ion-g-ion/torchTT/tree/main/examples/basic_linalg.py>`_ / `basic_linalg.ipynb <https://github.com/ion-g-ion/torchTT/tree/main/examples/basic_linalg.ipynb>`_: This tutorial presents all the algebra operations that can be performed in the TT format. `Try on Google Colab <https://colab.research.google.com/github/ion-g-ion/torchTT/blob/main/examples/basic_linalg.ipynb>`_. 
 - `efficient_linalg.py  <https://github.com/ion-g-ion/torchTT/tree/main/examples/efficient_linalg.py>`_ / `efficient_linalg.ipynb <https://github.com/ion-g-ion/torchTT/tree/main/examples/efficient_linalg.ipynb>`_: contains the DMRG for fast matves and AMEN for elementwise inversion in the TT format `Try on Google Colab <https://colab.research.google.com/github/ion-g-ion/torchTT/blob/main/examples/efficient_linalg.ipynb>`_. 
 - `automatic_differentiation.py <https://github.com/ion-g-ion/torchTT/tree/main/examples/automatic_differentiation.py)>`_ / `automatic_differentiation.ipynp <https://github.com/ion-g-ion/torchTT/tree/main/examples/automatic_differentiation.ipynb>`_: Basic tutorial on AD in ``torchtt``. `Try on Google Colab <https://colab.research.google.com/github/ion-g-ion/torchTT/blob/main/examples/automatic_differentiation.ipynb>`_. 
 - `cross_interpolation.py <https://github.com/ion-g-ion/torchTT/tree/main/examples/cross_interpolation.py>`_ / `cross_interpolation.ipynb <https://github.com/ion-g-ion/torchTT/tree/main/examples/cross_interpolation.ipynb>`_: In this script, the cross interpolation emthod is exemplified. `Try on Google Colab <https://colab.research.google.com/github/ion-g-ion/torchTT/blob/main/examples/cross_interpolation.ipynb>`_. 
 - `system_solvers.py <https://github.com/ion-g-ion/torchTT/tree/main/examples/system_solvers.py>`_ / `system_solvers.ipynb <https://github.com/ion-g-ion/torchTT/tree/main/examples/system_solvers.ipynb>`_: This contains the bais ussage of the multilinear solvers. `Try on Google Colab <https://colab.research.google.com/github/ion-g-ion/torchTT/blob/main/examples/system_solvers.ipynb>`_. 
 - `cuda.py <https://github.com/ion-g-ion/torchTT/tree/main/examples/cuda.py>`_ / `cuda.ipynb <https://github.com/ion-g-ion/torchTT/tree/main/examples/cuda.ipynb>`_: This provides an example on how to use the GPU acceleration. `Try on Google Colab <https://colab.research.google.com/github/ion-g-ion/torchTT/blob/main/examples/cuda.ipynb>`_. 
 - `basic_nn.py <https://github.com/ion-g-ion/torchTT/tree/main/examples/basic_nn.py>`_ / `basic_nn.ipynb  <https://github.com/ion-g-ion/torchTT/tree/main/examples/basic_nn.ipynb>`_: This provides an example on how to use the TT neural network layers. `Try on Google Colab <https://colab.research.google.com/github/ion-g-ion/torchTT/blob/main/examples/basic_nn.ipynb>`_. 
 - `mnist_nn.py  <https://github.com/ion-g-ion/torchTT/tree/main/examples/mnist_nn.py>`_ / `mnist_nn.ipynb  <https://github.com/ion-g-ion/torchTT/tree/main/examples/mnist_nn.ipynb>`_: Example of TT layers used for image classification. `Try on Google Colab <https://colab.research.google.com/github/ion-g-ion/torchTT/blob/main/examples/mnist_nn.ipynb>`_. 
