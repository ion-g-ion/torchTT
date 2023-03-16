Provides Tensor-Train (TT) decomposition using `pytorch` as backend.

Contains routines for computing the TT decomposition and all the basisc linear algebra in the TT format. Additionally, GPU support can be used thanks to the `pytorch` backend.   
It also has linear solvers in TT and cross approximation as well as automatic differentiation.


 

 What is the Tensor-Train format?
 --------------------------------

 The Tensor-Train (TT) format is a low-rank tensor decomposition format used to fight the curse of dimensionality. A d-dimensional tensor \(\mathsf{x} \in \mathbb{R} ^{n_1 \times n_2 \times \cdots \times n_d}\) can be expressed using algebraic operations between d smaller tensors.
 


 Utilities
 ---------

 * Example scripts (and ipy notebooks) can be found in the [examples/](https://github.com/ion-g-ion/torchTT/tree/main/examples) folder.
 * Tests can be found in the [tests/](https://github.com/ion-g-ion/torchTT/tree/main/tests) folder.