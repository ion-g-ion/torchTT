 What is the Tensor-Train format?
 --------------------------------
 The Tensor-Train (TT) format is a low-rank tensor decomposition format used to fight the curse of dimensionality. A d-dimensional tensor \(\mathsf{x} \in \mathbb{R} ^{n_1 \times n_2 \times \cdots \times n_d}\) can be expressed using algebraic operations between d smaller tensors:

 $$ \mathsf{x}_{i_1i_2...i_d} = \sum\limits_{s_0=1}^{r_0} \sum\limits_{s_1=1}^{r_1} \cdots \sum\limits_{s_{d-1}=1}^{r_{d-1}} \sum\limits_{s_d=1}^{r_d} \mathsf{g}^{(1)}_{s_0 i_1 s_1} \cdots \mathsf{g}^{(d)}_{s_{d-1} i_d s_d}, $$
where \(\mathbf{r} = (r_0,r_1,...,r_d), r_0 = r_d = 1\) is the TT rank and  \(\mathsf{g}^{(k)} \in \mathbb{R}^{r_{k-1} \times n_k \times r_k}\) are the TT cores.
The storage complexity is \(\mathcal{O}(nr^2d)\) instead of \(\mathcal{O}(n^d)\) if the rank remains bounded.
The class `torchtt.TT` is used to create tensors in the TT format.

 Utilities
 ---------
 * Example scripts (and ipy notebooks) can be found in the [examples/](https://github.com/ion-g-ion/torchTT/tree/main/examples) folder.
 * Tests can be found in the [tests/](https://github.com/ion-g-ion/torchTT/tree/main/tests) folder.