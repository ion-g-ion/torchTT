---
title: 'torchTT: A PyTorch-based Tensor Train Toolbox'
tags:
  - Python
  - PyTorch
  - tensor train decomposition
  - machine learning
  - high-dimensional data
  - scientific computing
authors:
  - name: [Author Name]
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: [Institution Name]
   index: 1
date: 6 January 2026
bibliography: paper.bib
---

# Summary

`torchTT` is a Python library that enables the use of the Tensor-Train (TT) decomposition [@oseledets2011tensor] within the PyTorch framework [@paszke2019pytorch]. It provides a comprehensive set of tools for creating, manipulating, and computing with tensors in the TT format, all while leveraging PyTorch's native capabilities such as GPU acceleration and automatic differentiation. The library is designed to facilitate the solution of high-dimensional problems in scientific computing and machine learning where the curse of dimensionality makes dense tensor operations intractable.

# Statement of Need

High-dimensional data and equations appear frequently in physics, chemistry, and engineering. Storing and manipulating these high-dimensional tensors explicitly is often impossible due to exponential scaling of memory and computational requirements. The Tensor-Train decomposition offers a compact representation by expressing a high-dimensional tensor as a sequence of lower-dimensional tensors (cores).

While there are existing software packages for TT decomposition, such as `ttpy` [@ttpy] (implemented in Python/Fortran) and `TensorLy` [@kossaifi2019tensorly] (general tensor learning), there is a need for a specialized toolbox that tightly integrates with the PyTorch ecosystem. `torchTT` addresses this by offering:
1.  **Native PyTorch Integration**: Tensors in `torchTT` can be seamlessly used with other PyTorch modules.
2.  **GPU Acceleration**: Operations can be executed on GPUs without code changes, a feature often missing or less integrated in other TT libraries.
3.  **Automatic Differentiation**: By building on top of PyTorch, `torchTT` supports autograd, allowing for gradient-based optimization of TT-cores, which is essential for deep learning applications and Riemannian optimization.
4.  **Advanced Solvers**: Unlike some general-purpose tensor libraries, `torchTT` implements specialized solvers like the Alternating Minimal Energy (AMEn) method [@dolgov2014alternating] for solving linear systems and performing element-wise operations in the TT format.

# State of the Field

`torchTT` sits alongside other tensor libraries but carves out a specific niche:
*   **`ttpy`**: A reference implementation for TT methods. It is highly efficient for CPU-based scientific computing but lacks native GPU support and automatic differentiation.
*   **`tntorch`** [@tntorch]: Another PyTorch-based library for tensor networks. `torchTT` distinguishes itself by a strong focus on linear algebra solvers (AMEn) and cross-approximation methods [@oseledets2010tt] for functional approximation, in addition to deep learning layers.
*   **`TensorLy`**: A widely used library for tensor learning that supports multiple backends. While it supports TT decomposition, `torchTT` provides a more specialized API for linear algebra operations (e.g., solving $Ax=b$ directly in TT format) and cross-approximation.

# Functionality and Examples

The library is structured into several modules:
*   `torchtt`: The core module containing the `TT` class and basic linear algebra operations (addition, multiplication, norms, etc.).
*   `torchtt.solvers`: Implements the AMEn solver for solving linear systems and performing complex operations like division or element-wise functions.
*   `torchtt.nn`: Contains neural network layers (linear, convolutional) adapted for TT-tensors.
*   `torchtt.manifold`: Provides tools for Riemannian optimization on the manifold of tensors with fixed TT-ranks.
*   `torchtt.interpolate`: Implements the TT-Cross approximation method for constructing TT-tensors from function evaluations or black-box entries.

The repository includes an `examples/` directory containing basic tutorials to demonstrate the usage of these modules. These examples cover:
*   Basic tensor algebra and manipulation.
*   Simple optimization problems using Riemannian gradients.
*   Construction of TT-tensors using cross-approximation.
*   Basic usage of TT-layers in neural networks (e.g., for MNIST classification).

While the provided examples are introductory, the library's capabilities extend to complex research applications.

# Research Impact and Applications

The features provided by `torchTT` enable research and development in several computationally intensive areas:

*   **High-Dimensional PDEs and ODEs**: The AMEn solver and efficient matrix-vector products allow for the solution of Partial Differential Equations (PDEs) and large systems of Ordinary Differential Equations (ODEs) in high dimensions, breaking the curse of dimensionality.
*   **Computational Chemistry**: The toolbox can be used to represent and manipulate high-dimensional wavefunctions or potential energy surfaces.
*   **Compressed Deep Learning**: The `torchtt.nn` module facilitates the development of compact neural network architectures, reducing parameter counts and memory footprint while maintaining performance.

# AI Usage Disclosure

Artificial Intelligence tools were used to rephrase and refine the text of this paper. The core software implementation and the original draft of the content were produced by the authors.

# References

