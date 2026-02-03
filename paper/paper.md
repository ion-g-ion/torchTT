---
title: 'torchTT: A PyTorch-based Tensor-Train Toolbox'
tags:
  - Python
  - PyTorch
  - tensor decomposition
  - tensor train
  - high-dimensional data
  - scientific computing
authors:
  - name: Ion Gabriel Ion
    orcid: 0000-0002-2932-0396
    affiliation: 1
affiliations:
 - name: Terra Quantum AG
   index: 1
date: 6 January 2026
bibliography: paper.bib
---

# Summary

`torchTT` is a Python library enabling Tensor-Train (TT) decomposition [@oseledets2011tensor] within the PyTorch framework [@paszke2019pytorch]. It allows users to create and manipulate high-dimensional tensors in compressed TT format using PyTorch-like syntax and semantics. Operations such as basic linear algebra, matrix-vector products, and Kronecker products can be performed directly on the compressed representation without ever forming the full tensor. All computations leverage PyTorch’s capabilities: TT instances live on the same device as their underlying data, facilitating seamless GPU acceleration, and automatic differentiation works through TT operations. Under the hood, `torchTT` exploits low-rank structure in high-dimensional arrays, dramatically reducing memory and computation compared to dense tensors. These features make `torchTT` especially suitable for problems with millions or billions of degrees of freedom where explicit storage is impossible.

# Statement of Need

High-dimensional tensors arise in many areas of science and engineering, such as the discretization of $d$-dimensional PDEs, state spaces for ODE systems, and many-body wavefunctions in quantum chemistry. Naively storing or computing with a full tensor scales exponentially in $d$ (the curse of dimensionality). The TT decomposition combats this by factorizing a tensor into a sequence of smaller, low-rank 3D cores.

However, existing TT software (e.g., `ttpy` [@ttpy], `TensorLy` [@kossaifi2019tensorly]) often lacks tight integration with modern machine learning tools or specialized solvers. `torchTT` addresses this gap by providing a native PyTorch implementation. Compared to `ttpy`, it offers native GPU acceleration and automatic differentiation, enabling seamless integration with deep learning workflows. In contrast to `tntorch` [@tntorch] and `TensorLy`, `torchTT` provides specialized capabilities for solving linear systems and constructing tensors from function evaluations (cross-approximation). This combination of features makes it particularly suitable for scientific machine learning, where one must frequently compute with compressed tensors or solve large linear systems directly in TT format.

# Software Design

`torchTT` is designed to provide a high-level, Pythonic interface for tensor-train operations while maintaining the performance and flexibility required for scientific computing and deep learning.

## Core architecture and dual backend

The central component of the library is the `torchtt.TT` class, which provides a unified representation for both TT-tensors and TT-matrices (tensor operators). Key design features include:

*   **Abstraction**: TT-tensors and TT-matrices are abstracted into a single class that wraps the underlying TT representation (a list of TT cores stored as PyTorch tensors), so users do not need to manipulate cores directly.
*   **Python-like API**: Operations such as addition (`+`), subtraction (`-`), elementwise multiplication (`*`), and matrix multiplication (`@`) are overloaded to work directly on `torchtt.TT` instances.
*   **Device management**: Standard PyTorch methods like `.to(device)`, `.cuda()`, and `.cpu()` are supported, enabling TT objects and their computations to run on CPU or GPU.
*   **Seamless switching to a C++ backend**: In performance-critical routines, `torchTT` can transparently use an optional C++ extension when available (while keeping a pure-PyTorch fallback), enabling higher performance in selected algorithms.

# Functionality and Features

Beyond the core class, the `torchTT` library is organized into specialized modules targeting specific scientific and machine learning applications:

*   **Linear System Solvers** (`torchtt.solvers`): Implements advanced solvers for linear systems $Ax=b$ directly in the TT format, including the AMEn method [@dolgov2014alternating]. These solvers dynamically adapt the TT ranks during iteration to maintain accuracy. The module also includes elementwise division and inversion routines.
*   **Cross Approximation** (`torchtt.interpolate`): Provides adaptive TT-cross interpolation methods [@oseledets2010tt]. This module constructs TT approximations of multivariate functions from black-box evaluations, essential for tasks where the tensor is too large to be formed explicitly but can be sampled.
*   **Neural Network Layers** (`torchtt.nn`): Defines PyTorch-compatible neural network layers, such as `LinearLayerTT`. These layers parametrize weights as TT-matrices, enabling massive parameter reduction. They inherit from `torch.nn.Module`, making them drop-in replacements for standard layers in deep learning pipelines.
*   **Manifold Optimization** (`torchtt.manifold`): Offers tools for Riemannian optimization on the manifold of tensors with fixed TT-rank. This includes projections onto the tangent space and Riemannian gradient calculation, facilitating advanced optimization tasks like tensor completion.

## Example: Solving a 4D PDE

The following example demonstrates how to solve a 4-dimensional Poisson equation $\Delta u = f$ on $[0,1]^4$ with zero boundary conditions using `torchTT`. The right-hand side $f$ is constructed using cross-approximation, and the system is solved directly in the TT format using the AMEn solver. Additional runnable examples are provided in the repository's `examples/` folder.

```python
import torchtt as tntt
import torch as tn

# Define the problem size
N, d = 64, 4 

# Construct the 1D Laplacian operator (finite difference)
L1d = (tn.diag(tn.ones(N-1),-1) + tn.diag(tn.ones(N-1),1) - \
      2*tn.eye(N)) / (1/(N-1))**2
L1d[0,1] = L1d[-1,-2] = 0 # Boundary conditions
L1d_tt = tntt.TT(L1d, shape=[(N,N)])

# Construct the d-dimensional Laplacian
L = L1d_tt ** tntt.eye([N]*3) + tntt.eye([N]) ** L1d_tt ** tntt.eye([N]*2) + \
    tntt.eye([N]*2) ** L1d_tt ** tntt.eye([N]) + tntt.eye([N]*3) ** L1d_tt 

# Create the grid for cross approximation
x = tntt.meshgrid([tn.linspace(0,1,N)]*d)

# Approximate the RHS f(x) = x1(x1-1)*...*xd(xd-1)
f = tntt.interpolate.function_interpolate(lambda x: tn.prod(x * (x - 1), dim=1), x, eps=1e-8)

# Solve the linear system Ax = f using AMEn on GPU
u = tntt.solvers.amen_solve(L.to('cuda'), f.to('cuda'), eps=1e-6)
```

# Research Impact and Applications

`torchTT` opens up new possibilities for research in domains plagued by high dimensionality:

*   **High-Dimensional PDEs/ODEs**: `torchTT` can solve PDEs and dynamical systems in high dimensions by representing solution spaces and operators in TT form. Built-in solvers like AMEn make it practical to compute time steps of multi-dimensional PDEs, breaking the curse of dimensionality. Recent applications include isogeometric analysis for parameter-dependent geometries [@ion2022tensor_iga] and solving the chemical master equation for parameter inference [@ion2022tensor_cme].
*   **Computational Physics and Chemistry**: The library enables efficient representation of high-dimensional objects like quantum many-body wavefunctions. Operators can be applied via TT-matrix-vector products, and observables computed via TT inner products, facilitating large-scale simulations. Furthermore, it supports analyzing entanglement scaling in Matrix Product State (MPS) representations and constructing shallow quantum circuits [@bohun2024entanglement].
*   **Machine Learning with Compressed Models**: TT-layers enable significant compression of model parameters in deep learning. `torchTT`'s compatibility with autograd allows for end-to-end training of compressed architectures and efficient on-device inference.
*   **Data Science and Function Approximation**: The library is useful for surrogate modeling and uncertainty quantification, where high-dimensional response surfaces can be approximated adaptively using cross-approximation routines. Novel methods for local interpolation allow for constructing fine-scale TT representations from coarse-grid approximations with error guarantees [@guzman2026local].

# Acknowledgements

The authors thank the Terra Quantum AG team for their valuable discussions, feedback, and support.

# AI Usage Disclosure

Artificial Intelligence tools were used to rephrase and refine the text of this paper. The core software implementation and the original draft of the content were produced by the authors.

# References
