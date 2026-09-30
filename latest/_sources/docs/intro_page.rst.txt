

Welcome to torchTT
==================

Tensor-Train decomposition in `pytorch`

Tensor-Train (TT) decomposition package written only in Python on top of `pytorch`. Supports GPU acceleration and automatic differentiation.
It also contains methods for solving multilinear systems :cite:p:`dolgov2014alternating` and for adaptive cross approximation :cite:p:`oseledets2010tt` in the TT format, inspired by the `MATLAB TT-Toolbox <https://github.com/oseledets/TT-Toolbox>`_.
Abbreviations used throughout the documentation are explained in the :ref:`glossary-label`, and the cited publications are listed in the :ref:`references-label`.

Some routines are implemented in C++ for an increased execution speed.

``torchtt`` helps researchers and engineers tackle high-dimensional problems in
scientific computing by working directly with compressed representations of large
arrays.
It is designed for PyTorch users, making it easy to replace dense tensor operations
with operations in the tensor-train (TT) format.
For problems that admit low-rank TT representations, ``torchtt`` can make
high-dimensional experiments practical even on a laptop.
Users should be familiar with linear algebra and the basics of the TT format; the
`basic tutorial <https://github.com/ion-g-ion/torchTT/blob/main/examples/basic_tutorial.ipynb>`_
provides a starting point.
Example applications include solving linear systems arising from discretized
high-dimensional differential equations, such as the Fokker–Planck and chemical
master equations; approximating black-box multivariate functions for surrogate
modeling and uncertainty quantification; and compressing neural-network weight
matrices to reduce the number of trainable parameters.
