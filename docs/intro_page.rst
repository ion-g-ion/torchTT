

Welcome to torchTT
==================

Tensor-Train decomposition in `pytorch`

Tensor-Train decomposition package written only in Python on top of `pytorch`. Supports GPU acceleration and automatic differentiation.
It also contains routines for solving linear systems in the TT format and performing adaptive cross approximation  (the AMEN solver/cross interpolation is inspired form the `MATLAB TT-Toolbox <https://github.com/oseledets/TT-Toolbox>`_).

Some routines are implemented in C++ for an increased execution speed.