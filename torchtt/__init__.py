
r"""
# TorchTT

Provides Tensor-Train (TT) decomposition using pytorch as backend.

Contains routines for computing the TT decomposition and all the basisc linar algebra in the TT format     
It also has linear solvers in TT and cross approximation as well as automatic differentiation.
"""

from .torchtt import TT, eye, zeros, kron, ones, random, randn, reshape, meshgrid , dot, elementwise_divide, numel, rank1TT 
from .solvers import amen_solve
from .grad import grad, watch, unwatch
from .manifold import riemannian_gradient, riemannian_projection
from .interpolate import function_interpolate, dmrg_cross
