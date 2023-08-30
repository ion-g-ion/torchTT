
r"""
Provides Tensor-Train (TT) decomposition using `pytorch` as backend.

Contains routines for computing the TT decomposition and all the basisc linear algebra in the TT format. Additionally, GPU support can be used thanks to the `pytorch` backend.   
It also has linear solvers in TT and cross approximation as well as automatic differentiation.

.. include:: INTRO.md 

"""

from .torchtt import TT, eye, zeros, kron, ones, random, randn, reshape, meshgrid , dot, elementwise_divide, numel, rank1TT, bilinear_form, diag, permute, load, save, cat 
from ._amen import amen_mm, amen_mv
from . import solvers
from . import grad
# from .grad import grad, watch, unwatch
from . import manifold
from . import interpolate
from . import nn
# from .errors import *

try:
    import torchttcpp
    _flag_use_cpp = True
except:
    import warnings
    warnings.warn(
        "\x1B[33m\nC++ implementation not available. Using pure Python.\n\033[0m")
    _flag_use_cpp = False


def cpp_enabled():
    """
    Is the C++ backend enabled?

    Returns:
        bool: the flag
    """
    return _flag_use_cpp

__all__ = ['TT', 'eye', 'zeros', 'kron', 'ones', 'random', 'randn', 'reshape', 'meshgrid' , 'dot', 'elementwise_divide', 'numel', 'rank1TT', 'bilinear_form', 'diag', 'permute', 'load', 'save', 'cat', 'amen_mm', 'amen_mv', 'cpp_available']