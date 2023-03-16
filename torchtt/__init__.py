
r"""

.. include:: INTRO.md 

"""

from .torchtt import TT, eye, zeros, kron, ones, random, randn, reshape, meshgrid , dot, elementwise_divide, numel, rank1TT, bilinear_form, diag, permute, load, save 
from . import solvers
from . import grad
# from .grad import grad, watch, unwatch
from . import manifold
from . import interpolate
from . import nn
# from .errors import *
