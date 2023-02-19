
r"""

Provides Tensor-Train (TT) decomposition using `pytorch` as backend.

Contains routines for computing the TT decomposition and all the basisc linear algebra in the TT format. Additionally, GPU support can be used thanks to the `pytorch` backend.   
It also has linear solvers in TT and cross approximation as well as automatic differentiation.

Utilities:

 * Example scripts (and ipy notebooks) can be found in the [examples/](https://github.com/ion-g-ion/torchTT/tree/main/examples) folder.
 * Tests can be found in the [tests/](https://github.com/ion-g-ion/torchTT/tree/main/tests) folder.

"""

from .torchtt import TT, eye, zeros, kron, ones, random, randn, reshape, meshgrid , dot, elementwise_divide, numel, rank1TT, bilinear_form, diag, permute, load, save 
from . import solvers
from . import grad
# from .grad import grad, watch, unwatch
from . import manifold
from . import interpolate
from . import nn
# from .errors import *
