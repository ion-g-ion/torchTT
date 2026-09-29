"""
Univariate bases for continuous functions with tensor-train coefficients.

Use one basis per coordinate and store the expansion coefficients in a
``torchtt.TT``. Basis evaluations and integration weights can then be contracted
with the TT cores to evaluate and integrate the function without constructing a
dense coefficient tensor. These bases also support ``torchtt.nn.TTDensityLayer``
for density estimation.
"""
from .basis import BaseBasis, BSplineBasis, GaussianBasis

__all__ = ['BaseBasis', 'BSplineBasis', 'GaussianBasis']
