.. _functional-label:

torchtt.functional: continuous function bases
=============================================

.. py:module:: torchtt.functional

The subpackage includes the abstract :class:`torchtt.functional.BaseBasis`
class and its :class:`torchtt.functional.BSplineBasis` and
:class:`torchtt.functional.GaussianBasis` implementations.

Evaluation, interpolation, and integration
------------------------------------------

Both concrete bases follow these conventions:

* ``basis.n`` is the number of basis functions.
* ``basis(x)`` returns shape ``(basis.n, *x.shape)``. The first axis indexes
  basis functions; ``basis(x, derivative=True)`` evaluates their first
  derivatives with respect to ``x``. Evaluation also supports PyTorch autograd
  with respect to the input points.
* ``points, matrix = basis.interpolating_points()`` returns shapes ``(n,)``
  and ``(n, n)``. The matrix has sample points in rows and basis functions in
  columns: ``matrix = basis(points).T``. Thus interpolation coefficients solve
  ``matrix @ coefficients = function_values``. Gaussian interpolation can be
  ill-conditioned for closely spaced centers or wide basis functions.
* ``basis.integration_weights()`` returns one integral per basis function,
  shape ``(n,)``. Spline weights include any decaying tails, and Gaussian
  weights integrate over the entire real line.

The bases are PyTorch modules. Construct and move the basis, coefficients, and
input tensors to a common device and dtype, for example with
``basis.to(device=device, dtype=torch.float64)``.

Density estimation with TTDensityLayer
--------------------------------------

:class:`torchtt.nn.TTDensityLayer` uses these bases to evaluate a normalized
probability density from TT parameters. A neural network can supply the flat
parameter vector, allowing the density to depend on conditioning data.
The layer uses squared TT coefficients with the basis values and normalizes
using their integration weights. Optional coordinate transformations include
the Jacobian factor in the density evaluation.

Examples
--------

The existing tutorials in ``examples/`` demonstrate these APIs:

* :doc:`Basis functions <../examples/basis>`: B-spline and Gaussian evaluation,
  interpolation, integration, derivatives, and spline boundary conditions.
* :doc:`Deep TT density estimation <../examples/deep_tt_density>`:
  ``TTDensityLayer`` with Gaussian bases and coordinate transformations.
* :doc:`Fokker-Planck PINN <../examples/fokker_planck_pinn>`: a neural network
  supplies the density layer's parameters as a function of time.

API reference
-------------

BaseBasis
~~~~~~~~~

.. autoclass:: torchtt.functional.BaseBasis
   :members:
   :special-members: __call__
   :show-inheritance:

BSplineBasis
~~~~~~~~~~~~

.. autoclass:: torchtt.functional.BSplineBasis
   :members:
   :special-members: __call__
   :class-doc-from: both
   :show-inheritance:

GaussianBasis
~~~~~~~~~~~~~

.. autoclass:: torchtt.functional.GaussianBasis
   :members:
   :special-members: __call__
   :class-doc-from: both
   :show-inheritance:
