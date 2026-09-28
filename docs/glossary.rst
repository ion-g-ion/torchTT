.. _glossary-label:

Glossary
========

Abbreviations used in the documentation, the examples and the API reference.

.. glossary::
   :sorted:

   AD
      Automatic differentiation. Derivatives of a computation are obtained by applying the
      chain rule to its elementary operations. ``torchtt`` builds on the PyTorch autograd
      engine, so gradients can be taken with respect to the TT cores (see
      :mod:`torchtt.grad`).

   AMEn
      Alternating minimal energy method :cite:p:`dolgov2014alternating`. It updates one
      TT core at a time and enriches the approximation with a low-rank approximation of
      the residual, so the TT ranks adapt during the iteration. ``torchtt`` uses it to
      solve linear systems (:func:`torchtt.solvers.amen_solve`), for matrix products
      (:func:`torchtt.amen_mv`, :func:`torchtt.amen_mm`), for elementwise division
      (:func:`torchtt.elementwise_divide`) and for cross interpolation
      (``method='amen'`` in :func:`torchtt.interpolate.function_interpolate`).

   BiCGSTAB
      Biconjugate gradient stabilized method, an iterative solver for non-symmetric linear
      systems. :func:`torchtt.solvers.amen_solve` can use it for its local systems
      (``local_solver=2``).

   DMRG
      Density matrix renormalization group. The name comes from quantum physics
      :cite:p:`white1992density`; for the TT format it denotes schemes that optimize two
      neighbouring TT cores at a time, which lets the TT ranks adapt
      :cite:p:`oseledets2011dmrg,savostyanov2011fast`.
      ``torchtt`` uses it for matrix-vector products (:meth:`torchtt.TT.fast_matvec`),
      elementwise products (:func:`torchtt.dmrg_hadamard`) and cross interpolation
      (``method='dmrg'`` in :func:`torchtt.interpolate.function_interpolate`).

   GMRES
      Generalized minimal residual method, an iterative solver for non-symmetric linear
      systems. :func:`torchtt.solvers.amen_solve` uses it for its local systems by default
      (``local_solver=1``).

   QR
      QR decomposition. It factors a matrix into a matrix with orthonormal columns
      (:math:`Q`) and an upper triangular matrix (:math:`R`). It is used to orthogonalize
      TT cores.

   QTT
      Quantized tensor train :cite:p:`oseledets2010approximation,khoromskij2011quantics`: the
      TT format applied to a tensor that has been reshaped so that every mode has size 2.
      See :ref:`qtt-label`.

   SVD
      Singular value decomposition. Truncating it is how TT ranks are reduced, both when a
      full tensor is decomposed and when a TT tensor is rounded.

   TT
      Tensor train :cite:p:`oseledets2011tensor`. A format that
      represents a :math:`d`-dimensional tensor by :math:`d` three-dimensional tensors, the
      TT cores. See :ref:`about-tt-label`.

   TTM
   TT matrix
      A linear operator (tensor operator) in the TT format. Its cores are
      four-dimensional, with one row and one column index each. A ``torchtt.TT`` is a TT
      matrix when its shape is given as a list of ``(rows, columns)`` tuples.
