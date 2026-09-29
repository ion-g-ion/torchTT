"""
Products in the TT format that truncate the rank of the result.
The algorithms are implemented in :mod:`torchtt.methods`.

"""
import sys
import torchtt
from torchtt.methods import _as_method
from torchtt.errors import *


def matvec(A, x, eps=1e-12, rmax=sys.maxsize, method='dmrg', initial=None, verbose=False):
    """
    Matrix-vector product between a TT matrix and a TT tensor with rank truncation.
    Computes an approximation of ``(A @ x).round(eps, rmax)``, which can be much cheaper than forming the exact product ``A @ x`` (its rank is the product of the ranks of the operands).

    Choosing the method (see :mod:`torchtt.methods` and :ref:`products-label`):

     * ``'dmrg'`` (default): best when the product rank ``rank(A) * rank(x)`` is large and the result has a much smaller rank. Uses the C++ implementation if available.
     * ``'direct'``: forms ``A @ x`` and rounds it. Faster when the product rank is moderate (up to a few hundred) or when the result cannot be compressed much.
     * ``'amen'``: alternative to ``'dmrg'`` for real tensors, usually slower. The rank grows by at most ``kickrank`` per sweep, so give an ``initial`` guess or increase ``nswp`` if the result has a large rank.
     * ``'swap'``: no iterations and deterministic, usually slower than ``'dmrg'``.

    Examples:

        .. code-block:: python

            import torchtt as tntt

            A = tntt.randn([(4, 4)] * 8, [1] + 7 * [3] + [1])
            x = tntt.randn([4] * 8, [1] + 7 * [4] + [1])

            y = tntt.matvec(A, x, eps=1e-10)                                          # DMRG
            y = tntt.matvec(A, x, eps=1e-10, method='direct')                         # (A @ x).round(1e-10)
            y = tntt.matvec(A, x, eps=1e-10, method=tntt.methods.DMRG(nswp=40))      # DMRG with 40 sweeps
            y = tntt.matvec(A, x, eps=1e-10, method='amen', initial=y)                # AMEn with an initial guess

    Args:
        A (torchtt.TT): the TT matrix.
        x (torchtt.TT): the TT tensor.
        eps (float, optional): relative accuracy. Defaults to 1e-12.
        rmax (int, optional): maximum rank of the result. Defaults to the maximum possible integer.
        method (str | torchtt.methods.DMRG | torchtt.methods.Direct | torchtt.methods.AMEn | torchtt.methods.Swap, optional): the method, given by name (``'dmrg'``, ``'direct'``, ``'amen'`` or ``'swap'``) or as an instance to change its parameters. Defaults to 'dmrg'.
        initial (torchtt.TT, optional): initial guess of the result (only for ``'dmrg'`` and ``'amen'``). None means that the method chooses it. Defaults to None.
        verbose (bool, optional): print information about the iterations (only for ``'dmrg'`` and ``'amen'``). Defaults to False.

    Raises:
        InvalidArguments: The operands must be torchtt.TT instances.
        IncompatibleTypes: The first operand must be a TT matrix and the second a TT tensor.
        ShapeMismatch: The column shape of the TT matrix must be the shape of the TT tensor.
        ShapeMismatch: The initial guess must be a TT tensor with the row shape of the TT matrix.
        InvalidArguments: The method is unknown or does not use an initial guess.

    Returns:
        torchtt.TT: the result.
    """
    if not isinstance(A, torchtt.TT) or not isinstance(x, torchtt.TT):
        raise InvalidArguments("The operands must be torchtt.TT instances.")
    if not A.is_ttm or x.is_ttm:
        raise IncompatibleTypes("The first operand must be a TT matrix and the second a TT tensor.")
    if A.N != x.N:
        raise ShapeMismatch("The column shape of the TT matrix %s must be the shape of the TT tensor %s." % (str(A.N), str(x.N)))
    if initial is not None and (not isinstance(initial, torchtt.TT) or initial.is_ttm or initial.N != A.M):
        raise ShapeMismatch("The initial guess must be a TT tensor of shape %s." % str(A.M))

    result = _as_method(method).matvec(A, x, eps, rmax, initial, verbose)

    # the rank enrichment of the iterative methods can exceed rmax
    if max(result.R) > rmax:
        result = result.round(eps, rmax)

    return result


def matmat(A, B, eps=1e-12, rmax=sys.maxsize, method='direct', initial=None, verbose=False):
    """
    Product of two TT matrices with rank truncation.
    Computes an approximation of ``(A @ B).round(eps, rmax)``.

    Choosing the method (see :mod:`torchtt.methods` and :ref:`products-label`):

     * ``'direct'`` (default): forms ``A @ B`` and rounds it. Best when the product rank ``rank(A) * rank(B)`` is moderate (up to a few hundred) or when the result cannot be compressed much.
     * ``'amen'``: best when the product rank is large and the result has a much smaller rank (real tensors only). The rank grows by at most ``kickrank`` per sweep, so give an ``initial`` guess or increase ``nswp`` if the result has a large rank.
     * ``'swap'``: no iterations and deterministic, usually slower than ``'direct'`` and ``'amen'``.

    Examples:

        .. code-block:: python

            import torchtt as tntt

            A = tntt.randn([(4, 4)] * 8, [1] + 7 * [3] + [1])
            B = tntt.randn([(4, 4)] * 8, [1] + 7 * [4] + [1])

            C = tntt.matmat(A, B, eps=1e-10)                                           # (A @ B).round(1e-10)
            C = tntt.matmat(A, B, eps=1e-10, method=tntt.methods.AMEn(kickrank=8))    # AMEn

    Args:
        A (torchtt.TT): the first TT matrix.
        B (torchtt.TT): the second TT matrix.
        eps (float, optional): relative accuracy. Defaults to 1e-12.
        rmax (int, optional): maximum rank of the result. Defaults to the maximum possible integer.
        method (str | torchtt.methods.Direct | torchtt.methods.AMEn | torchtt.methods.Swap, optional): the method, given by name (``'direct'``, ``'amen'`` or ``'swap'``) or as an instance to change its parameters. Defaults to 'direct'.
        initial (torchtt.TT, optional): initial guess of the result (only for ``'amen'``). None means that the method chooses it. Defaults to None.
        verbose (bool, optional): print information about the iterations (only for ``'amen'``). Defaults to False.

    Raises:
        InvalidArguments: The operands must be torchtt.TT instances.
        IncompatibleTypes: Both operands must be TT matrices.
        ShapeMismatch: The column shape of the first TT matrix must be the row shape of the second.
        ShapeMismatch: The initial guess must be a TT matrix with the shape of the result.
        InvalidArguments: The method is unknown, does not implement this product or does not use an initial guess.

    Returns:
        torchtt.TT: the result.
    """
    if not isinstance(A, torchtt.TT) or not isinstance(B, torchtt.TT):
        raise InvalidArguments("The operands must be torchtt.TT instances.")
    if not A.is_ttm or not B.is_ttm:
        raise IncompatibleTypes("Both operands must be TT matrices.")
    if A.N != B.M:
        raise ShapeMismatch("The column shape of the first TT matrix %s must be the row shape of the second %s." % (str(A.N), str(B.M)))
    if initial is not None and (not isinstance(initial, torchtt.TT) or not initial.is_ttm or initial.M != A.M or initial.N != B.N):
        raise ShapeMismatch("The initial guess must be a TT matrix of shape %s x %s." % (str(A.M), str(B.N)))

    method = _as_method(method)
    if not hasattr(method, 'matmat'):
        raise InvalidArguments("The method %s does not implement the product of TT matrices. Choose between 'direct', 'amen' and 'swap'." % type(method).__name__)

    result = method.matmat(A, B, eps, rmax, initial, verbose)

    # the rank enrichment of the iterative methods can exceed rmax
    if max(result.R) > rmax:
        result = result.round(eps, rmax)

    return result


def hadamard(x, y, eps=1e-12, rmax=sys.maxsize, method='dmrg', initial=None, verbose=False):
    """
    Elementwise (Hadamard) product of two TT tensors or two TT matrices with rank truncation.
    Computes an approximation of ``(x * y).round(eps, rmax)``, which can be much cheaper than forming the exact product ``x * y`` (its rank is the product of the ranks of the operands).

    Choosing the method (see :mod:`torchtt.methods` and :ref:`products-label`):

     * ``'dmrg'`` (default): best when the product rank ``rank(x) * rank(y)`` is large and the result has a much smaller rank.
     * ``'direct'``: forms ``x * y`` and rounds it. Faster when the product rank is moderate (up to a few hundred) or when the result cannot be compressed much.
     * ``'swap'``: no iterations and deterministic, usually slower than ``'dmrg'``.

    Examples:

        .. code-block:: python

            import torchtt as tntt

            x = tntt.randn([4] * 8, [1] + 7 * [3] + [1])
            y = tntt.randn([4] * 8, [1] + 7 * [4] + [1])

            z = tntt.hadamard(x, y, eps=1e-10)                      # DMRG
            z = tntt.hadamard(x, y, eps=1e-10, method='direct')     # (x * y).round(1e-10)

    Args:
        x (torchtt.TT): the first operand.
        y (torchtt.TT): the second operand.
        eps (float, optional): relative accuracy. Defaults to 1e-12.
        rmax (int, optional): maximum rank of the result. Defaults to the maximum possible integer.
        method (str | torchtt.methods.DMRG | torchtt.methods.Direct | torchtt.methods.Swap, optional): the method, given by name (``'dmrg'``, ``'direct'`` or ``'swap'``) or as an instance to change its parameters. Defaults to 'dmrg'.
        initial (torchtt.TT, optional): initial guess of the result (only for ``'dmrg'``). None means that the method chooses it. Defaults to None.
        verbose (bool, optional): print information about the iterations (only for ``'dmrg'``). Defaults to False.

    Raises:
        InvalidArguments: The operands must be torchtt.TT instances.
        IncompatibleTypes: The operands must be both TT tensors or both TT matrices.
        ShapeMismatch: The operands must have the same shape.
        ShapeMismatch: The initial guess must have the shape of the operands.
        InvalidArguments: The method is unknown, does not implement this product or does not use an initial guess.

    Returns:
        torchtt.TT: the result.
    """
    if not isinstance(x, torchtt.TT) or not isinstance(y, torchtt.TT):
        raise InvalidArguments("The operands must be torchtt.TT instances.")
    if x.is_ttm != y.is_ttm:
        raise IncompatibleTypes("The operands must be both TT tensors or both TT matrices.")
    if x.N != y.N or (x.is_ttm and x.M != y.M):
        raise ShapeMismatch("The operands must have the same shape.")
    if initial is not None and (not isinstance(initial, torchtt.TT) or initial.is_ttm != x.is_ttm or initial.N != x.N or (x.is_ttm and initial.M != x.M)):
        raise ShapeMismatch("The initial guess must have the shape of the operands.")

    method = _as_method(method)
    if not hasattr(method, 'hadamard'):
        raise InvalidArguments("The method %s does not implement the elementwise product. Choose between 'dmrg', 'direct' and 'swap'." % type(method).__name__)

    result = method.hadamard(x, y, eps, rmax, initial, verbose)

    # the rank enrichment of the iterative methods can exceed rmax
    if max(result.R) > rmax:
        result = result.round(eps, rmax)

    return result
