"""
Methods for the products that truncate the rank of the result: :func:`torchtt.matvec`, :func:`torchtt.matmat` and :func:`torchtt.hadamard`.

Each function computes an approximation of the product followed by rounding, e.g. ``(A @ x).round(eps, rmax)``.
The algorithm is chosen with the ``method`` argument, which is either the name of the method or an instance of one of the classes below.
Use an instance to change the parameters of the method:

.. code-block:: python

    import torchtt as tntt

    y = tntt.matvec(A, x, eps=1e-10)                                          # default method ('dmrg')
    y = tntt.matvec(A, x, eps=1e-10, method='direct')                         # other method, default parameters
    y = tntt.matvec(A, x, eps=1e-10, method=tntt.methods.DMRG(nswp=40))      # other parameters

The methods and the products they implement:

.. list-table::
   :header-rows: 1

   * - Method
     - ``matvec``
     - ``matmat``
     - ``hadamard``
     - Use it when
   * - ``'direct'`` (:class:`Direct`)
     - yes
     - yes (default)
     - yes
     - the product rank ``rank(a) * rank(b)`` is moderate (up to a few hundred) or the result cannot be compressed much.
   * - ``'dmrg'`` (:class:`DMRG`)
     - yes (default)
     - no
     - yes (default)
     - the product rank is large and the result has a much smaller rank.
   * - ``'amen'`` (:class:`AMEn`)
     - yes
     - yes
     - no
     - the product of TT matrices has a large product rank and a result of small rank (real tensors only).
   * - ``'swap'`` (:class:`Swap`)
     - yes
     - yes
     - yes
     - a non-iterative and deterministic algorithm is needed.

See :ref:`products-label` for a longer discussion.
"""

from dataclasses import dataclass
import torch as tn
import opt_einsum as oe
import torchtt
from torchtt._dmrg import dmrg_matvec, dmrg_hadamard_python
from torchtt._amen import _amen_mm_python
from torchtt._fast_mult import swap_cores
from torchtt.errors import *


def _no_initial_guess(method, initial):
    """
    Raise an error if an initial guess is passed to a method that does not use one.

    Args:
        method (object): the method.
        initial (None | torchtt.TT): the initial guess.

    Raises:
        InvalidArguments: The method does not use an initial guess.
    """
    if initial is not None:
        raise InvalidArguments("The method %s does not use an initial guess." % type(method).__name__)


@dataclass(frozen=True)
class Direct():
    """
    Compute the exact product and round it with :meth:`torchtt.TT.round` :cite:p:`oseledets2011tensor`.

    The rank of the exact product is the product of the ranks of the operands and the rounding costs :math:`\\mathcal{O}(dn(r_ar_b)^3)`.
    It is the best choice when the product rank :math:`r_ar_b` is moderate (up to a few hundred) or when the result cannot be compressed much, since no iterations are needed.
    Its cost and memory become large for large product ranks; use :class:`DMRG` or :class:`AMEn` if the result has a much smaller rank.
    The result is deterministic and its rank is the smallest one for the accuracy ``eps``.

    Implements all the products and does not use an initial guess.
    """

    def matvec(self, A, x, eps, rmax, initial, verbose):
        """
        Matrix-vector product. Called by :func:`torchtt.matvec`, which also checks the arguments.

        Args:
            A (torchtt.TT): the TT matrix.
            x (torchtt.TT): the TT tensor.
            eps (float): relative accuracy.
            rmax (int): maximum rank.
            initial (None): not used, must be None.
            verbose (bool): not used.

        Returns:
            torchtt.TT: the result.
        """
        _no_initial_guess(self, initial)
        return (A @ x).round(eps, rmax)

    def matmat(self, A, B, eps, rmax, initial, verbose):
        """
        Product of two TT matrices. Called by :func:`torchtt.matmat`, which also checks the arguments.

        Args:
            A (torchtt.TT): the first TT matrix.
            B (torchtt.TT): the second TT matrix.
            eps (float): relative accuracy.
            rmax (int): maximum rank.
            initial (None): not used, must be None.
            verbose (bool): not used.

        Returns:
            torchtt.TT: the result.
        """
        _no_initial_guess(self, initial)
        return (A @ B).round(eps, rmax)

    def hadamard(self, x, y, eps, rmax, initial, verbose):
        """
        Elementwise product. Called by :func:`torchtt.hadamard`, which also checks the arguments.

        Args:
            x (torchtt.TT): the first operand.
            y (torchtt.TT): the second operand.
            eps (float): relative accuracy.
            rmax (int): maximum rank.
            initial (None): not used, must be None.
            verbose (bool): not used.

        Returns:
            torchtt.TT: the result.
        """
        _no_initial_guess(self, initial)
        return (x * y).round(eps, rmax)


@dataclass(frozen=True)
class DMRG():
    """
    Density matrix renormalization group (DMRG) method :cite:p:`oseledets2011dmrg`.

    Optimizes two neighbouring cores of the result at a time, without forming the exact product.
    The cost depends on the rank of the result and not on the product of the ranks of the operands, so it pays off when the result has a much smaller rank than the product rank.
    If the result cannot be compressed, :class:`Direct` is faster.
    The iteration starts from a random tensor (or from the ``initial`` guess) and stops when the cores change by less than ``eps`` or after ``nswp`` sweeps.
    The ranks of the result can exceed the smallest ones for the accuracy ``eps`` by up to ``kickrank``; use :meth:`torchtt.TT.round` if this matters.

    Implements the matrix-vector product (with a C++ implementation) and the elementwise product.

    Args:
        nswp (int, optional): maximum number of sweeps. Defaults to 20.
        kickrank (int, optional): rank enrichment with random vectors after each local update. Defaults to 4.
        use_cpp (bool, optional): use the C++ implementation if available (only for the matrix-vector product). Defaults to True.

    Raises:
        InvalidArguments: nswp must be a positive integer.
        InvalidArguments: kickrank must be a non-negative integer.
    """
    nswp: int = 20
    kickrank: int = 4
    use_cpp: bool = True

    def __post_init__(self):
        if not isinstance(self.nswp, int) or self.nswp < 1:
            raise InvalidArguments("nswp must be a positive integer.")
        if not isinstance(self.kickrank, int) or self.kickrank < 0:
            raise InvalidArguments("kickrank must be a non-negative integer.")

    def matvec(self, A, x, eps, rmax, initial, verbose):
        """
        Matrix-vector product. Called by :func:`torchtt.matvec`, which also checks the arguments.

        Args:
            A (torchtt.TT): the TT matrix.
            x (torchtt.TT): the TT tensor.
            eps (float): relative accuracy.
            rmax (int): maximum rank.
            initial (None | torchtt.TT): initial guess of the result (None means random). It is not modified.
            verbose (bool): print information about the sweeps.

        Returns:
            torchtt.TT: the result.
        """
        # the Python implementation changes the cores of the initial guess
        y0 = None if initial is None else initial.clone()
        return dmrg_matvec(A, x, y0=y0, nswp=self.nswp, eps=eps, rmax=rmax, kickrank=self.kickrank, verb=verbose, use_cpp=self.use_cpp)

    def hadamard(self, x, y, eps, rmax, initial, verbose):
        """
        Elementwise product. Called by :func:`torchtt.hadamard`, which also checks the arguments.
        TT matrices are multiplied as TT tensors with the mode sizes ``M[k]*N[k]``.

        Args:
            x (torchtt.TT): the first operand.
            y (torchtt.TT): the second operand.
            eps (float): relative accuracy.
            rmax (int): maximum rank.
            initial (None | torchtt.TT): initial guess of the result (None means random). It is not modified.
            verbose (bool): print information about the sweeps.

        Returns:
            torchtt.TT: the result.
        """
        is_ttm = x.is_ttm
        if is_ttm:
            # merge the row and column modes and multiply as TT tensors
            M, N = x.M, x.N
            x = torchtt.TT([tn.reshape(c, [c.shape[0], -1, c.shape[-1]]) for c in x.cores])
            y = torchtt.TT([tn.reshape(c, [c.shape[0], -1, c.shape[-1]]) for c in y.cores])
            if initial is not None:
                initial = torchtt.TT([tn.reshape(c, [c.shape[0], -1, c.shape[-1]]) for c in initial.cores])

        # the Python implementation changes the cores of the initial guess
        z0 = None if initial is None else initial.clone()
        z = dmrg_hadamard_python(x, y, z0, self.nswp, eps, rmax, self.kickrank, verbose)

        if is_ttm:
            z = torchtt.TT([tn.reshape(c, [c.shape[0], m, n, c.shape[-1]]) for c, m, n in zip(z.cores, M, N)])

        return z


@dataclass(frozen=True)
class AMEn():
    """
    Alternating minimal energy (AMEn) method :cite:p:`dolgov2014alternating`.

    Updates one core of the result at a time and enriches it with an approximation of the residual, without forming the exact product.
    Like :class:`DMRG`, it pays off when the result has a much smaller rank than the product rank, and it is the only such method for the product of TT matrices.
    For the matrix-vector product :class:`DMRG` is usually faster, since the cost of AMEn grows faster with the ranks of the operands.
    Without an initial guess it starts from rank 1 and the rank grows by at most ``kickrank + kick2`` per sweep.
    A result with larger rank than ``1 + nswp * (kickrank + kick2)`` is therefore not reached and the returned tensor is not accurate: increase ``nswp`` or ``kickrank``, or pass an ``initial`` guess.

    Implements the matrix-vector product and the product of TT matrices, for real tensors only.

    Args:
        nswp (int, optional): maximum number of sweeps. Defaults to 22.
        kickrank (int, optional): rank enrichment with the residual after each local update. Defaults to 4.
        kick2 (int, optional): additional rank enrichment with random vectors. Defaults to 0.

    Raises:
        InvalidArguments: nswp must be a positive integer.
        InvalidArguments: kickrank and kick2 must be non-negative integers.
    """
    nswp: int = 22
    kickrank: int = 4
    kick2: int = 0

    def __post_init__(self):
        if not isinstance(self.nswp, int) or self.nswp < 1:
            raise InvalidArguments("nswp must be a positive integer.")
        if not isinstance(self.kickrank, int) or self.kickrank < 0 or not isinstance(self.kick2, int) or self.kick2 < 0:
            raise InvalidArguments("kickrank and kick2 must be non-negative integers.")

    def matvec(self, A, x, eps, rmax, initial, verbose):
        """
        Matrix-vector product. Called by :func:`torchtt.matvec`, which also checks the arguments.

        Args:
            A (torchtt.TT): the TT matrix.
            x (torchtt.TT): the TT tensor.
            eps (float): relative accuracy.
            rmax (int): maximum rank.
            initial (None | torchtt.TT): initial guess of the result (None means a zero tensor of rank 1).
            verbose (bool): print information about the sweeps.

        Raises:
            IncompatibleTypes: AMEn does not support complex tensors.

        Returns:
            torchtt.TT: the result.
        """
        if A.cores[0].is_complex() or x.cores[0].is_complex():
            raise IncompatibleTypes("AMEn does not support complex tensors. Use the method 'dmrg' or 'direct'.")

        return _amen_mm_python(A.cores, [c[:, :, None, :] for c in x.cores], A.M, [1]*len(A.M), A.N, False, self.nswp, None if initial is None else initial.cores, None if initial is None else initial.R, eps, rmax, self.kickrank, self.kick2, verbose)

    def matmat(self, A, B, eps, rmax, initial, verbose):
        """
        Product of two TT matrices. Called by :func:`torchtt.matmat`, which also checks the arguments.

        Args:
            A (torchtt.TT): the first TT matrix.
            B (torchtt.TT): the second TT matrix.
            eps (float): relative accuracy.
            rmax (int): maximum rank.
            initial (None | torchtt.TT): initial guess of the result (None means a zero TT matrix of rank 1).
            verbose (bool): print information about the sweeps.

        Raises:
            IncompatibleTypes: AMEn does not support complex tensors.

        Returns:
            torchtt.TT: the result.
        """
        if A.cores[0].is_complex() or B.cores[0].is_complex():
            raise IncompatibleTypes("AMEn does not support complex tensors. Use the method 'direct'.")

        return _amen_mm_python(A.cores, B.cores, A.M, B.N, A.N, True, self.nswp, None if initial is None else initial.cores, None if initial is None else initial.R, eps, rmax, self.kickrank, self.kick2, verbose)


@dataclass(frozen=True)
class Swap():
    """
    Product by swapping neighbouring cores :cite:p:`michailidis2025elementwise`.

    The cores of the first operand are moved through the cores of the second one with :math:`d(d-1)/2` swaps of neighbouring cores, each truncated with an SVD.
    There are no iterations and no initial guess, so the result is deterministic.
    Every swap is truncated with the relative accuracy ``eps``, so the error of the result can be larger than ``eps``.
    Like for :class:`DMRG`, the cost depends mostly on the ranks of the result, but it grows quadratically with the number of dimensions and quickly with the mode sizes, and it is usually slower than :class:`DMRG`.
    When the intermediate tensors have large ranks (e.g. for smooth functions in the QTT format), the ranks of the result can be much larger than needed.

    Implements all the products and does not use an initial guess.
    """

    def matvec(self, A, x, eps, rmax, initial, verbose):
        """
        Matrix-vector product. Called by :func:`torchtt.matvec`, which also checks the arguments.

        Args:
            A (torchtt.TT): the TT matrix.
            x (torchtt.TT): the TT tensor.
            eps (float): relative accuracy of each swap.
            rmax (int): maximum rank.
            initial (None): not used, must be None.
            verbose (bool): not used.

        Returns:
            torchtt.TT: the result.
        """
        _no_initial_guess(self, initial)
        d = len(A.N)

        cores = [tn.permute(c, [2, 1, 0]) for c in x.cores[::-1]]
        for i in range(d):
            cores[0] = oe.contract("mabk,kbn->man", A.cores[d-i-1], cores[0])

            if i != d-1:
                for j in range(i, -1, -1):
                    cores[j], cores[j+1] = swap_cores(cores[j], cores[j+1], eps, rmax)

        return torchtt.TT(cores)

    def matmat(self, A, B, eps, rmax, initial, verbose):
        """
        Product of two TT matrices. Called by :func:`torchtt.matmat`, which also checks the arguments.

        Args:
            A (torchtt.TT): the first TT matrix.
            B (torchtt.TT): the second TT matrix.
            eps (float): relative accuracy of each swap.
            rmax (int): maximum rank.
            initial (None): not used, must be None.
            verbose (bool): not used.

        Returns:
            torchtt.TT: the result.
        """
        _no_initial_guess(self, initial)
        d = len(A.N)

        cores = [tn.permute(c, [3, 1, 2, 0]) for c in B.cores[::-1]]
        for i in range(d):
            cores[0] = oe.contract("mabk,kbcn->macn", A.cores[d-i-1], cores[0])

            if i != d-1:
                for j in range(i, -1, -1):
                    cores[j], cores[j+1] = swap_cores(cores[j], cores[j+1], eps, rmax)

        return torchtt.TT(cores)

    def hadamard(self, x, y, eps, rmax, initial, verbose):
        """
        Elementwise product. Called by :func:`torchtt.hadamard`, which also checks the arguments.

        Args:
            x (torchtt.TT): the first operand.
            y (torchtt.TT): the second operand.
            eps (float): relative accuracy of each swap.
            rmax (int): maximum rank.
            initial (None): not used, must be None.
            verbose (bool): not used.

        Returns:
            torchtt.TT: the result.
        """
        _no_initial_guess(self, initial)
        d = len(x.N)

        if x.is_ttm:
            cores = [tn.permute(c, [3, 1, 2, 0]) for c in y.cores[::-1]]
            for i in range(d):
                cores[0] = oe.contract("maAk,kbBn,AB,ab->maAn", x.cores[d-i-1], cores[0], tn.eye(x.N[d-i-1], device=x.cores[d-i-1].device, dtype=cores[0].dtype), tn.eye(x.M[d-i-1], device=x.cores[d-i-1].device, dtype=cores[0].dtype))

                if i != d-1:
                    for j in range(i, -1, -1):
                        cores[j], cores[j+1] = swap_cores(cores[j], cores[j+1], eps, rmax)
        else:
            cores = [tn.permute(c, [2, 1, 0]) for c in y.cores[::-1]]
            for i in range(d):
                cores[0] = oe.contract("mak,kbn,ab->man", x.cores[d-i-1], cores[0], tn.eye(x.N[d-i-1], device=x.cores[d-i-1].device, dtype=cores[0].dtype))

                if i != d-1:
                    for j in range(i, -1, -1):
                        cores[j], cores[j+1] = swap_cores(cores[j], cores[j+1], eps, rmax)

        return torchtt.TT(cores)


def _as_method(method):
    """
    Return the instance of the method given by name or by instance.

    Args:
        method (str | Direct | DMRG | AMEn | Swap): the method.

    Raises:
        InvalidArguments: Unknown method name.
        InvalidArguments: The method must be a string or an instance of Direct, DMRG, AMEn or Swap.

    Returns:
        Direct | DMRG | AMEn | Swap: the method.
    """
    if isinstance(method, str):
        if method == 'direct':
            return Direct()
        elif method == 'dmrg':
            return DMRG()
        elif method == 'amen':
            return AMEn()
        elif method == 'swap':
            return Swap()
        else:
            raise InvalidArguments("Unknown method '%s'. Choose between 'direct', 'dmrg', 'amen' and 'swap'." % method)
    elif isinstance(method, (Direct, DMRG, AMEn, Swap)):
        return method
    else:
        raise InvalidArguments("The method must be a string or an instance of torchtt.methods.Direct, DMRG, AMEn or Swap.")
