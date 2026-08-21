"""
Basis functions for functional tensor decompositions.

This module provides basis function classes for use in functional tensor train 
representations. The bases can be used for interpolation, quadrature, and other 
numerical methods.

All implementations are pure PyTorch to support automatic differentiation.
"""
import torch
import numpy as np
import math
from abc import ABC, abstractmethod
from typing import Tuple


class BaseBasis(ABC, torch.nn.Module):
    """
    Abstract base class for all basis functions.
    
    All basis classes must implement the following:
        - `n` property: the dimension (number of basis functions)
        - `__call__`: evaluate the basis at given points
        - `__repr__`: string representation
        - `interpolating_points`: return points where the basis evaluation matrix is invertible
        - `integration_weights`: return the integral of each basis function
    """
    
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def n(self) -> int:
        """
        The dimension of the basis (number of basis functions).
        
        Returns:
            int: the number of basis functions.
        """
        pass
    
    @abstractmethod
    def __call__(self, x: torch.Tensor, derivative: bool = False) -> torch.Tensor:
        """
        Evaluate the basis functions at the given points.
        
        Each basis function is evaluated at every point in `x`, and the results
        are stacked along a new first dimension.
        
        Args:
            x (torch.Tensor): points where the basis is evaluated. Arbitrary shape `(...)`.
            derivative (bool, optional): if True, evaluate the derivative. Defaults to False.
            
        Returns:
            torch.Tensor: the basis evaluated at x. Shape `(n, ...)` where n is the 
                basis dimension and `...` is the shape of the input.
        """
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        """
        Return a string representation of the basis.
        
        Returns:
            str: string representation.
        """
        pass
    
    @abstractmethod
    def interpolating_points(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the interpolating points and the basis evaluated at these points.
        
        The returned matrix (from evaluating the basis at these points) is guaranteed 
        to be invertible (non-singular). This property is essential for interpolation:
        given function values at these points, one can uniquely determine the 
        coefficients of the basis expansion by solving the linear system.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - points: the interpolating points as a vector of shape `(n,)`
                - matrix: the basis evaluated at these points, shape `(n, n)`, invertible
        """
        pass
    
    @abstractmethod
    def integration_weights(self) -> torch.Tensor:
        """
        Return the integrals of each individual basis function.
        
        Computes the integral of each basis function over its support,
        which is useful for various numerical methods and tensor operations.
        
        Returns:
            torch.Tensor: a vector of shape `(n,)` containing the integral
                of each basis function.
        """
        pass


class BSplineBasis(BaseBasis):
    """
    B-spline basis functions (pure PyTorch implementation).

    This class implements a univariate B-spline basis defined by a knot vector and
    polynomial degree using the Cox-de Boor recursion formula. The implementation
    is fully differentiable via PyTorch autograd.

    The behavior at each boundary can be selected via the `bc` argument:

        - ``"clamped"`` (default): the boundary knot is repeated, the basis is
          clamped and the first/last basis function reaches 1 at the boundary.
        - ``"zero"``: like ``"clamped"``, but the basis function that is nonzero
          at the boundary is omitted. Every function in the span (and hence any
          expansion) vanishes at that boundary. Reduces `n` by one per side.
        - ``"decay"``: the domain becomes unbounded on that side. The knot vector
          is extended with `deg` uniformly spaced phantom knots and the basis
          functions whose support crosses the boundary knot are continued beyond
          it by tails of the form sum_{j=1..deg} a_j * exp(-j * rate * |x - boundary|),
          with the `deg` coefficients matched in value and the first deg-1
          derivatives. The basis is therefore C^(deg-1) on the whole unbounded
          domain and all integrals remain finite. The crossing functions whose
          Greville center would fall *outside* [a, b] are dropped, so the
          outermost retained function peaks at the boundary and decays
          monotonically past it (no redundant bump in the tail). As a result the
          partition of unity is only approximate within roughly `deg` knot spans
          of a decay boundary.

    The number of basis functions is `n = len(knots) + deg - 1`, reduced by the
    dropped boundary functions: one per ``"zero"`` side and the number of
    out-of-domain crossing functions per ``"decay"`` side (roughly deg/2).

    Attributes:
        n (int): the dimension of the basis (number of B-spline functions)
        deg (int): the polynomial degree of the B-splines

    Example:
        >>> knots = torch.linspace(0, 1, 5)
        >>> basis = BSplineBasis(knots, deg=3)
        >>> x = torch.linspace(0, 1, 100, requires_grad=True)
        >>> B = basis(x)  # shape: (n, 100), supports autograd
        >>> B.sum().backward()  # gradients flow through
        >>>
        >>> # Supports arbitrary input shapes
        >>> x_batch = torch.rand(32, 10, requires_grad=True)
        >>> B_batch = basis(x_batch)  # shape: (n, 32, 10)
        >>>
        >>> # Sparse evaluation: only the deg+1 functions that are nonzero at
        >>> # each point, plus their indices -- never forms the (n, ...) matrix
        >>> values, indices = basis.eval_sparse(x)  # both shape: (deg+1, 100)
        >>>
        >>> # Density-friendly variant: vanishes at 0, unbounded to the right
        >>> basis_pdf = BSplineBasis(knots, deg=3, bc=("zero", "decay"))

    Note:
        Evaluation is local: the knot span of each point is found by a binary
        search and only the `deg + 1` B-splines that are nonzero there are
        computed, by a vectorised de Boor recursion. The cost is `O(deg^2)` per
        point and is independent of the number of knots, and the whole recursion
        is a handful of elementwise tensor operations, which keeps it fast on
        both CPU and GPU and differentiable through autograd. `eval_sparse`
        returns that compressed result directly; `__call__` scatters it into the
        dense `(n, ...)` matrix. The mass, stiffness and advection matrices are
        assembled from the same local blocks, so they too scale linearly in `n`.

    Note:
        For ``"decay"`` sides the exponential tails are matched to the spline jet
        at the boundary. The tails are the general polynomial in the mapped
        variable `t = 1 - exp(-rate * |x - boundary|)`, which sends the unbounded
        side onto `[0, 1)`, constrained to vanish at `t = 1` so that they stay
        integrable; in `x` this is a sum of `deg` geometrically spaced decaying
        exponentials with no growing factor. With the default rate `deg / h` the
        tails are non-negative and monotone to machine precision. Rates much
        smaller than `deg / h` can still produce a small negative undershoot;
        prefer increasing the spacing of the boundary knots over lowering the
        rate if heavier tails are needed.
    """

    _BC_MODES = ("clamped", "zero", "decay")

    def __init__(self, knots: torch.Tensor, deg: int, bc="clamped", decay_rate=None):
        """
        Initialize a B-spline basis.

        Args:
            knots (torch.Tensor): the interior knot vector (at least 2 entries).
                For "clamped"/"zero" sides the boundary knot is automatically
                repeated `deg` times; for "decay" sides `deg` phantom knots with
                the spacing of the adjacent knot interval are appended instead.
            deg (int): the polynomial degree of the B-splines (e.g., 3 for cubic).
            bc (str or tuple[str, str], optional): boundary mode for the
                (left, right) side, each one of "clamped", "zero" or "decay".
                A single string applies to both sides. Modes other than
                "clamped" require deg >= 1. Defaults to "clamped".
            decay_rate (float or tuple[float, float], optional): exponential decay
                rate(s) for "decay" sides; the slowest term of a tail behaves like
                exp(-decay_rate * |x - boundary|), so this sets the far-field rate.
                A None entry uses the default `deg / h`, where h is the adjacent
                knot spacing. Defaults to None.
        """
        super().__init__()
        if not isinstance(knots, torch.Tensor):
            knots = torch.tensor(knots, dtype=torch.float64)

        knots = knots.flatten().to(torch.float64)
        if knots.numel() < 2:
            raise ValueError("At least two knots are required.")

        self._deg = int(deg)

        if isinstance(bc, str):
            bc = (bc, bc)
        bc = tuple(bc)
        if len(bc) != 2 or any(side not in self._BC_MODES for side in bc):
            raise ValueError(f"bc must be one of {self._BC_MODES} per side, got {bc}.")
        if self._deg == 0 and bc != ("clamped", "clamped"):
            raise ValueError("Boundary modes 'zero' and 'decay' require deg >= 1.")
        self._bc = bc

        # Knot spacing adjacent to each boundary (sets the phantom knot spacing
        # and the scaling of the tail variable for "decay" sides)
        self._h = (float(knots[1] - knots[0]), float(knots[-1] - knots[-2]))

        if decay_rate is None or not isinstance(decay_rate, (tuple, list)):
            decay_rate = (decay_rate, decay_rate)
        rates = []
        for side in range(2):
            if bc[side] == "decay":
                if self._h[side] <= 0:
                    raise ValueError("A 'decay' side needs a positive adjacent knot spacing.")
                rate = decay_rate[side]
                rate = self._deg / self._h[side] if rate is None else float(rate)
                if rate <= 0:
                    raise ValueError("decay_rate must be positive.")
                rates.append(rate)
            else:
                rates.append(None)
        self._decay_rates = tuple(rates)
        # Decay rates in the scaled tail variable s = |x - boundary| / h
        self._mu = tuple(None if r is None else r * h for r, h in zip(rates, self._h))

        # Total number of B-splines on the extended knot vector
        self._n_splines = knots.size(0) + self._deg - 1

        # Extend knots per side: boundary repetitions for clamped/zero, phantom
        # knots for decay (registered as buffer, not parameter)
        if bc[0] == "decay":
            left = knots[0] - self._h[0] * torch.arange(self._deg, 0, -1, dtype=torch.float64)
        else:
            left = knots[0].expand(self._deg)
        if bc[1] == "decay":
            right = knots[-1] + self._h[1] * torch.arange(1, self._deg + 1, dtype=torch.float64)
        else:
            right = knots[-1].expand(self._deg)
        self.register_buffer('_knots', torch.cat([left, knots, right]))
        self.register_buffer('_interior_knots', knots.clone())

        self._interval = (float(knots[0].item()), float(knots[-1].item()))

        # Gather offsets used by the local evaluation:
        # tw[a] = t[k - deg + a] for the knot window, and the basis indices
        # k - deg .. k of the deg+1 B-splines that are nonzero on span k
        self.register_buffer('_win_offsets',
                             torch.arange(-self._deg, self._deg + 2), persistent=False)
        self.register_buffer('_idx_offsets',
                             torch.arange(-self._deg, 1), persistent=False)
        # Span search bounds, resolved lazily per dtype (see _span_bounds)
        self._span_bounds_cache = None

        # Retained range [_lo, _hi) of the extended B-spline set:
        #   - "zero" drops the single boundary function (it is the only one
        #     nonzero at the boundary, so the span vanishes there);
        #   - "decay" drops the crossing functions whose Greville center lies
        #     strictly outside [a, b]. Those would peak in the tail and produce
        #     a redundant bump; dropping them leaves the outermost decay
        #     function peaking at the boundary and decaying monotonically past
        #     it. self._tail_keep records how many of the deg crossing functions
        #     survive on each side (for the analytic tail terms below).
        a, b = self._interval
        tol = 1e-9 * (1.0 + max(abs(a), abs(b)))

        def _greville(i):
            return float(self._knots[i + 1:i + self._deg + 1].sum()) / self._deg

        n_drop = [0, 0]
        if bc[0] == "decay":
            n_drop[0] = sum(1 for i in range(self._deg) if _greville(i) < a - tol)
        if bc[1] == "decay":
            n_drop[1] = sum(1 for i in range(self._n_splines - self._deg, self._n_splines)
                            if _greville(i) > b + tol)

        self._lo = 1 if bc[0] == "zero" else n_drop[0]
        self._hi = self._n_splines - (1 if bc[1] == "zero" else n_drop[1])
        self._n = self._hi - self._lo
        self._tail_keep = (self._deg - n_drop[0], self._deg - n_drop[1])
        if self._n < 1:
            raise ValueError("Too few knots for the requested degree and boundary modes.")

        # Coefficients of the exponential-sum tails for each "decay" side
        for side, name in enumerate(('_tail_a_left', '_tail_a_right')):
            if bc[side] == "decay":
                a = self._compute_tail_coefficients(side)
            else:
                a = torch.zeros(0, 0, dtype=torch.float64)
            self.register_buffer(name, a)
    
    @property
    def n(self) -> int:
        """
        The dimension of the basis (number of basis functions).
        
        Returns:
            int: the number of basis functions.
        """
        return self._n
    
    @property
    def deg(self) -> int:
        """
        The polynomial degree of the B-splines.
        
        Returns:
            int: the degree.
        """
        return self._deg
    
    @property
    def interval(self) -> Tuple[float, float]:
        """
        The interval on which the basis is defined.
        
        Returns:
            Tuple[float, float]: (start, end) of the interval.
        """
        return self._interval
    
    @property
    def knots(self) -> torch.Tensor:
        """
        The extended knot vector (including boundary repetitions for
        "clamped"/"zero" sides and phantom knots for "decay" sides).

        Returns:
            torch.Tensor: the knot vector.
        """
        return self._knots.clone()

    @property
    def bc(self) -> Tuple[str, str]:
        """
        The boundary modes for the (left, right) side.

        Returns:
            Tuple[str, str]: each one of "clamped", "zero" or "decay".
        """
        return self._bc

    @property
    def decay_rate(self) -> Tuple[float, float]:
        """
        The exponential decay rates per side (None for non-"decay" sides).

        Returns:
            Tuple[float, float]: the (left, right) decay rates.
        """
        return self._decay_rates
    
    def _apply(self, *args, **kwargs):
        """
        Invalidate the cached span bounds whenever the buffers are moved or cast.

        `nn.Module.to`, `.float()`, `.double()`, `.cuda()` and friends all funnel
        through `_apply`. A cast can collapse neighbouring knots (and is not
        undone by casting back), so which spans are degenerate has to be
        re-derived from the knots as they are stored afterwards.
        """
        self._span_bounds_cache = None
        return super()._apply(*args, **kwargs)

    def _span_bounds(self) -> Tuple[int, int]:
        """
        First and last knot span that covers [a, b] and is non-degenerate.

        Points outside [a, b] have no span of their own and are attached to the
        nearest of these two, so both must be non-degenerate for the Cox-de Boor
        denominators to stay positive. Degeneracy is a property of the knots as
        they are currently stored: casting the basis to a coarser dtype can
        collapse two neighbouring knots onto the same value, so the bounds are
        resolved against `self._knots` and cached per dtype rather than fixed at
        construction time.

        Returns:
            Tuple[int, int]: the (first, last) usable span index.
        """
        if self._span_bounds_cache is None or self._span_bounds_cache[0] != self._knots.dtype:
            gaps = ((self._knots[1:] - self._knots[:-1]) > 0)[self._deg:self._n_splines]
            gaps = gaps.nonzero().flatten()
            if gaps.numel() == 0:
                raise ValueError(
                    f"The knot vector has no non-degenerate span in {self._knots.dtype}.")
            self._span_bounds_cache = (self._knots.dtype,
                                       self._deg + int(gaps[0]), self._deg + int(gaps[-1]))
        return self._span_bounds_cache[1], self._span_bounds_cache[2]

    def _spans(self, x: torch.Tensor) -> torch.Tensor:
        """
        Locate the knot span of every point.

        Returns the index `k` with `t_k <= x < t_{k+1}` on the extended knot
        vector. Only the `deg + 1` B-splines `B_{k-deg}, ..., B_k` are nonzero on
        that span; this is what makes the evaluation cost independent of the
        number of knots.

        The search is a pure ordering decision -- no arithmetic and no tolerance
        -- so the interval a point is assigned to is exact, and a point sitting
        exactly on a knot lands in the span starting there (the right-continuous
        convention). Because the search returns the *last* knot not exceeding x,
        the span it reports is always non-degenerate; only points outside [a, b],
        which are clamped onto a boundary span, need the bounds of
        `_span_bounds` to guarantee the same.

        Args:
            x (torch.Tensor): flattened points of shape (m,).

        Returns:
            torch.Tensor: integer span indices of shape (m,).
        """
        t = self._knots
        dtype = torch.promote_types(x.dtype, t.dtype)
        k = torch.searchsorted(t.to(dtype).contiguous(), x.to(dtype).contiguous(), right=True)
        return (k - 1).clamp_(*self._span_bounds())

    def _local_values(self, x: torch.Tensor, k: torch.Tensor, derivative: bool = False) -> torch.Tensor:
        """
        Cox-de Boor recursion restricted to the deg+1 B-splines that are nonzero
        on the span of each point.

        The triangular scheme of de Boor is run in parallel over all points, with
        Python loops of length `deg` only: the work per point is O(deg^2) and
        does not grow with the number of knots. Every quantity is a slice of a
        single gathered knot window, so the whole recursion is a handful of
        elementwise kernels on tensors of shape (<= deg+1, m).

        For derivatives the recursion stops one level early and the last level
        applies B'_{i,p} = p * [B_{i,p-1}/(t_{i+p}-t_i) - B_{i+1,p-1}/(t_{i+p+1}-t_{i+1})].

        The x-dependence cancels in every denominator, leaving knot differences
        that always straddle the (non-degenerate) span of the point and are
        therefore strictly positive: no division guard is needed, and repeated
        knots cannot produce NaNs.

        Args:
            x (torch.Tensor): flattened points of shape (m,), inside [a, b].
            k (torch.Tensor): span indices of shape (m,), as returned by `_spans`.
            derivative (bool, optional): if True, evaluate the first derivative.

        Returns:
            torch.Tensor: values of shape (deg+1, m); row q holds the value of
                the B-spline with index `k - deg + q`.
        """
        p = self._deg
        m = x.shape[0]
        dtype = torch.promote_types(x.dtype, self._knots.dtype)

        # Degree 0 is piecewise constant: its derivative vanishes
        if derivative and p == 0:
            return torch.zeros(1, m, dtype=dtype, device=x.device)

        # Knot window of every point: tw[a] = t[k - deg + a] for a = 0..2*deg+1.
        # One gather; all knots used below are contiguous slices of it.
        tw = self._knots.to(dtype)[self._win_offsets.unsqueeze(1) + k.unsqueeze(0)]
        xr = x.to(dtype).unsqueeze(0)

        # Degree 0: the single B-spline that is nonzero on the span
        N = torch.ones(1, m, dtype=dtype, device=x.device)

        for j in range(1, (p - 1 if derivative else p) + 1):
            lo = tw[p + 1 - j:p + 1]          # t_{k+r+1-j}, r = 0..j-1
            hi = tw[p + 1:p + 1 + j]          # t_{k+r+1},   r = 0..j-1
            tmp = N / (hi - lo)
            N = (torch.nn.functional.pad((hi - xr) * tmp, (0, 0, 0, 1))
                 + torch.nn.functional.pad((xr - lo) * tmp, (0, 0, 1, 0)))

        if derivative:
            g = N / (tw[p + 1:2 * p + 1] - tw[1:p + 1])
            N = p * (torch.nn.functional.pad(g, (0, 0, 1, 0))
                     - torch.nn.functional.pad(g, (0, 0, 0, 1)))

        return N

    def _eval_spline_raw(self, x: torch.Tensor, derivative: bool = False) -> torch.Tensor:
        """
        Evaluate all B-splines of the extended knot vector (dense, no boundary modes).

        No basis functions are dropped and no exponential tails are added; the
        values are those of the polynomial B-splines on [a, b] and zero outside.

        Args:
            x (torch.Tensor): flattened points of shape (m,)
            derivative (bool, optional): if True, evaluate the first derivative.

        Returns:
            torch.Tensor: values of shape (n_splines, m).
        """
        a, b = self._interval
        k = self._spans(x)
        vals = self._local_values(x.clamp(a, b), k, derivative)
        vals = vals * ((x >= a) & (x <= b)).unsqueeze(0)
        idx = k.unsqueeze(0) + self._idx_offsets.unsqueeze(1)
        out = torch.zeros(self._n_splines, x.shape[0], dtype=vals.dtype, device=x.device)
        return out.scatter_add_(0, idx, vals).to(x.dtype)

    def _compute_tail_coefficients(self, side: int) -> torch.Tensor:
        """
        Compute the coefficients of the exponential-sum tails.

        For a "decay" side, the `deg` basis functions whose support crosses the
        boundary are continued beyond it by

            T(s) = sum_{j=1..deg} a_j * exp(-j * mu * s),

        where s = |x - boundary| / h is the scaled distance to the boundary and
        mu = rate * h. This is the general polynomial in the mapped variable
        t = 1 - exp(-mu * s), which sends the unbounded side onto [0, 1),
        constrained to vanish at t = 1 (i.e. at infinity) so that the tail is
        integrable. Vanishing at t = 1 is what drops the constant term of the
        polynomial, so no term of T(s) is a bare exponential multiplied by a
        growing factor: every term decays. That is what keeps the matched tails
        non-negative and monotone at the default rate.

        The `deg` coefficients are fixed by matching the value and the first
        deg-1 derivatives of the spline at the boundary, so the basis stays
        C^(deg-1) across it. Writing the boundary jet as J_k = d^k/ds^k, the
        conditions are the Vandermonde system sum_j a_j * (-j*mu)^k = J_k.

        Args:
            side (int): 0 for the left boundary, 1 for the right.

        Returns:
            torch.Tensor: tail coefficients of shape (deg, deg); row i holds
                (a_1, ..., a_deg) for the i-th crossing basis function.
        """
        p = self._deg
        mu = self._mu[side]

        # Sample the boundary-adjacent polynomial piece at Chebyshev points of
        # the scaled span variable s in (-1, 0)
        theta = (2.0 * torch.arange(p + 1, dtype=torch.float64) + 1.0) * (math.pi / (2 * (p + 1)))
        s = -(1.0 + torch.cos(theta)) / 2.0
        if side == 1:
            xs = self._interval[1] + self._h[1] * s
            vals = self._eval_spline_raw(xs)[self._n_splines - p:]
        else:
            xs = self._interval[0] - self._h[0] * s
            vals = self._eval_spline_raw(xs)[:p]

        # Recover the piece coefficients in s (exact: the piece is a polynomial
        # of degree <= deg on the boundary span)
        V = s.unsqueeze(1) ** torch.arange(p + 1, dtype=torch.float64)
        piece = torch.linalg.solve(V, vals.t()).t()

        # Boundary jet in s: J_k = d^k/ds^k at s = 0 = k! * piece_k
        factorials = torch.tensor([float(math.factorial(k)) for k in range(p)], dtype=torch.float64)
        jet = piece[:, :p] * factorials

        # Match the jet: A_{kj} = (-j*mu)^k, solve A @ a = jet^T
        rates = -mu * torch.arange(1, p + 1, dtype=torch.float64)
        A = rates.unsqueeze(0) ** torch.arange(p, dtype=torch.float64).unsqueeze(1)
        return torch.linalg.solve(A, jet.t()).t()

    def _eval_tail(self, x: torch.Tensor, side: int, derivative: bool = False) -> torch.Tensor:
        """
        Evaluate the exponential tails of the crossing functions of a "decay" side.

        Args:
            x (torch.Tensor): flattened points of shape (m,)
            side (int): 0 for the left boundary, 1 for the right.
            derivative (bool, optional): if True, evaluate the first derivative.

        Returns:
            torch.Tensor: tail values of shape (deg, m), zero inside the interval.
        """
        p = self._deg
        mu = self._mu[side]
        h = self._h[side]
        boundary = self._interval[side]
        a = (self._tail_a_right if side == 1 else self._tail_a_left).to(dtype=x.dtype, device=x.device)

        # relu keeps the exponent bounded for points inside the interval so no
        # overflow (and no NaN gradient) can leak through the mask
        if side == 1:
            s = torch.relu((x - boundary) / h)
            outside = x > boundary
            dfac = 1.0 / h
        else:
            s = torch.relu((boundary - x) / h)
            outside = x < boundary
            dfac = -1.0 / h

        j_idx = torch.arange(1, p + 1, dtype=x.dtype, device=x.device)
        if derivative:
            # d/dx [a_j e^{-j mu s}] = dfac * (-j mu) a_j e^{-j mu s}
            coef = a * (-mu * j_idx)
        else:
            dfac = 1.0
            coef = a

        # exponents are <= 0 on the tail side, so no overflow is possible
        E = torch.exp(-mu * j_idx.unsqueeze(1) * s.unsqueeze(0))
        return dfac * (coef @ E) * outside

    def eval_sparse(self, x: torch.Tensor, derivative: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sparse (compressed) evaluation of the basis.

        At most `deg + 1` basis functions are nonzero at any point, so instead of
        the dense `(n, ...)` matrix returned by `__call__` this returns only those
        values together with the index of the basis function each one belongs to.
        Both the work and the memory are `O(deg^2)` per point and do **not** grow
        with the number of knots (or basis functions), which is what makes the
        evaluation cheap for fine knot vectors. It is fully differentiable in `x`
        and runs as a handful of elementwise kernels, so it is fast on CPU and GPU
        alike.

        Args:
            x (torch.Tensor): points where the basis is evaluated. Arbitrary
                shape `(...)`.
            derivative (bool, optional): if True, evaluate the first derivative
                of the basis functions. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - values: shape `(deg+1, ...)`, the nonzero basis values.
                - indices: shape `(deg+1, ...)`, integer indices into `[0, n)`
                  of the corresponding basis functions.

            Slots that carry no basis function (points outside the domain, or
            functions dropped by a "zero"/"decay" boundary) have value exactly 0
            and an in-range but otherwise meaningless index, so the pair can be
            gathered or scattered without any further masking.

        Example:
            >>> basis = BSplineBasis(torch.linspace(0, 1, 1000), deg=3)
            >>> x = torch.rand(10000, requires_grad=True)
            >>> v, i = basis.eval_sparse(x)     # (4, 10000) instead of (1001, 10000)
            >>> f = (c[i] * v).sum(0)           # == c @ basis(x), for coefficients c
            >>> f.sum().backward()              # gradients flow through
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self._knots.dtype)

        p = self._deg
        a, b = self._interval
        shape = x.shape
        xw = x.reshape(-1).to(torch.promote_types(x.dtype, self._knots.dtype))

        # Polynomial part: clamping keeps the (masked out) values of far away
        # points finite, so no inf * 0 = NaN can leak into the gradient
        k = self._spans(xw)
        vals = self._local_values(xw.clamp(a, b), k, derivative)
        vals = vals * ((xw >= a) & (xw <= b)).unsqueeze(0)

        # On "decay" sides the crossing functions continue past the boundary as
        # exponential tails. Points beyond the boundary have their span clamped
        # to the boundary span, so the tails land on the right slots: the deg
        # crossing functions are the last (right) / first (left) deg of them.
        if self._bc[1] == "decay":
            tail = self._eval_tail(xw, side=1, derivative=derivative)
            vals = vals + torch.nn.functional.pad(tail, (0, 0, 1, 0))
        if self._bc[0] == "decay":
            tail = self._eval_tail(xw, side=0, derivative=derivative)
            vals = vals + torch.nn.functional.pad(tail, (0, 0, 0, 1))

        idx = k.unsqueeze(0) + self._idx_offsets.unsqueeze(1) - self._lo

        # Boundary modes drop basis functions at the ends: blank out the slots
        # that would refer to them and park their index on a valid entry
        if self._lo != 0 or self._hi != self._n_splines:
            vals = vals * ((idx >= 0) & (idx < self._n))
            idx = idx.clamp(0, self._n - 1)

        return vals.to(x.dtype).view(p + 1, *shape), idx.view(p + 1, *shape)

    def __call__(self, x: torch.Tensor, derivative: bool = False) -> torch.Tensor:
        """
        Evaluate the B-spline basis functions at the given points.

        Each basis function is evaluated at every point in `x`, and the results
        are stacked along a new first dimension. This method is fully differentiable 
        via PyTorch autograd.

        Only `deg + 1` basis functions are nonzero at any point: they are computed
        by `eval_sparse` in `O(deg^2)` per point, independently of the number of
        knots, and scattered into the dense output. Use `eval_sparse` directly to
        skip the dense `(n, ...)` result altogether.

        Args:
            x (torch.Tensor): points where the basis is evaluated. 
                Arbitrary shape `(...)`.
            derivative (bool, optional): if True, evaluate the first derivative 
                of the basis functions. Defaults to False.

        Returns:
            torch.Tensor: the B-splines evaluated at x. Shape `(n, ...)` where n 
                is the number of basis functions and `...` is the shape of the input.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self._knots.dtype)

        vals, idx = self.eval_sparse(x, derivative)

        m = x.numel()
        dense = torch.zeros(self._n, m, dtype=vals.dtype, device=vals.device)
        dense.scatter_add_(0, idx.reshape(self._deg + 1, m), vals.reshape(self._deg + 1, m))
        return dense.view(self._n, *x.shape)

    def __repr__(self) -> str:
        """
        Return a string representation of the B-spline basis.
        
        Returns:
            str: string representation.
        """
        s = f"BSplineBasis(n={self._n}, deg={self._deg}, interval={self._interval}"
        if self._bc != ("clamped", "clamped"):
            s += f", bc={self._bc}"
        if "decay" in self._bc:
            s += f", decay_rate={self._decay_rates}"
        return s + ")"
    
    def interpolating_points(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the interpolating points (Greville abscissae) and the basis matrix.
        
        The interpolating points are the Greville abscissae, defined as the 
        averages of `deg` consecutive knots. When the basis is evaluated at 
        these points, the resulting matrix is guaranteed to be invertible 
        (non-singular).
        
        This property is fundamental for B-spline interpolation: given function 
        values f_i at the interpolating points x_i, the coefficients c_j of the 
        B-spline expansion can be uniquely determined by solving:
        
            B(x_i) @ c = f
        
        where B(x_i) is the invertible matrix returned by this method.

        For "zero" sides the boundary point is dropped together with the omitted
        basis function. For "decay" sides the dropped out-of-domain crossing
        functions take their (past-boundary) Greville points with them, so all
        retained interpolating points lie within [a, b], the outermost sitting
        exactly at the boundary.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - points: the Greville abscissae as a vector of shape `(n,)`
                - matrix: the basis evaluated at these points, shape `(n, n)`,
                  which is guaranteed to be invertible
        """
        # Greville abscissae of the retained basis functions: the sliding mean
        # of deg consecutive knots, as one windowed reduction (no Python loop).
        # For deg = 0 the basis functions are span indicators, whose natural
        # interpolating points are the span midpoints.
        if self._deg == 0:
            pts = (0.5 * (self._knots[:-1] + self._knots[1:]))[self._lo:self._hi]
        else:
            pts = (self._knots[1:].unfold(0, self._deg, 1).sum(1) / self._deg)[self._lo:self._hi]
        
        # Evaluate basis at these points (detached for stability)
        with torch.no_grad():
            matrix = self(pts).t()
        
        return pts, matrix
    
    def integration_weights(self) -> torch.Tensor:
        """
        Return the integrals of each individual B-spline basis function.

        For B-splines supported inside the interval, the integral of basis
        function B_{i,p}(x) over its entire support is given by the analytical
        formula:

            ∫ B_{i,p}(x) dx = (t_{i+p+1} - t_i) / (p+1)

        where t_i are the knots in the extended knot vector and p is the degree.
        For "decay" sides, the bounded part is integrated exactly with
        Gauss-Legendre quadrature and the analytic integral of the exponential
        tails is added, so the weights are the integrals over the whole
        (unbounded) domain.

        Returns:
            torch.Tensor: a vector of shape `(n,)` containing the integral of
                each B-spline basis function.
        """
        device = self._knots.device
        dtype = self._knots.dtype
        if "decay" not in self._bc:
            # Analytical formula for B-spline integrals of the retained functions:
            # int B_{i,p}(x) dx = (t_{i+p+1} - t_i) / (p + 1)
            t = self._knots
            return (t[self._lo + self._deg + 1:self._hi + self._deg + 1]
                    - t[self._lo:self._hi]) / (self._deg + 1)

        # Bounded part (exact: Gauss-Legendre with deg+1 points per knot span),
        # accumulated from the sparse evaluation so the cost stays linear in n
        pts, w = self._quadrature_bounded(self._deg + 1)
        v, i = self.eval_sparse(pts)
        integrals = torch.zeros(self._n, dtype=dtype, device=device)
        integrals.scatter_add_(0, i.reshape(-1), (v * w).reshape(-1))

        # Analytic tail mass: ∫_0^∞ e^{-j mu s} h ds = h / (j mu)
        for side in (0, 1):
            if self._bc[side] != "decay":
                continue
            mu = self._mu[side]
            a = self._tail_a_right if side == 1 else self._tail_a_left
            moments = torch.tensor([1.0 / ((j + 1) * mu) for j in range(self._deg)],
                                   dtype=dtype, device=device)
            tail_mass = self._h[side] * (a @ moments)
            # Only the retained crossing functions receive tail mass. The a rows
            # are ordered from the innermost crossing function outwards, so the
            # retained ones are the innermost `keep` (side 1) / outermost-dropped
            # `deg - keep` excluded (side 0).
            keep = self._tail_keep[side]
            if keep == 0:
                continue
            if side == 1:
                integrals[-keep:] = integrals[-keep:] + tail_mass[:keep]
            else:
                integrals[:keep] = integrals[:keep] + tail_mass[self._deg - keep:]

        return integrals

    def _quadrature_bounded(self, degree: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gauss-Legendre quadrature on the bounded knot interval [a, b].

        Args:
            degree (int): the number of quadrature points per knot span.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - points: 1D tensor of quadrature points.
                - weights: 1D tensor of corresponding quadrature weights.
        """
        # Gauss-Legendre points and weights on [-1, 1]
        x_ref, w_ref = np.polynomial.legendre.leggauss(degree)
        x_ref = torch.tensor(x_ref, dtype=self._knots.dtype, device=self._knots.device)
        w_ref = torch.tensor(w_ref, dtype=self._knots.dtype, device=self._knots.device)

        # Unique interior knots to define non-zero spans (phantom knots of
        # "decay" sides are excluded; tails are handled separately)
        unique_knots = torch.unique(self._interior_knots)

        # Map the reference rule onto every span at once (span-major ordering)
        a = unique_knots[:-1].unsqueeze(1)
        b = unique_knots[1:].unsqueeze(1)
        half = 0.5 * (b - a)

        points = (half * x_ref + 0.5 * (a + b)).reshape(-1)
        weights = (half * w_ref).reshape(-1)

        return points, weights

    def quadrature(self, degree: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute quadrature points and weights for integrating functions in the B-spline basis.

        It computes Gauss-Legendre quadrature points and weights for each non-zero
        length interval between the knots. For "decay" sides, points mapped through
        the tail variable t = 1 - exp(-rate * |x - boundary|) are appended on the
        unbounded tail. In t the tails are polynomials vanishing at t = 1, so with
        `degree` >= deg the rule is exact both for the basis tails themselves and
        for their pairwise products (e.g. mass matrix entries).

        Args:
            degree (int): the number of quadrature points per interval (and per tail).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - points: 1D tensor of quadrature points (ascending).
                - weights: 1D tensor of corresponding quadrature weights.
        """
        points, weights = self._quadrature_bounded(degree)
        if "decay" not in self._bc:
            return points, weights

        # Substituting t = 1 - e^{-λ|x-b|} maps the tail onto the unit interval:
        #   ∫_b^∞ f dx = (1/λ) ∫_0^1 f(b - ln(e)/λ) de/e,   e = 1 - t = e^{-λ(x-b)}
        # The tails are polynomials in t vanishing at t = 1, hence polynomials in e
        # with no constant term, so f/e is a polynomial and Gauss-Legendre is exact.
        u_np, w_np = np.polynomial.legendre.leggauss(degree)
        eps = torch.tensor(0.5 * (u_np + 1.0), dtype=self._knots.dtype, device=self._knots.device)
        w_eps = torch.tensor(0.5 * w_np, dtype=self._knots.dtype, device=self._knots.device)
        offset = -torch.log(eps)

        if self._bc[1] == "decay":
            lam = self._decay_rates[1]
            # e ascending -> x descending, so flip to keep the points ascending
            points = torch.cat([points, torch.flip(self._interval[1] + offset / lam, [0])])
            weights = torch.cat([weights, torch.flip(w_eps / (lam * eps), [0])])
        if self._bc[0] == "decay":
            lam = self._decay_rates[0]
            points = torch.cat([self._interval[0] - offset / lam, points])
            weights = torch.cat([w_eps / (lam * eps), weights])

        return points, weights

    def _tail_matrix_blocks(self, side: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Analytic tail contributions of a "decay" side to the mass, stiffness and
        advection matrices.

        Only the `deg` crossing basis functions are nonzero beyond the boundary,
        so each matrix receives a (deg, deg) block. With tails
        T_i(s) = sum_j a_ij e^{-j mu s} the single moment ∫_0^∞ e^{-(j+l) mu s} ds
        = 1 / ((j+l) mu) turns every block into a closed-form bilinear form in the
        tail coefficients.

        Args:
            side (int): 0 for the left boundary, 1 for the right.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (deg, deg) blocks
                of the mass, stiffness and advection matrices.
        """
        p = self._deg
        mu = self._mu[side]
        h = self._h[side]
        a = self._tail_a_right if side == 1 else self._tail_a_left

        j_idx = torch.arange(1, p + 1, dtype=a.dtype, device=a.device)
        # S_{jl} = j + l, the exponent index of the product of two tail terms
        S = j_idx.unsqueeze(1) + j_idx.unsqueeze(0)

        # dx = h ds on both sides; d/dx = ±(1/h) d/ds
        dfac = 1.0 / h if side == 1 else -1.0 / h
        # mass:      h * ∫ e^{-(j+l) mu s} ds           = h / ((j+l) mu)
        # stiffness: dfac^2 h mu^2 j l * 1/((j+l) mu)   = dfac^2 h mu * j l / (j+l)
        # advection: dfac h mu (-l) * 1/((j+l) mu)      = dfac h * (-l) / (j+l)
        mass = h * (a @ (1.0 / (S * mu)) @ a.t())
        stiffness = (dfac * dfac * h * mu) * (a @ ((j_idx.unsqueeze(1) * j_idx.unsqueeze(0)) / S) @ a.t())
        advection = (dfac * h) * (a @ (-j_idx.unsqueeze(0) / S) @ a.t())
        return mass, stiffness, advection

    def _add_tail_blocks(self, matrix: torch.Tensor, which: int) -> torch.Tensor:
        """
        Add the analytic tail blocks of all "decay" sides to a matrix computed
        on the bounded interval.

        Args:
            matrix (torch.Tensor): matrix of shape (n, n) over the bounded part.
            which (int): 0 for mass, 1 for stiffness, 2 for advection.

        Returns:
            torch.Tensor: the corrected matrix.
        """
        for side in (0, 1):
            if self._bc[side] != "decay":
                continue
            keep = self._tail_keep[side]
            if keep == 0:
                continue
            block = self._tail_matrix_blocks(side)[which].to(matrix.dtype)
            # Restrict the (deg, deg) tail block to the retained crossing
            # functions (see integration_weights for the row ordering).
            if side == 1:
                matrix[-keep:, -keep:] = matrix[-keep:, -keep:] + block[:keep, :keep]
            else:
                matrix[:keep, :keep] = matrix[:keep, :keep] + block[self._deg - keep:, self._deg - keep:]
        return matrix

    def _gram_matrix(self, degree: int, d_row: bool, d_col: bool) -> torch.Tensor:
        """
        Assemble int D^a B_i(x) D^b B_j(x) dx over the bounded interval [a, b].

        The matrix is banded: at a quadrature point only deg+1 basis functions
        are nonzero, so the sparse evaluation gives the (deg+1) x (deg+1) block
        each point contributes and the blocks are scattered into the result. The
        cost is O(deg^2) per quadrature point instead of the O(n^2) of a dense
        `B @ diag(w) @ B.T` product, so assembly scales linearly in the number
        of basis functions.

        Args:
            degree (int): the number of quadrature points per knot span.
            d_row (bool): differentiate the row (test) functions.
            d_col (bool): differentiate the column (trial) functions.

        Returns:
            torch.Tensor: matrix of shape (n, n) over the bounded part.
        """
        pts, w = self._quadrature_bounded(degree)
        vr, ir = self.eval_sparse(pts, derivative=d_row)
        vc, ic = (vr, ir) if d_row == d_col else self.eval_sparse(pts, derivative=d_col)

        # Outer product of the two local blocks at every quadrature point,
        # accumulated into the flattened (n, n) matrix
        blocks = (vr * w).unsqueeze(1) * vc.unsqueeze(0)
        flat_idx = ir.unsqueeze(1) * self._n + ic.unsqueeze(0)
        out = torch.zeros(self._n * self._n, dtype=blocks.dtype, device=blocks.device)
        return out.scatter_add_(0, flat_idx.reshape(-1), blocks.reshape(-1)).view(self._n, self._n)

    def mass_matrix(self, degree: int = None) -> torch.Tensor:
        """
        Compute the mass matrix M_ij = int B_i(x) B_j(x) dx.

        For "decay" sides the integral extends over the unbounded tail; the tail
        contribution is added analytically.

        Args:
            degree (int, optional): The number of quadrature points per interval.
                If not provided, defaults to deg + 1 to ensure exact integration
                of polynomials of degree up to 2*deg.

        Returns:
            torch.Tensor: The mass matrix of shape (n, n).
        """
        if degree is None:
            # Product of two B-splines of degree p has degree 2p.
            # Gauss-Legendre with k points integrates polynomials of degree up to 2k-1 exactly.
            # So 2k - 1 >= 2p => k >= p + 1.
            degree = self._deg + 1

        # M_ij = sum_k B_i(x_k) B_j(x_k) w_k, assembled band by band
        M = self._gram_matrix(degree, d_row=False, d_col=False)

        return self._add_tail_blocks(M, 0)

    def stiffness_matrix(self, degree: int = None) -> torch.Tensor:
        """
        Compute the stiffness matrix S_ij = int B'_i(x) B'_j(x) dx.

        For "decay" sides the integral extends over the unbounded tail; the tail
        contribution is added analytically.

        Args:
            degree (int, optional): The number of quadrature points per interval.
                If not provided, defaults to deg + 1.

        Returns:
            torch.Tensor: The stiffness matrix of shape (n, n).
        """
        if degree is None:
            degree = self._deg + 1

        # S_ij = sum_k B'_i(x_k) B'_j(x_k) w_k, assembled band by band
        S = self._gram_matrix(degree, d_row=True, d_col=True)

        return self._add_tail_blocks(S, 1)

    def advection_matrix(self, degree: int = None) -> torch.Tensor:
        """
        Compute the advection matrix C_ij = int B_i(x) B'_j(x) dx.

        For "decay" sides the integral extends over the unbounded tail; the tail
        contribution is added analytically.

        Args:
            degree (int, optional): The number of quadrature points per interval.
                If not provided, defaults to deg + 1.

        Returns:
            torch.Tensor: The advection matrix of shape (n, n).
        """
        if degree is None:
            degree = self._deg + 1

        # C_ij = sum_k B_i(x_k) B'_j(x_k) w_k, assembled band by band
        C = self._gram_matrix(degree, d_row=False, d_col=True)

        return self._add_tail_blocks(C, 2)


class GaussianBasis(BaseBasis):
    """
    Gaussian basis functions.
    
    The basis functions are Gaussians centered at the knots.
    phi_i(x) = exp(-(x - mu_i)^2 / (2 * sigma_i^2))
    
    The centers mu_i are the knots themselves.
    The widths sigma_i are determined by the 'delta_overlap' parameter,
    which specifies the distance to the neighboring knot that defines the scale.
    Specifically, sigma_i is set to the distance between the i-th knot and the 
    (i + delta_overlap)-th knot. For indices where this neighbor doesn't exist 
    (near the end), the distance to the (i - delta_overlap)-th knot is used.
    """
    
    def __init__(self, knots: torch.Tensor, delta_overlap: int):
        """
        Initialize a Gaussian basis.
        
        Args:
            knots (torch.Tensor): the centers of the Gaussian functions.
            delta_overlap (int): the knot stride used to determine the width (sigma)
                of the Gaussians. Must be >= 1.
        """
        super().__init__()
        if not isinstance(knots, torch.Tensor):
            knots = torch.tensor(knots, dtype=torch.float64)
        
        knots = knots.flatten().to(torch.float64)
        if knots.numel() == 0:
             raise ValueError("Knots must not be empty.")
             
        self._knots = knots
        self._n = knots.size(0)
        self._delta_overlap = int(delta_overlap)
        
        if self._delta_overlap < 1:
            raise ValueError("delta_overlap must be a positive integer.")
            
        # Compute sigmas
        sigmas = torch.zeros_like(knots)
        k = self._delta_overlap
        
        for i in range(self._n):
            if i + k < self._n:
                # Use forward distance
                sigmas[i] = torch.abs(knots[i + k] - knots[i])
            elif i - k >= 0:
                # Use backward distance
                sigmas[i] = torch.abs(knots[i] - knots[i - k])
            else:
                # Fallback for very small number of knots (n <= k)
                if self._n > 1:
                     sigmas[i] = torch.abs(knots[-1] - knots[0])
                else:
                     sigmas[i] = torch.tensor(1.0, dtype=knots.dtype, device=knots.device)
        
        # Avoid zero sigma
        sigmas = torch.maximum(sigmas, torch.tensor(1e-10, dtype=sigmas.dtype, device=sigmas.device))
        
        self.register_buffer('_centers', knots)
        self.register_buffer('_sigmas', sigmas)
        
    @property
    def n(self) -> int:
        return self._n

    def __call__(self, x: torch.Tensor, derivative: bool = False) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self._centers.dtype, device=self._centers.device)
        
        original_shape = x.shape
        x_flat = x.flatten()
        
        # Expand dims for broadcasting: (n, m)
        x_expanded = x_flat.unsqueeze(0)
        centers_expanded = self._centers.unsqueeze(1)
        sigmas_expanded = self._sigmas.unsqueeze(1)
        
        diff = x_expanded - centers_expanded
        exponent = -(diff ** 2) / (2 * sigmas_expanded ** 2)
        values = torch.exp(exponent)
        
        if derivative:
            # d/dx = values * (-2 * (x - mu) / (2 * sigma^2)) = values * (-(x - mu) / sigma^2)
            derivs = values * (-diff / (sigmas_expanded ** 2))
            return derivs.view(self._n, *original_shape)
        else:
            return values.view(self._n, *original_shape)

    def __repr__(self) -> str:
        return f"GaussianBasis(n={self._n}, delta_overlap={self._delta_overlap})"
        
    def interpolating_points(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Use centers as interpolating points
        pts = self._centers
        with torch.no_grad():
             matrix = self(pts)
        return pts, matrix

    def integration_weights(self) -> torch.Tensor:
         """
         Computes the integral of the Gaussian basis functions over the entire real line (-inf, inf).
         Integral = sigma * sqrt(2 * pi)
         """
         sigma = self._sigmas
         return sigma * math.sqrt(2.0 * math.pi)
