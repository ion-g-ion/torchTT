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
          it by tails of the form q(x) * exp(-rate * |x - boundary|), where q is a
          polynomial of degree deg-1 matched in value and the first deg-1
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
        >>> # Density-friendly variant: vanishes at 0, unbounded to the right
        >>> basis_pdf = BSplineBasis(knots, deg=3, bc=("zero", "decay"))

    Note:
        For ``"decay"`` sides the exponential tails are matched to the spline jet
        at the boundary. With the default rate `deg / h` the tails are essentially
        non-negative (for deg >= 3 a negative undershoot below 1% of the basis
        peak remains). Rates much smaller than `deg / h` make the matched
        polynomial factor dominate over a long range and produce large negative
        undershoots in the tails; prefer increasing the spacing of the boundary
        knots over lowering the rate if heavier tails are needed.
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
                rate(s) for "decay" sides; the tails behave like
                exp(-decay_rate * |x - boundary|). A None entry uses the default
                `deg / h`, where h is the adjacent knot spacing. Defaults to None.
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

        # Polynomial factors (in the scaled tail variable) of the smooth
        # exponential tails for each "decay" side
        for side, name in enumerate(('_tail_q_left', '_tail_q_right')):
            if bc[side] == "decay":
                q = self._compute_tail_coefficients(side)
            else:
                q = torch.zeros(0, 0, dtype=torch.float64)
            self.register_buffer(name, q)
    
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
    
    def _eval_spline_raw(self, x: torch.Tensor, derivative: bool = False) -> torch.Tensor:
        """
        Evaluate all B-splines of the extended knot vector via Cox-de Boor recursion.

        No boundary modes are applied here: no basis functions are dropped, and
        for "decay" sides the values on the phantom spans are the polynomial
        continuations, not the exponential tails.

        For derivatives, the last recursion level applies the formula:
        B'_{i,p}(x) = p * [B_{i,p-1}(x)/(t_{i+p}-t_i) - B_{i+1,p-1}(x)/(t_{i+p+1}-t_{i+1})]

        Args:
            x (torch.Tensor): flattened points of shape (m,)
            derivative (bool, optional): if True, evaluate the first derivative.

        Returns:
            torch.Tensor: values of shape (n_splines, m).
        """
        m = x.shape[0]
        num_intervals = self._knots.shape[0] - 1

        # Degree 0: derivative is zero everywhere (piecewise constant)
        if derivative and self._deg == 0:
            return torch.zeros(self._n_splines, m, dtype=x.dtype, device=x.device)

        # For a clamped/zero right boundary the last basis function must include
        # the right endpoint; for "decay" the boundary knot is interior-like and
        # x = b is seeded in the first phantom span instead.
        inclusive_last = self._bc[1] != "decay"

        # Initialize degree-0 basis functions (piecewise constant)
        # result[i, j] = 1 if knots[i] <= x[j] < knots[i+1], else 0
        result = torch.zeros(num_intervals, m, dtype=x.dtype, device=x.device)

        for i in range(num_intervals):
            if inclusive_last and i == self._n_splines - 1:
                mask = (x >= self._knots[i]) & (x <= self._knots[i + 1])
            else:
                mask = (x >= self._knots[i]) & (x < self._knots[i + 1])
            result[i] = torch.where(mask, torch.ones_like(x), torch.zeros_like(x))

        # Cox-de Boor recursion
        for d in range(self._deg):
            new_result = torch.zeros_like(result)
            for i in range(num_intervals - d - 1):
                denom1 = self._knots[i + d + 1] - self._knots[i]
                denom2 = self._knots[i + d + 2] - self._knots[i + 1]

                if derivative and d == self._deg - 1:
                    # At the last recursion level, apply the derivative formula
                    if denom1 != 0:
                        a = self._deg * result[i] / denom1
                    else:
                        a = torch.zeros_like(x)

                    if denom2 != 0:
                        b = -self._deg * result[i + 1] / denom2
                    else:
                        b = torch.zeros_like(x)
                else:
                    # First term: B_{i,d}(x) * (x - t_i) / (t_{i+d+1} - t_i)
                    if denom1 != 0:
                        a = result[i] * (x - self._knots[i]) / denom1
                    else:
                        a = torch.zeros_like(x)

                    # Second term: B_{i+1,d}(x) * (t_{i+d+2} - x) / (t_{i+d+2} - t_{i+1})
                    if denom2 != 0:
                        b = result[i + 1] * (self._knots[i + d + 2] - x) / denom2
                    else:
                        b = torch.zeros_like(x)

                new_result[i] = a + b

            result = new_result

        return result[:self._n_splines]

    def _compute_tail_coefficients(self, side: int) -> torch.Tensor:
        """
        Compute the polynomial factors of the smooth exponential tails.

        For a "decay" side, the `deg` basis functions whose support crosses the
        boundary are continued beyond it by q(s) * exp(-mu * s), where
        s = |x - boundary| / h is the scaled distance to the boundary. The
        polynomial q (degree deg-1) is chosen so that the value and the first
        deg-1 derivatives match the spline at the boundary, which is equivalent
        to q(s) = (boundary spline piece in s) * exp(+mu * s) truncated at
        degree deg-1.

        Args:
            side (int): 0 for the left boundary, 1 for the right.

        Returns:
            torch.Tensor: tail coefficients of shape (deg, deg); row j holds the
                monomial coefficients of q for the j-th crossing basis function.
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

        # Truncated product with the exponential series exp(+mu*s)
        U = torch.zeros(p, p, dtype=torch.float64)
        for r in range(p):
            for k in range(r, p):
                U[r, k] = mu ** (k - r) / math.factorial(k - r)
        return piece[:, :p] @ U

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
        q = (self._tail_q_right if side == 1 else self._tail_q_left).to(dtype=x.dtype, device=x.device)

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

        if derivative:
            # d/dx [q(s) e^{-mu s}] = dfac * (q'(s) - mu q(s)) e^{-mu s}
            coef = -mu * q
            if p > 1:
                coef[:, :-1] += q[:, 1:] * torch.arange(1, p, dtype=q.dtype, device=q.device)
        else:
            dfac = 1.0
            coef = q

        powers = s.unsqueeze(0) ** torch.arange(p, dtype=x.dtype, device=x.device).unsqueeze(1)
        return dfac * (coef @ powers) * torch.exp(-mu * s) * outside

    def _eval_basis_modes(self, x: torch.Tensor, original_shape: tuple, derivative: bool) -> torch.Tensor:
        """
        Evaluate the basis (or its derivative) with the boundary modes applied.

        Args:
            x (torch.Tensor): flattened points of shape (m,)
            original_shape (tuple): original shape of input for reshaping output
            derivative (bool): if True, evaluate the first derivative.

        Returns:
            torch.Tensor: basis values of shape (n, *original_shape)
        """
        result = self._eval_spline_raw(x, derivative=derivative)

        # On "decay" sides discard the polynomial continuation on the phantom
        # spans and add the smooth exponential tails instead
        if self._bc[1] == "decay":
            result = result * (x <= self._interval[1])
        if self._bc[0] == "decay":
            result = result * (x >= self._interval[0])
        if self._bc[1] == "decay":
            tail = self._eval_tail(x, side=1, derivative=derivative)
            result = result + torch.cat([result.new_zeros(self._n_splines - self._deg, x.shape[0]), tail])
        if self._bc[0] == "decay":
            tail = self._eval_tail(x, side=0, derivative=derivative)
            result = result + torch.cat([tail, result.new_zeros(self._n_splines - self._deg, x.shape[0])])

        # "zero" sides drop the boundary basis function
        return result[self._lo:self._hi].view(self._n, *original_shape)
    
    def __call__(self, x: torch.Tensor, derivative: bool = False) -> torch.Tensor:
        """
        Evaluate the B-spline basis functions at the given points.
        
        Each basis function is evaluated at every point in `x`, and the results
        are stacked along a new first dimension. This method is fully differentiable 
        via PyTorch autograd.
        
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
            x = torch.tensor(x, dtype=torch.float64)

        # Store original shape for output reshaping
        original_shape = x.shape
        x_flat = x.flatten()

        return self._eval_basis_modes(x_flat, original_shape, derivative)
    
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
        # Compute Greville abscissae of the retained basis functions
        pts = torch.stack([
            self._knots[i + 1:i + self._deg + 1].sum() / self._deg
            for i in range(self._lo, self._hi)
        ])
        
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
        if "decay" not in self._bc:
            # Analytical formula for B-spline integrals of the retained functions
            integrals = torch.zeros(self._n, dtype=torch.float64, device=device)
            for j, i in enumerate(range(self._lo, self._hi)):
                # Integral of B_{i,p}(x) = (t_{i+p+1} - t_i) / (p+1)
                knot_diff = self._knots[i + self._deg + 1] - self._knots[i]
                integrals[j] = knot_diff / (self._deg + 1)
            return integrals

        # Bounded part (exact: Gauss-Legendre with deg+1 points per knot span)
        pts, w = self._quadrature_bounded(self._deg + 1)
        integrals = self(pts) @ w

        # Analytic tail mass: ∫_0^∞ s^j e^{-mu s} h ds = h * j! / mu^{j+1}
        for side in (0, 1):
            if self._bc[side] != "decay":
                continue
            mu = self._mu[side]
            q = self._tail_q_right if side == 1 else self._tail_q_left
            moments = torch.tensor([math.factorial(j) / mu ** (j + 1) for j in range(self._deg)],
                                   dtype=torch.float64, device=device)
            tail_mass = self._h[side] * (q @ moments)
            # Only the retained crossing functions receive tail mass. The q rows
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

        points = []
        weights = []

        for i in range(len(unique_knots) - 1):
            a = unique_knots[i]
            b = unique_knots[i+1]

            # Map points and weights to [a, b]
            x_scaled = 0.5 * (b - a) * x_ref + 0.5 * (a + b)
            w_scaled = 0.5 * (b - a) * w_ref

            points.append(x_scaled)
            weights.append(w_scaled)

        if len(points) == 0:
             return torch.empty(0, dtype=self._knots.dtype, device=self._knots.device), \
                    torch.empty(0, dtype=self._knots.dtype, device=self._knots.device)

        return torch.cat(points), torch.cat(weights)

    def quadrature(self, degree: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute quadrature points and weights for integrating functions in the B-spline basis.

        It computes Gauss-Legendre quadrature points and weights for each non-zero
        length interval between the knots. For "decay" sides, mapped Gauss-Laguerre
        points are appended on the unbounded tail; the resulting rule is exact for
        the basis tails themselves and rapidly convergent for products of tails
        (e.g. mass matrix entries).

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

        # Gauss-Laguerre rule mapped to the tail: ∫_b^∞ f dx ≈ Σ_j (w_j e^{u_j} / λ) f(b + u_j/λ)
        u_np, w_np = np.polynomial.laguerre.laggauss(degree)
        u = torch.tensor(u_np, dtype=self._knots.dtype, device=self._knots.device)
        total = torch.tensor(w_np * np.exp(u_np), dtype=self._knots.dtype, device=self._knots.device)

        if self._bc[1] == "decay":
            lam = self._decay_rates[1]
            points = torch.cat([points, self._interval[1] + u / lam])
            weights = torch.cat([weights, total / lam])
        if self._bc[0] == "decay":
            lam = self._decay_rates[0]
            points = torch.cat([torch.flip(self._interval[0] - u / lam, [0]), points])
            weights = torch.cat([torch.flip(total / lam, [0]), weights])

        return points, weights

    def _tail_matrix_blocks(self, side: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Analytic tail contributions of a "decay" side to the mass, stiffness and
        advection matrices.

        Only the `deg` crossing basis functions are nonzero beyond the boundary,
        so each matrix receives a (deg, deg) block. With T_i = q_i(s) e^{-mu s}
        and the moments ∫_0^∞ s^m e^{-2 mu s} ds = m! / (2 mu)^{m+1}, the blocks
        are bilinear forms in the tail polynomial coefficients.

        Args:
            side (int): 0 for the left boundary, 1 for the right.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (deg, deg) blocks
                of the mass, stiffness and advection matrices.
        """
        p = self._deg
        mu = self._mu[side]
        h = self._h[side]
        q = self._tail_q_right if side == 1 else self._tail_q_left

        # Coefficients of the polynomial factor of the tail derivative: q'(s) - mu q(s)
        r = -mu * q
        if p > 1:
            r[:, :-1] += q[:, 1:] * torch.arange(1, p, dtype=q.dtype, device=q.device)

        # Hankel moment matrix H_kl = (k+l)! / (2 mu)^{k+l+1}
        H = torch.tensor([[math.factorial(k + l) / (2 * mu) ** (k + l + 1) for l in range(p)]
                          for k in range(p)], dtype=q.dtype, device=q.device)

        # dx = h ds on both sides; d/dx = ±(1/h) d/ds
        dfac = 1.0 / h if side == 1 else -1.0 / h
        mass = h * (q @ H @ q.t())
        stiffness = (h * dfac * dfac) * (r @ H @ r.t())
        advection = (h * dfac) * (q @ H @ r.t())
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

        pts, w = self._quadrature_bounded(degree)

        # Evaluate basis at quadrature points
        B = self(pts)  # Shape: (n, num_pts)

        # M_ij = sum_k B_i(x_k) B_j(x_k) w_k
        # Equivalent to B @ diag(w) @ B.T
        M = B @ (w.unsqueeze(1) * B.t())

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

        pts, w = self._quadrature_bounded(degree)

        # Evaluate basis derivatives at quadrature points
        B_prime = self(pts, derivative=True)  # Shape: (n, num_pts)

        # S_ij = sum_k B'_i(x_k) B'_j(x_k) w_k
        S = B_prime @ (w.unsqueeze(1) * B_prime.t())

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

        pts, w = self._quadrature_bounded(degree)

        # Evaluate basis and derivatives at quadrature points
        B = self(pts)  # Shape: (n, num_pts)
        B_prime = self(pts, derivative=True)  # Shape: (n, num_pts)

        # C_ij = sum_k B_i(x_k) B'_j(x_k) w_k
        # Equivalent to B @ diag(w) @ B_prime.T
        C = B @ (w.unsqueeze(1) * B_prime.t())

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
