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
    
    The number of basis functions is `n = len(knots) + deg - 1`.
    
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
    """
    
    def __init__(self, knots: torch.Tensor, deg: int):
        """
        Initialize a B-spline basis.
        
        Args:
            knots (torch.Tensor): the interior knot vector. The boundary 
                knots are automatically repeated `deg` times.
            deg (int): the polynomial degree of the B-splines (e.g., 3 for cubic).
        """
        super().__init__()
        if not isinstance(knots, torch.Tensor):
            knots = torch.tensor(knots, dtype=torch.float64)
        
        knots = knots.flatten().to(torch.float64)
        
        self._deg = int(deg)
        self._n = knots.size(0) + self._deg - 1
        
        # Extend knots with boundary repetitions (registered as buffer, not parameter)
        _knots = torch.cat([
            knots[0].expand(self._deg),
            knots,
            knots[-1].expand(self._deg)
        ])
        self.register_buffer('_knots', _knots)
        
        self._interval = (float(knots[0].item()), float(knots[-1].item()))
    
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
        The extended knot vector (including boundary repetitions).
        
        Returns:
            torch.Tensor: the knot vector.
        """
        return self._knots.clone()
    
    def _eval_basis(self, x: torch.Tensor, original_shape: tuple) -> torch.Tensor:
        """
        Evaluate B-spline basis using Cox-de Boor recursion.
        
        Args:
            x (torch.Tensor): flattened points of shape (m,)
            original_shape (tuple): original shape of input for reshaping output
            
        Returns:
            torch.Tensor: basis values of shape (n, *original_shape)
        """
        m = x.shape[0]
        num_intervals = self._knots.shape[0] - 1
        
        # Initialize degree-0 basis functions (piecewise constant)
        # result[i, j] = 1 if knots[i] <= x[j] < knots[i+1], else 0
        result = torch.zeros(num_intervals, m, dtype=x.dtype, device=x.device)
        
        for i in range(num_intervals):
            # Special handling for the last basis function to include the right endpoint
            if i == self._n - 1:
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
        
        # Reshape to (n, *original_shape)
        return result[:self._n].view(self._n, *original_shape)
    
    def _eval_basis_derivative(self, x: torch.Tensor, original_shape: tuple) -> torch.Tensor:
        """
        Evaluate the derivative of B-spline basis using Cox-de Boor recursion.
        
        Uses the formula: B'_{i,p}(x) = p * [B_{i,p-1}(x)/(t_{i+p}-t_i) - B_{i+1,p-1}(x)/(t_{i+p+1}-t_{i+1})]
        
        Args:
            x (torch.Tensor): flattened points of shape (m,)
            original_shape (tuple): original shape of input for reshaping output
            
        Returns:
            torch.Tensor: derivative values of shape (n, *original_shape)
        """
        # Degree 0: derivative is zero everywhere (piecewise constant)
        if self._deg == 0:
            return torch.zeros(self._n, *original_shape, dtype=x.dtype, device=x.device)
        
        m = x.shape[0]
        num_intervals = self._knots.shape[0] - 1
        
        # Initialize degree-0 basis functions
        result = torch.zeros(num_intervals, m, dtype=x.dtype, device=x.device)
        
        for i in range(num_intervals):
            if i == self._n - 1:
                mask = (x >= self._knots[i]) & (x <= self._knots[i + 1])
            else:
                mask = (x >= self._knots[i]) & (x < self._knots[i + 1])
            result[i] = torch.where(mask, torch.ones_like(x), torch.zeros_like(x))
        
        # Modified Cox-de Boor recursion for derivatives
        for d in range(self._deg):
            new_result = torch.zeros_like(result)
            for i in range(num_intervals - d - 1):
                denom1 = self._knots[i + d + 1] - self._knots[i]
                denom2 = self._knots[i + d + 2] - self._knots[i + 1]
                
                if d == self._deg - 1:
                    # At the last recursion level, apply derivative formula
                    if denom1 != 0:
                        a = self._deg * result[i] / denom1
                    else:
                        a = torch.zeros_like(x)
                    
                    if denom2 != 0:
                        b = -self._deg * result[i + 1] / denom2
                    else:
                        b = torch.zeros_like(x)
                else:
                    # Standard recursion for lower degrees
                    if denom1 != 0:
                        a = result[i] * (x - self._knots[i]) / denom1
                    else:
                        a = torch.zeros_like(x)
                    
                    if denom2 != 0:
                        b = result[i + 1] * (self._knots[i + d + 2] - x) / denom2
                    else:
                        b = torch.zeros_like(x)
                
                new_result[i] = a + b
            
            result = new_result
        
        # Reshape to (n, *original_shape)
        return result[:self._n].view(self._n, *original_shape)
    
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
        
        if derivative:
            return self._eval_basis_derivative(x_flat, original_shape)
        else:
            return self._eval_basis(x_flat, original_shape)
    
    def __repr__(self) -> str:
        """
        Return a string representation of the B-spline basis.
        
        Returns:
            str: string representation.
        """
        return f"BSplineBasis(n={self._n}, deg={self._deg}, interval={self._interval})"
    
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
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - points: the Greville abscissae as a vector of shape `(n,)`
                - matrix: the basis evaluated at these points, shape `(n, n)`, 
                  which is guaranteed to be invertible
        """
        # Compute Greville abscissae: average of deg consecutive knots
        pts = torch.stack([
            self._knots[i + 1:i + self._deg + 1].sum() / self._deg 
            for i in range(self._n)
        ])
        
        # Evaluate basis at these points (detached for stability)
        with torch.no_grad():
            matrix = self(pts).t()
        
        return pts, matrix
    
    def integration_weights(self) -> torch.Tensor:
        """
        Return the integrals of each individual B-spline basis function.
        
        For B-splines, the integral of basis function B_{i,p}(x) over its 
        entire support is given by the analytical formula:
        
            ∫ B_{i,p}(x) dx = (t_{i+p+1} - t_i) / (p+1)
        
        where t_i are the knots in the extended knot vector and p is the degree.
        
        Returns:
            torch.Tensor: a vector of shape `(n,)` containing the integral of 
                each B-spline basis function.
        """
        # Analytical formula for B-spline integrals
        integrals = torch.zeros(self._n, dtype=torch.float64, device=self._knots.device)
        
        for i in range(self._n):
            # Integral of B_{i,p}(x) = (t_{i+p+1} - t_i) / (p+1)
            knot_diff = self._knots[i + self._deg + 1] - self._knots[i]
            integrals[i] = knot_diff / (self._deg + 1)
        
        return integrals


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
