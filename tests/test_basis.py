"""
Test suite for basis functions in torchtt.functional.basis.

Tests cover:
- Generic basis functionality (parametrized for extensibility)
- BSpline-specific properties
- Device compatibility (CPU and CUDA)
- Automatic differentiation
"""
import torch
import pytest
import numpy as np

from torchtt.functional import BSplineBasis


# =============================================================================
# Fixtures and Helpers
# =============================================================================

def rel_err(t, ref):
    """Compute relative error between two tensors."""
    ref_norm = torch.linalg.norm(ref)
    if ref_norm == 0:
        return torch.linalg.norm(t - ref)
    return (torch.linalg.norm(t - ref) / ref_norm).item()


def make_bspline_basis(deg, num_knots=6):
    """Factory function for BSplineBasis."""
    knots = torch.linspace(0, 1, num_knots, dtype=torch.float64)
    return BSplineBasis(knots, deg=deg)


# Parametrize basis factories for extensibility to other basis types
BASIS_FACTORIES = [
    pytest.param(make_bspline_basis, id="BSpline"),
]

DEGREES = [1, 2, 3]

# Device fixture with conditional CUDA
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.fixture(params=DEVICES)
def device(request):
    """Fixture that provides available devices."""
    return request.param


# =============================================================================
# Parametrized Tests (Generic Basis Functionality)
# =============================================================================

@pytest.mark.parametrize("basis_factory", BASIS_FACTORIES)
@pytest.mark.parametrize("deg", DEGREES)
def test_basis_dimension(basis_factory, deg):
    """Verify n property returns correct number of basis functions."""
    basis = basis_factory(deg)
    # For BSpline: n = num_knots + deg - 1 = 6 + deg - 1 = 5 + deg
    expected_n = 5 + deg
    assert basis.n == expected_n, f"Expected n={expected_n}, got n={basis.n}"


@pytest.mark.parametrize("basis_factory", BASIS_FACTORIES)
@pytest.mark.parametrize("deg", DEGREES)
def test_basis_evaluation_shape(basis_factory, deg):
    """Verify calling basis returns shape (n, ...) matching input shape."""
    basis = basis_factory(deg)
    
    # Test 1D input
    x1d = torch.linspace(0, 1, 50, dtype=torch.float64)
    B1d = basis(x1d)
    assert B1d.shape == (basis.n, 50), f"Expected shape ({basis.n}, 50), got {B1d.shape}"
    
    # Test 2D input
    x2d = torch.rand(10, 20, dtype=torch.float64)
    B2d = basis(x2d)
    assert B2d.shape == (basis.n, 10, 20), f"Expected shape ({basis.n}, 10, 20), got {B2d.shape}"
    
    # Test scalar input
    x_scalar = torch.tensor(0.5, dtype=torch.float64)
    B_scalar = basis(x_scalar)
    assert B_scalar.shape == (basis.n,), f"Expected shape ({basis.n},), got {B_scalar.shape}"


@pytest.mark.parametrize("basis_factory", BASIS_FACTORIES)
@pytest.mark.parametrize("deg", DEGREES)
def test_interpolating_points_matrix_invertibility(basis_factory, deg):
    """Check that interpolating points return an invertible matrix."""
    basis = basis_factory(deg)
    pts, matrix = basis.interpolating_points()
    
    # Check shapes
    assert pts.shape == (basis.n,), f"Points shape mismatch: {pts.shape}"
    assert matrix.shape == (basis.n, basis.n), f"Matrix shape mismatch: {matrix.shape}"
    
    # Check invertibility via determinant
    det = torch.linalg.det(matrix)
    assert torch.abs(det) > 1e-10, f"Matrix is singular (det={det.item():.2e})"
    
    # Also check via condition number
    cond = torch.linalg.cond(matrix)
    assert cond < 1e10, f"Matrix is ill-conditioned (cond={cond.item():.2e})"


@pytest.mark.parametrize("basis_factory", BASIS_FACTORIES)
@pytest.mark.parametrize("deg", DEGREES)
def test_polynomial_interpolation(basis_factory, deg):
    """
    Test polynomial interpolation accuracy.
    
    For a polynomial of degree <= basis degree, the interpolation should be exact.
    """
    basis = basis_factory(deg)
    
    # Define polynomial: f(x) = x^2 + 3x + 1 (degree 2)
    # For deg >= 2, this should be exactly representable
    def poly_func(x):
        return x**2 + 3*x + 1
    
    # Get interpolating points and matrix
    pts, matrix = basis.interpolating_points()
    
    # Evaluate polynomial at interpolating points
    f_vals = poly_func(pts)
    
    # Solve for coefficients: matrix.T @ c = f_vals
    # Note: matrix has shape (n, n) where matrix[i, j] = B_i(x_j)
    # We need to solve sum_i c_i * B_i(x_j) = f(x_j) for all j
    # This is matrix.T @ c = f_vals
    coeffs = torch.linalg.solve(matrix.T, f_vals)
    
    # Evaluate approximation at test points
    x_test = torch.linspace(0, 1, 100, dtype=torch.float64)
    B_test = basis(x_test)  # shape: (n, 100)
    
    # Approximation: sum_i c_i * B_i(x) = coeffs @ B_test
    approx = coeffs @ B_test
    exact = poly_func(x_test)
    
    # For degree >= 2, quadratic should be exactly representable
    if deg >= 2:
        assert rel_err(approx, exact) < 1e-10, \
            f"Polynomial interpolation failed: rel_err={rel_err(approx, exact):.2e}"
    else:
        # For degree 1, we only check that it's a reasonable approximation
        assert rel_err(approx, exact) < 0.5, \
            f"Linear approximation too poor: rel_err={rel_err(approx, exact):.2e}"


@pytest.mark.parametrize("basis_factory", BASIS_FACTORIES)
@pytest.mark.parametrize("deg", DEGREES)
def test_batched_evaluation(basis_factory, deg):
    """Test that basis correctly handles batched inputs."""
    basis = basis_factory(deg)
    
    # Test 2D batch: (batch, points)
    batch_size = 8
    num_points = 32
    x_batched = torch.rand(batch_size, num_points, dtype=torch.float64)
    B_batched = basis(x_batched)
    
    assert B_batched.shape == (basis.n, batch_size, num_points), \
        f"Batched shape mismatch: {B_batched.shape}"
    
    # Verify consistency: evaluate each batch element separately
    for i in range(batch_size):
        B_single = basis(x_batched[i])
        assert torch.allclose(B_batched[:, i, :], B_single, atol=1e-14), \
            f"Batched evaluation inconsistent at batch index {i}"
    
    # Test 3D batch: (batch1, batch2, points)
    x_3d = torch.rand(4, 5, 20, dtype=torch.float64)
    B_3d = basis(x_3d)
    assert B_3d.shape == (basis.n, 4, 5, 20), f"3D batch shape mismatch: {B_3d.shape}"


@pytest.mark.parametrize("basis_factory", BASIS_FACTORIES)
@pytest.mark.parametrize("deg", DEGREES)
def test_autograd_forward(basis_factory, deg):
    """Test gradients flow through basis evaluation using torch.autograd.grad."""
    basis = basis_factory(deg)
    
    x = torch.linspace(0.1, 0.9, 50, dtype=torch.float64, requires_grad=True)
    B = basis(x)
    
    # Compute some scalar loss
    loss = B.sum()
    
    # Compute gradient
    grad, = torch.autograd.grad(loss, x)
    
    assert grad.shape == x.shape, f"Gradient shape mismatch: {grad.shape}"
    assert not torch.isnan(grad).any(), "Gradient contains NaN"
    assert not torch.isinf(grad).any(), "Gradient contains Inf"


@pytest.mark.parametrize("basis_factory", BASIS_FACTORIES)
@pytest.mark.parametrize("deg", DEGREES)
def test_integration_weights(basis_factory, deg):
    """Test integration weights sum to interval length and are non-negative."""
    basis = basis_factory(deg)
    
    # Get integration weights
    weights = basis.integration_weights()
    
    # Check shape
    assert weights.shape == (basis.n,), f"Expected shape ({basis.n},), got {weights.shape}"
    
    # Check all weights are non-negative
    assert (weights >= 0).all(), "Integration weights must be non-negative"
    
    # Check total weight equals interval length (partition of unity property)
    total_weight = weights.sum().item()
    interval = basis.interval
    interval_length = interval[1] - interval[0]
    assert abs(total_weight - interval_length) < 1e-10, \
        f"Total weight {total_weight:.10f} != interval length {interval_length:.10f}"


# =============================================================================
# BSpline-Specific Tests
# =============================================================================

@pytest.mark.parametrize("deg", DEGREES)
def test_bspline_knot_extension(deg):
    """Verify boundary knots are repeated deg times."""
    knots = torch.linspace(0, 1, 6, dtype=torch.float64)
    basis = BSplineBasis(knots, deg=deg)
    
    extended_knots = basis.knots
    
    # First deg+1 knots should all equal knots[0]
    assert torch.allclose(extended_knots[:deg+1], knots[0].expand(deg+1)), \
        f"Left boundary knots not properly repeated"
    
    # Last deg+1 knots should all equal knots[-1]
    assert torch.allclose(extended_knots[-deg-1:], knots[-1].expand(deg+1)), \
        f"Right boundary knots not properly repeated"


@pytest.mark.parametrize("deg", DEGREES)
def test_bspline_partition_of_unity(deg):
    """B-splines sum to 1 at any point in the interval (partition of unity)."""
    basis = make_bspline_basis(deg)
    
    # Test at many points including boundaries
    x = torch.linspace(0, 1, 200, dtype=torch.float64)
    B = basis(x)  # shape: (n, 200)
    
    # Sum over all basis functions at each point
    sums = B.sum(dim=0)
    
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-12), \
        f"Partition of unity violated: max deviation = {(sums - 1).abs().max().item():.2e}"


@pytest.mark.parametrize("deg", DEGREES)
def test_bspline_non_negativity(deg):
    """B-splines are non-negative everywhere."""
    basis = make_bspline_basis(deg)
    
    x = torch.linspace(0, 1, 500, dtype=torch.float64)
    B = basis(x)
    
    assert (B >= -1e-14).all(), \
        f"Non-negativity violated: min value = {B.min().item():.2e}"


@pytest.mark.parametrize("deg", [2, 3])  # Need deg >= 1 for meaningful derivative
def test_bspline_derivative_evaluation(deg):
    """Test derivative computation against numerical differentiation."""
    basis = make_bspline_basis(deg, num_knots=8)
    
    # Evaluate at interior points (avoid boundary issues)
    x = torch.linspace(0.1, 0.9, 50, dtype=torch.float64)
    
    # Analytic derivative
    dB_analytic = basis(x, derivative=True)
    
    # Numerical derivative using finite differences
    eps = 1e-7
    B_plus = basis(x + eps)
    B_minus = basis(x - eps)
    dB_numerical = (B_plus - B_minus) / (2 * eps)
    
    assert torch.allclose(dB_analytic, dB_numerical, atol=1e-5), \
        f"Derivative mismatch: max error = {(dB_analytic - dB_numerical).abs().max().item():.2e}"


@pytest.mark.parametrize("deg", DEGREES)
def test_bspline_interval_property(deg):
    """Verify interval property matches knot bounds."""
    knots = torch.linspace(-2, 3, 10, dtype=torch.float64)
    basis = BSplineBasis(knots, deg=deg)
    
    interval = basis.interval
    assert interval[0] == -2.0, f"Interval start mismatch: {interval[0]}"
    assert interval[1] == 3.0, f"Interval end mismatch: {interval[1]}"


def test_bspline_repr():
    """Test string representation."""
    basis = make_bspline_basis(deg=3)
    repr_str = repr(basis)
    
    assert "BSplineBasis" in repr_str
    assert "deg=3" in repr_str
    assert "n=" in repr_str


def test_bspline_different_knot_spacings():
    """Test BSpline with non-uniform knot spacing."""
    # Non-uniform knots
    knots = torch.tensor([0.0, 0.1, 0.3, 0.7, 0.8, 1.0], dtype=torch.float64)
    basis = BSplineBasis(knots, deg=2)
    
    x = torch.linspace(0, 1, 100, dtype=torch.float64)
    B = basis(x)
    
    # Should still satisfy partition of unity
    sums = B.sum(dim=0)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-12), \
        "Partition of unity failed with non-uniform knots"


# =============================================================================
# Device Testing (CPU and CUDA)
# =============================================================================

def test_basis_evaluation_on_device(device):
    """Verify basis evaluation works and output stays on same device."""
    basis = make_bspline_basis(deg=3)
    
    x = torch.linspace(0, 1, 50, dtype=torch.float64, device=device)
    B = basis(x)
    
    assert B.device.type == device, f"Output device mismatch: {B.device} != {device}"
    assert B.shape == (basis.n, 50)
    
    # Check partition of unity on device
    sums = B.sum(dim=0)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-12)


def test_interpolation_on_device(device):
    """Full interpolation pipeline on specified device."""
    basis = make_bspline_basis(deg=3)
    
    def poly_func(x):
        return x**3 - 2*x**2 + x + 0.5
    
    # Get interpolating points
    pts, matrix = basis.interpolating_points()
    pts = pts.to(device)
    matrix = matrix.to(device)
    
    # Solve for coefficients
    f_vals = poly_func(pts)
    coeffs = torch.linalg.solve(matrix.T, f_vals)
    
    # Evaluate on device
    x_test = torch.linspace(0, 1, 100, dtype=torch.float64, device=device)
    B_test = basis(x_test)
    
    approx = coeffs @ B_test
    exact = poly_func(x_test)
    
    assert approx.device.type == device
    assert rel_err(approx, exact) < 1e-10, \
        f"Interpolation on {device} failed: rel_err={rel_err(approx, exact):.2e}"


def test_autograd_on_device(device):
    """Gradient computation on specified device."""
    basis = make_bspline_basis(deg=3)
    
    x = torch.linspace(0.1, 0.9, 50, dtype=torch.float64, device=device, requires_grad=True)
    B = basis(x)
    
    loss = (B ** 2).sum()
    grad, = torch.autograd.grad(loss, x)
    
    assert grad.device.type == device, f"Gradient device mismatch"
    assert not torch.isnan(grad).any()
    assert not torch.isinf(grad).any()


def test_batched_evaluation_on_device(device):
    """Test batched evaluation on specified device."""
    basis = make_bspline_basis(deg=2)
    
    x = torch.rand(16, 32, dtype=torch.float64, device=device)
    B = basis(x)
    
    assert B.device.type == device
    assert B.shape == (basis.n, 16, 32)


# =============================================================================
# Edge Cases and Robustness
# =============================================================================

def test_bspline_boundary_evaluation():
    """Test evaluation exactly at boundaries."""
    basis = make_bspline_basis(deg=3)
    
    # Evaluate at exact boundaries
    x = torch.tensor([0.0, 1.0], dtype=torch.float64)
    B = basis(x)
    
    # Should still sum to 1
    sums = B.sum(dim=0)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-12)
    
    # First basis function should be 1 at x=0, last at x=1
    assert abs(B[0, 0].item() - 1.0) < 1e-12, "First basis should be 1 at left boundary"
    assert abs(B[-1, 1].item() - 1.0) < 1e-12, "Last basis should be 1 at right boundary"


def test_bspline_single_point():
    """Test evaluation at a single point."""
    basis = make_bspline_basis(deg=2)
    
    x = torch.tensor(0.5, dtype=torch.float64)
    B = basis(x)
    
    assert B.shape == (basis.n,)
    assert abs(B.sum().item() - 1.0) < 1e-12


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

