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
    coeffs = torch.linalg.solve(matrix, f_vals)
    
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
    coeffs = torch.linalg.solve(matrix, f_vals)
    
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


@pytest.mark.parametrize("deg", DEGREES)
def test_bspline_mass_matrix(deg):
    """Test mass matrix properties."""
    basis = make_bspline_basis(deg)
    M = basis.mass_matrix()
    
    # Check shape
    assert M.shape == (basis.n, basis.n)
    
    # Mass matrix must be symmetric
    assert torch.allclose(M, M.T, atol=1e-12), "Mass matrix is not symmetric"
    
    # Sum of rows should equal integration weights (partition of unity)
    # sum_j M_ij = sum_j int B_i B_j dx = int B_i sum_j B_j dx = int B_i dx
    row_sums = M.sum(dim=1)
    expected_weights = basis.integration_weights()
    assert torch.allclose(row_sums, expected_weights, atol=1e-12), "Mass matrix row sums do not match integration weights"


@pytest.mark.parametrize("deg", [2, 3])  # Need deg >= 1 for meaningful derivative
def test_bspline_stiffness_matrix(deg):
    """Test stiffness matrix properties."""
    basis = make_bspline_basis(deg)
    S = basis.stiffness_matrix()
    
    # Check shape
    assert S.shape == (basis.n, basis.n)
    
    # Stiffness matrix must be symmetric
    assert torch.allclose(S, S.T, atol=1e-12), "Stiffness matrix is not symmetric"
    
    # Sum of rows should be zero (partition of unity => sum B_j = 1 => sum B'_j = 0)
    # sum_j S_ij = int B'_i sum_j B'_j dx = 0
    row_sums = S.sum(dim=1)
    assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-12), "Stiffness matrix row sums are not zero"


@pytest.mark.parametrize("deg", [2, 3])  # Need deg >= 1 for meaningful derivative
def test_bspline_advection_matrix(deg):
    """Test advection matrix properties."""
    basis = make_bspline_basis(deg)
    C = basis.advection_matrix()
    
    # Check shape
    assert C.shape == (basis.n, basis.n)
    
    # Sum of rows should be zero (sum B'_j = 0)
    row_sums = C.sum(dim=1)
    assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-12), "Advection matrix row sums are not zero"
    
    # C + C^T should only have non-zeros at the boundary corners
    # C_ij + C_ji = int B_i B'_j + B'_i B_j dx = int d/dx(B_i B_j) dx = [B_i B_j]_a^b
    C_sym = C + C.T
    expected_C_sym = torch.zeros_like(C_sym)
    expected_C_sym[0, 0] = -1.0  # B_0(a) = 1, B_0(b) = 0 => 0 - 1 = -1
    expected_C_sym[-1, -1] = 1.0  # B_n(a) = 0, B_n(b) = 1 => 1 - 0 = 1
    
    assert torch.allclose(C_sym, expected_C_sym, atol=1e-12), "C + C.T does not match expected boundary values"


# =============================================================================
# Boundary Modes (bc): "zero" and "decay"
# =============================================================================

BC_CONFIGS = [
    ("zero", "clamped"),
    ("clamped", "zero"),
    ("zero", "zero"),
    ("clamped", "decay"),
    ("decay", "clamped"),
    ("decay", "decay"),
    ("zero", "decay"),
    ("decay", "zero"),
]


def make_bc_basis(deg, bc, num_knots=6, decay_rate=None):
    """Factory for BSplineBasis with boundary modes on [0, 1]."""
    knots = torch.linspace(0, 1, num_knots, dtype=torch.float64)
    return BSplineBasis(knots, deg=deg, bc=bc, decay_rate=decay_rate)


@pytest.mark.parametrize("deg", DEGREES)
@pytest.mark.parametrize("bc", [("zero", "clamped"), ("clamped", "zero"), ("zero", "zero")])
def test_bspline_bc_zero_dimension_and_boundary_values(deg, bc):
    """'zero' sides reduce n by one and make all basis functions vanish there."""
    basis = make_bc_basis(deg, bc)
    n_clamped = 5 + deg
    n_expected = n_clamped - bc.count("zero")
    assert basis.n == n_expected, f"Expected n={n_expected}, got n={basis.n}"

    B = basis(torch.tensor([0.0, 1.0], dtype=torch.float64))
    if bc[0] == "zero":
        assert B[:, 0].abs().max() < 1e-14, "Basis does not vanish at the left boundary"
    if bc[1] == "zero":
        assert B[:, 1].abs().max() < 1e-14, "Basis does not vanish at the right boundary"


@pytest.mark.parametrize("deg", DEGREES)
def test_bspline_bc_zero_weights_match_clamped(deg):
    """'zero' integration weights equal the clamped weights of the retained functions."""
    clamped = make_bspline_basis(deg)
    w_clamped = clamped.integration_weights()

    w_left = make_bc_basis(deg, ("zero", "clamped")).integration_weights()
    w_both = make_bc_basis(deg, ("zero", "zero")).integration_weights()

    assert torch.allclose(w_left, w_clamped[1:], atol=1e-14)
    assert torch.allclose(w_both, w_clamped[1:-1], atol=1e-14)


@pytest.mark.parametrize("deg", DEGREES)
@pytest.mark.parametrize("bc", [("clamped", "decay"), ("decay", "clamped"), ("decay", "decay")])
def test_bspline_decay_partition_of_unity_deep_interior(deg, bc):
    """'decay' sides drop the out-of-domain crossing functions, so partition of
    unity is lost within ~deg knot spans of a decay boundary but still holds in
    the deep interior away from such boundaries."""
    basis = make_bc_basis(deg, bc, num_knots=12)
    a, b = basis.interval
    margin = deg * (b - a) / 11  # ~deg knot spans
    lo = a + (margin if bc[0] == "decay" else 0.0)
    hi = b - (margin if bc[1] == "decay" else 0.0)
    x = torch.linspace(lo, hi, 200, dtype=torch.float64)
    sums = basis(x).sum(dim=0)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-10), \
        f"Partition of unity violated in the deep interior: max dev = {(sums - 1).abs().max().item():.2e}"


@pytest.mark.parametrize("deg", DEGREES)
@pytest.mark.parametrize("bc", [("clamped", "decay"), ("decay", "clamped"), ("decay", "decay"),
                                ("zero", "decay"), ("decay", "zero")])
def test_bspline_decay_no_function_centered_outside(deg, bc):
    """No retained basis function is centered (Greville) outside [a, b]: the
    outermost decay function peaks at the boundary and decays monotonically."""
    basis = make_bc_basis(deg, bc)
    a, b = basis.interval
    pts, _ = basis.interpolating_points()
    assert (pts >= a - 1e-9).all() and (pts <= b + 1e-9).all(), \
        f"Some interpolating point lies outside [a, b]: {pts}"

    # The outermost decay function decays monotonically past the boundary
    if bc[1] == "decay":
        xr = torch.linspace(b, b + 5.0 / basis.decay_rate[1], 300, dtype=torch.float64)
        last = basis(xr)[-1]
        assert (last[1:] <= last[:-1] + 1e-9).all(), "Right decay tail is not monotone"
    if bc[0] == "decay":
        xl = torch.linspace(a - 5.0 / basis.decay_rate[0], a, 300, dtype=torch.float64)
        first = basis(xl)[0]
        assert (first[:-1] <= first[1:] + 1e-9).all(), "Left decay tail is not monotone"


@pytest.mark.parametrize("deg", [1, 2, 3, 4])
def test_bspline_decay_smoothness_at_boundary(deg):
    """The exponential tails join the spline with C^(deg-1) smoothness."""
    basis = make_bc_basis(deg, ("clamped", "decay"))
    eps = 1e-6
    xl = torch.tensor([1.0 - eps], dtype=torch.float64)
    xr = torch.tensor([1.0 + eps], dtype=torch.float64)

    # Value continuity (a discontinuity would show up as an O(1) jump)
    jump0 = (basis(xl) - basis(xr)).abs().max().item()
    assert jump0 < 1e-4, f"Value jump at the boundary: {jump0:.2e}"

    if deg >= 2:
        # First derivative continuity (a kink would show up as an O(deg/h) jump)
        jump1 = (basis(xl, derivative=True) - basis(xr, derivative=True)).abs().max().item()
        assert jump1 < 1e-2, f"First derivative jump at the boundary: {jump1:.2e}"

    if deg >= 3:
        # Second derivative continuity via finite differences of the first derivative
        eps = 1e-5
        ts = lambda v: torch.tensor([v], dtype=torch.float64)
        fd2_l = (basis(ts(1.0 - eps), derivative=True) - basis(ts(1.0 - 2 * eps), derivative=True)) / eps
        fd2_r = (basis(ts(1.0 + 2 * eps), derivative=True) - basis(ts(1.0 + eps), derivative=True)) / eps
        jump2 = (fd2_l - fd2_r).abs().max().item()
        scale = 1.0 + max(fd2_l.abs().max().item(), fd2_r.abs().max().item())
        assert jump2 < 1e-2 * scale, f"Second derivative jump at the boundary: {jump2:.2e}"


@pytest.mark.parametrize("deg", DEGREES)
@pytest.mark.parametrize("bc", [("clamped", "decay"), ("zero", "decay"), ("decay", "decay")])
def test_bspline_decay_integration_weights_numeric(deg, bc):
    """Integration weights over the unbounded domain match dense numerical integration."""
    basis = make_bc_basis(deg, bc)
    lam_l, lam_r = basis.decay_rate
    lo = -50.0 / lam_l if lam_l else 0.0
    hi = 1.0 + (50.0 / lam_r if lam_r else 0.0)
    x = torch.linspace(lo, hi, 400_001, dtype=torch.float64)
    w_numeric = torch.trapezoid(basis(x), x, dim=1)
    w = basis.integration_weights()
    assert torch.allclose(w, w_numeric, atol=1e-6), \
        f"Weights mismatch: {(w - w_numeric).abs().max().item():.2e}"


def test_bspline_decay_weight_closed_form_deg1():
    """For deg=1 the last weight is the hat area h/2 plus the tail mass 1/lambda."""
    basis = make_bc_basis(1, ("clamped", "decay"), decay_rate=3.0)
    w = basis.integration_weights()
    h = 0.2  # spacing of linspace(0, 1, 6)
    assert abs(w[-1].item() - (h / 2 + 1 / 3.0)) < 1e-12


@pytest.mark.parametrize("deg", [1, 2, 3, 4])
@pytest.mark.parametrize("bc", BC_CONFIGS)
def test_bspline_bc_interpolation_roundtrip(deg, bc):
    """Interpolating points give a well-conditioned matrix; coefficients of any
    function in the span are recovered exactly from its values at the points."""
    basis = make_bc_basis(deg, bc)
    pts, matrix = basis.interpolating_points()

    assert pts.shape == (basis.n,)
    assert matrix.shape == (basis.n, basis.n)
    assert torch.linalg.cond(matrix) < 1e4

    # 'decay' sides drop out-of-domain crossing functions, so all retained
    # interpolating points stay within [0, 1]
    assert (pts >= -1e-9).all() and (pts <= 1.0 + 1e-9).all(), \
        f"Interpolating points should lie within [0, 1] after trimming, got {pts}"

    gen = torch.Generator().manual_seed(42)
    coeffs = torch.randn(basis.n, dtype=torch.float64, generator=gen)
    f_vals = coeffs @ basis(pts)
    coeffs_rec = torch.linalg.solve(matrix, f_vals)
    assert torch.allclose(coeffs, coeffs_rec, atol=1e-10), \
        f"Round-trip failed: {(coeffs - coeffs_rec).abs().max().item():.2e}"


@pytest.mark.parametrize("deg", DEGREES)
def test_bspline_decay_quadrature_consistency(deg):
    """Quadrature with tail points integrates span functions to the same value
    as the analytic integration weights."""
    basis = make_bc_basis(deg, ("decay", "decay"))
    gen = torch.Generator().manual_seed(0)
    coeffs = torch.rand(basis.n, dtype=torch.float64, generator=gen)

    pts, w = basis.quadrature(deg + 1)
    integral_quad = (coeffs @ basis(pts)) @ w
    integral_weights = coeffs @ basis.integration_weights()
    assert abs(integral_quad.item() - integral_weights.item()) < 1e-12


@pytest.mark.parametrize("deg", [2, 3])
def test_bspline_decay_matrices(deg):
    """Mass/stiffness/advection matrices with decay tails match dense numerical
    integration over a truncated domain; the mass matrix is SPD."""
    basis = make_bc_basis(deg, ("clamped", "decay"))
    lam = basis.decay_rate[1]
    x = torch.linspace(0, 1 + 40 / lam, 1_000_001, dtype=torch.float64)
    B = basis(x)
    Bp = basis(x, derivative=True)

    M = basis.mass_matrix()
    S = basis.stiffness_matrix()
    C = basis.advection_matrix()

    M_num = torch.trapezoid(B.unsqueeze(1) * B.unsqueeze(0), x, dim=2)
    S_num = torch.trapezoid(Bp.unsqueeze(1) * Bp.unsqueeze(0), x, dim=2)
    C_num = torch.trapezoid(B.unsqueeze(1) * Bp.unsqueeze(0), x, dim=2)

    assert rel_err(M, M_num) < 1e-6, f"Mass matrix mismatch: {rel_err(M, M_num):.2e}"
    assert rel_err(S, S_num) < 1e-6, f"Stiffness matrix mismatch: {rel_err(S, S_num):.2e}"
    assert rel_err(C, C_num) < 1e-6, f"Advection matrix mismatch: {rel_err(C, C_num):.2e}"

    assert torch.allclose(M, M.T, atol=1e-12), "Mass matrix is not symmetric"
    assert torch.linalg.eigvalsh(M).min() > 0, "Mass matrix is not positive definite"


@pytest.mark.parametrize("deg", DEGREES)
def test_bspline_decay_mirror_symmetry(deg):
    """A left 'decay' basis is the mirror image of a right 'decay' basis."""
    basis_l = make_bc_basis(deg, ("decay", "clamped"), decay_rate=4.0)
    basis_r = make_bc_basis(deg, ("clamped", "decay"), decay_rate=4.0)

    # Avoid knots: B-spline derivatives are one-sided at kinks for deg=1
    x = torch.linspace(-2, 1, 567, dtype=torch.float64) + 0.0123 / 567
    B_l = basis_l(x)
    B_r = basis_r(1.0 - x)
    assert torch.allclose(B_l, B_r.flip(0), atol=1e-12), "Mirror symmetry violated"

    dB_l = basis_l(x, derivative=True)
    dB_r = basis_r(1.0 - x, derivative=True)
    assert torch.allclose(dB_l, -dB_r.flip(0), atol=1e-10), "Mirror symmetry of derivatives violated"


def test_bspline_decay_autograd_tail():
    """Gradients flow through the exponential tail region and match the analytic derivative."""
    basis = make_bc_basis(3, ("clamped", "decay"))
    x = torch.linspace(0.5, 2.5, 100, dtype=torch.float64, requires_grad=True)
    B = basis(x)
    grad, = torch.autograd.grad(B.sum(), x)

    assert torch.isfinite(grad).all(), "Gradient contains NaN/Inf"
    assert grad[x.detach() > 1.0].abs().max() > 0, "Gradient vanishes in the tail region"

    dB = basis(x, derivative=True).sum(dim=0)
    assert torch.allclose(grad, dB.detach(), atol=1e-12), "Autograd disagrees with analytic derivative"


@pytest.mark.parametrize("deg", [2, 3, 4])
def test_bspline_decay_default_rate_tails_nearly_nonnegative(deg):
    """With the default decay rate the tails have at most a tiny undershoot."""
    basis = make_bc_basis(deg, ("clamped", "decay"))
    lam = basis.decay_rate[1]
    x = torch.linspace(1, 1 + 30 / lam, 20_001, dtype=torch.float64)
    assert basis(x).min() > -0.02, f"Tail undershoot too large: {basis(x).min().item():.2e}"


def test_bspline_bc_invalid_arguments():
    """Invalid boundary mode specifications raise ValueError."""
    knots = torch.linspace(0, 1, 6, dtype=torch.float64)
    with pytest.raises(ValueError):
        BSplineBasis(knots, deg=2, bc="bogus")
    with pytest.raises(ValueError):
        BSplineBasis(knots, deg=0, bc=("zero", "clamped"))
    with pytest.raises(ValueError):
        BSplineBasis(knots, deg=0, bc=("clamped", "decay"))
    with pytest.raises(ValueError):
        BSplineBasis(knots, deg=2, bc=("clamped", "decay"), decay_rate=-1.0)


def test_bspline_bc_repr():
    """Non-default boundary modes show up in the repr."""
    basis = make_bc_basis(2, ("zero", "decay"))
    repr_str = repr(basis)
    assert "BSplineBasis" in repr_str
    assert "zero" in repr_str and "decay" in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

