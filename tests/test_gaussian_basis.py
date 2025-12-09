"""
Test suite for Gaussian basis functions in torchtt.functional.basis.
"""
import torch
import pytest
import math
from torchtt.functional import GaussianBasis

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def gaussian_basis():
    knots = torch.linspace(0, 1, 10, dtype=torch.float64)
    return GaussianBasis(knots, delta_overlap=2)

# =============================================================================
# Generic Basis Functionality for Gaussian
# =============================================================================

def test_basis_dimension(gaussian_basis):
    """Verify n property returns correct number of basis functions."""
    # For Gaussian: n = num_knots
    assert gaussian_basis.n == 10

def test_basis_evaluation_shape(gaussian_basis):
    """Verify calling basis returns shape (n, ...) matching input shape."""
    # Test 1D input
    x1d = torch.linspace(0, 1, 50, dtype=torch.float64)
    B1d = gaussian_basis(x1d)
    assert B1d.shape == (gaussian_basis.n, 50)
    
    # Test 2D input
    x2d = torch.rand(10, 20, dtype=torch.float64)
    B2d = gaussian_basis(x2d)
    assert B2d.shape == (gaussian_basis.n, 10, 20)

def test_interpolating_points_matrix_invertibility(gaussian_basis):
    """Check that interpolating points return an invertible matrix."""
    pts, matrix = gaussian_basis.interpolating_points()
    
    assert pts.shape == (gaussian_basis.n,)
    assert matrix.shape == (gaussian_basis.n, gaussian_basis.n)
    
    # Check invertibility via condition number
    cond = torch.linalg.cond(matrix)
    # Gaussian matrices can be ill-conditioned if overlap is large, but should be solvable
    assert cond < 1e15, f"Matrix is too ill-conditioned (cond={cond.item():.2e})"

def test_batched_evaluation(gaussian_basis):
    """Test that basis correctly handles batched inputs."""
    batch_size = 8
    num_points = 32
    x_batched = torch.rand(batch_size, num_points, dtype=torch.float64)
    B_batched = gaussian_basis(x_batched)
    
    assert B_batched.shape == (gaussian_basis.n, batch_size, num_points)
    
    # Verify consistency
    B_single = gaussian_basis(x_batched[0])
    assert torch.allclose(B_batched[:, 0, :], B_single, atol=1e-14)

def test_autograd_forward(gaussian_basis):
    """Test gradients flow through basis evaluation."""
    x = torch.linspace(0.1, 0.9, 50, dtype=torch.float64, requires_grad=True)
    B = gaussian_basis(x)
    
    loss = B.sum()
    grad, = torch.autograd.grad(loss, x)
    
    assert grad.shape == x.shape
    assert not torch.isnan(grad).any()

def test_derivative_evaluation(gaussian_basis):
    """Test derivative computation against numerical differentiation."""
    x = torch.linspace(0.1, 0.9, 50, dtype=torch.float64)
    
    # Analytic derivative
    dB_analytic = gaussian_basis(x, derivative=True)
    
    # Numerical derivative
    eps = 1e-7
    B_plus = gaussian_basis(x + eps)
    B_minus = gaussian_basis(x - eps)
    dB_numerical = (B_plus - B_minus) / (2 * eps)
    
    # Gaussian derivative is smooth, should match well
    assert torch.allclose(dB_analytic, dB_numerical, atol=1e-5)

def test_integration_weights(gaussian_basis):
    """Test integration weights."""
    weights = gaussian_basis.integration_weights()
    
    assert weights.shape == (gaussian_basis.n,)
    assert (weights > 0).all()
    
    # The sum of weights isn't necessarily 1 for Gaussian basis, unlike B-splines.
    # Check if the integral matches the analytical formula for full real line
    
    # Sigma for delta_overlap=2 (from fixture)
    # knots = 0, 0.111..., 0.222... (10 points from 0 to 1)
    # gap = 1/9
    # sigma = 2 * gap = 2/9
    expected_sigma = 2.0 / 9.0
    expected_weight = expected_sigma * math.sqrt(2 * math.pi)
    
    # Check if weights are close to expected value (should be constant if knots are uniform)
    assert torch.allclose(weights, torch.tensor(expected_weight, dtype=torch.float64), atol=1e-6)

def test_gaussian_params():
    """Test Gaussian specific parameters."""
    knots = torch.linspace(0, 1, 5, dtype=torch.float64)
    basis = GaussianBasis(knots, delta_overlap=1)
    
    # Check centers
    assert torch.allclose(basis._centers, knots)
    
    # Check sigmas
    # For delta_overlap=1, sigma[i] = |knots[i+1] - knots[i]| = 0.25
    expected_sigma = 0.25
    assert torch.allclose(basis._sigmas, torch.tensor(expected_sigma, dtype=torch.float64))

def test_empty_knots():
    with pytest.raises(ValueError):
        GaussianBasis(torch.tensor([]), delta_overlap=1)

def test_invalid_delta():
    knots = torch.linspace(0, 1, 5)
    with pytest.raises(ValueError):
        GaussianBasis(knots, delta_overlap=0)


