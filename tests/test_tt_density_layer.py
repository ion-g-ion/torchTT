import torch as tn
import pytest
import torchtt
from torchtt.nn import (
    TTDensityLayer, 
    AffineTransform, 
    Rank1Shear, 
    TriangularShear, 
    TriangularPolyShear, 
    SinhArcsinhWarp, 
    ComposedTransform
)
from torchtt.functional.basis import BSplineBasis

def test_affine_transform():
    dim = 3
    transform = AffineTransform(dim)
    params = tn.randn(5, transform.input_requirement())
    x = tn.randn(5, dim)
    z, det_jac = transform(x, params)
    assert z.shape == x.shape
    assert det_jac.shape == (5,)

def test_rank1_shear():
    dim = 2
    transform = Rank1Shear(dim, degree=2, clip=5.0)
    params = tn.randn(4, transform.input_requirement())
    x = tn.randn(4, dim)
    z, det_jac = transform(x, params)
    assert z.shape == x.shape
    assert det_jac == 1.0

def test_triangular_shear():
    dim = 4
    transform = TriangularShear(dim)
    params = tn.randn(2, transform.input_requirement())
    x = tn.randn(2, dim)
    z, det_jac = transform(x, params)
    assert z.shape == x.shape
    assert det_jac == 1.0

def test_composed_transform():
    dim = 2
    t1 = AffineTransform(dim)
    t2 = Rank1Shear(dim, degree=3)
    t3 = SinhArcsinhWarp(dim)
    transform = ComposedTransform([t1, t2, t3])
    
    params = tn.randn(3, transform.input_requirement())
    x = tn.randn(3, dim)
    z, det_jac = transform(x, params)
    
    assert z.shape == x.shape
    assert det_jac.shape == (3,)

def test_tt_density_layer_bspline_decay():
    dim = 2
    knots = tn.linspace(-3, 3, 8, dtype=tn.float64)
    basis = [BSplineBasis(knots, deg=3, bc="decay") for _ in range(dim)]
    
    N = [b.n for b in basis]
    R = [1, 3, 1]
    
    transform = ComposedTransform([
        AffineTransform(dim), 
        Rank1Shear(dim, degree=2, clip=2.0)
    ])
    
    layer = TTDensityLayer(N, R, basis, transform=transform, dtype=tn.float64)
    
    batch_size = 5
    total_params = layer.input_requirement()
    
    tts = tn.randn(batch_size, total_params, dtype=tn.float64)
    x = tn.randn(batch_size, dim, dtype=tn.float64)
    
    # Forward pass
    pdf_eval = layer(tts, x)
    
    assert pdf_eval.shape == (batch_size,)
    assert not tn.isnan(pdf_eval).any()
    assert (pdf_eval >= 0).all() # Should be roughly positive or computable, though random cores might have negative regions

def test_tt_density_layer_no_transform():
    dim = 2
    knots = tn.linspace(-2, 2, 5, dtype=tn.float64)
    basis = [BSplineBasis(knots, deg=3, bc="clamped") for _ in range(dim)]
    
    N = [b.n for b in basis]
    R = [1, 2, 1]
    
    layer = TTDensityLayer(N, R, basis, transform=None, dtype=tn.float64)
    total_params = layer.input_requirement()
    
    tts = tn.randn(2, total_params, dtype=tn.float64)
    x = tn.randn(2, dim, dtype=tn.float64)
    
    pdf_eval = layer(tts, x)
    assert pdf_eval.shape == (2,)


@pytest.mark.parametrize("batch", [[], [4], [2, 3]])
def test_triangular_poly_shear(batch):
    transform = TriangularPolyShear(3, degree=2)
    params = tn.tensor([1., 2., 3., 4., 5., 6.], dtype=tn.float64)
    x = tn.randn(batch + [3], dtype=tn.float64)

    z, det_jac = transform(x, params)
    ref = tn.stack([x[..., 0] + x[..., 1] + 2*x[..., 1]**2 + 3*x[..., 2] + 4*x[..., 2]**2,
                    x[..., 1] + 5*x[..., 2] + 6*x[..., 2]**2, x[..., 2]], dim=-1)

    assert transform.input_requirement() == 6
    assert tn.allclose(z, ref, rtol=1e-12, atol=1e-12)
    point = x.reshape(-1, 3)[0]
    jac = tn.autograd.functional.jacobian(lambda v: transform(v, params)[0], point)
    assert tn.allclose(tn.linalg.det(jac), tn.tensor(det_jac, dtype=tn.float64))


@pytest.mark.parametrize("batch", [[], [3], [2, 3]])
def test_tt_density_layer_get_tt(batch):
    knots = tn.linspace(-2, 2, 5, dtype=tn.float64)
    basis = [BSplineBasis(knots, deg=2, bc='clamped') for _ in range(2)]
    N, R = [b.n for b in basis], [1, 2, 1]
    layer = TTDensityLayer(N, R, basis, transform=TriangularShear(2), dtype=tn.float64)
    count = tn.Size(batch).numel()
    tensors = [torchtt.randn(N, R) for _ in range(count)]
    params = tn.stack([tn.cat([c.flatten() for c in t.cores] + [tn.ones(1, dtype=tn.float64)]) for t in tensors])
    params = params.reshape(batch + [layer.input_requirement()])

    result = layer.get_tt(params)

    assert len(result) == count
    for t, ref in zip(result, tensors):
        assert t.N == N and t.R == R
        assert tn.equal(t.full(), ref.full())
