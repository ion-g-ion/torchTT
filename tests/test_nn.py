"""
Test the TT neural network layers.
"""
import pytest
import torchtt as tntt
import torch as tn


@pytest.mark.parametrize("dtype", [tn.float32, tn.float64])
@pytest.mark.parametrize("initializer", ['He', 'Glo'])
@pytest.mark.parametrize("batch", [[], [5], [2, 3]])
def test_linear_layer(dtype, initializer, batch):
    tn.manual_seed(0)
    layer = tntt.nn.LinearLayerTT([2, 3], [3, 2], [1, 2, 1], dtype=dtype, initializer=initializer)
    with tn.no_grad():
        layer.bias.uniform_(-1, 1)
    x = tn.randn(batch + [2, 3], dtype=dtype, requires_grad=True)
    W = tntt.TT(list(layer.cores)).full()

    y = layer(x)
    ref = tn.einsum('...ij,abij->...ab', x, W) + layer.bias
    tol = 1e-5 if dtype == tn.float32 else 1e-12

    assert y.shape == tuple(batch + [3, 2])
    assert tn.allclose(y, ref, rtol=tol, atol=tol)
    params = [x] + list(layer.parameters())
    grads = tn.autograd.grad(y.square().sum(), params)
    refs = tn.autograd.grad(ref.square().sum(), params)
    assert all(tn.allclose(g, r, rtol=tol, atol=tol) for g, r in zip(grads, refs))


def test_linear_layer_invalid_initializer():
    with pytest.raises(tntt.errors.InvalidArguments):
        tntt.nn.LinearLayerTT([2, 3], [3, 2], [1, 2, 1], initializer='invalid')


def test_compressed_layer_linear():
    tn.manual_seed(1)
    layer = tntt.nn.CompressedTTLayer(
        [2, 3, 2], [3, 2, 4], [1, 2, 2, 1], [1, 16, 16, 1],
        activation=None, bias=False, dtype=tn.float64)
    x = tntt.randn([2, 3, 2], [1, 2, 2, 1], dtype=tn.float64)
    W = tntt.TT(list(layer.cores)).full()

    y = layer(x)
    ref = tn.einsum('abcijk,ijk->abc', W, x.full())

    assert y.N == [3, 2, 4]
    assert tn.linalg.norm(y.full()-ref) / tn.linalg.norm(ref) < 1e-12


def test_compressed_layer_single_mode():
    tn.manual_seed(2)
    layer = tntt.nn.CompressedTTLayer([3], [4], [1, 1], [1, 1], dtype=tn.float64)
    with tn.no_grad():
        layer.bias[0].fill_(0.5)
    x = tntt.TT(tn.tensor([1., -2., 3.], dtype=tn.float64))

    y = layer(x)
    ref = tn.relu(layer.cores[0][0, :, :, 0] @ x.full() + 0.5)

    assert y.R == [1, 1]
    assert tn.allclose(y.full(), ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("activation", [tn.relu, tn.tanh])
def test_compressed_layer_rank_limit(activation):
    tn.manual_seed(3)
    layer = tntt.nn.CompressedTTLayer(
        [3, 4, 3], [4, 3, 5], [1, 3, 3, 1], [1, 1, 2, 1], activation=activation, dtype=tn.float64)
    x = tntt.randn([3, 4, 3], [1, 3, 3, 1], dtype=tn.float64)

    y = layer(x)

    assert y.N == [4, 3, 5]
    assert all(r <= limit for r, limit in zip(y.R, layer.R_output))
    assert tn.isfinite(y.full()).all()
    if activation == tn.tanh:
        # ReLU can give repeated zero singular values, where SVD gradients are undefined.
        y.full().square().sum().backward()
        assert all(p.grad is not None and tn.isfinite(p.grad).all() for p in layer.parameters())


@pytest.mark.parametrize("rank", [[1, 2], [2, 2, 1], [1, 2, 2]])
def test_compressed_layer_invalid_rank(rank):
    with pytest.raises(tntt.errors.InvalidArguments):
        tntt.nn.CompressedTTLayer([2, 3], [3, 2], [1, 2, 1], rank)
