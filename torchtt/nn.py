"""
Implements a basic TT layer for constructing deep TT networks.

"""
import torch as tn
import torch.nn as nn
import torchtt 
import opt_einsum as oe
from ._aux_ops import dense_matvec
from .errors import *


class Transform(nn.Module):
    """
    Base class for diffeomorphism transformations used in TTDensityLayer.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def input_requirement(self):
        """Returns the number of parameters required from the NN per sample."""
        raise NotImplementedError

    def forward(self, x, params):
        """
        Applies the transformation.
        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).
            params (torch.Tensor): Parameter tensor of shape (..., input_requirement).
        Returns:
            z (torch.Tensor): Transformed tensor.
            det_jac (torch.Tensor or float): Determinant of the Jacobian.
        """
        raise NotImplementedError


class AffineTransform(Transform):
    """
    Affine transformation block:
    z = R(theta) diag(exp(a)) x + b

    Where R(theta) is a rotation matrix parameterized by Givens angles.
    det J = prod(exp(a))
    """
    def input_requirement(self):
        return self.dim * (self.dim - 1) // 2 + 2 * self.dim

    def _create_rotation_matrix(self, angles, device, dtype):
        batch_shape = angles.shape[:-1]
        R = tn.eye(self.dim, device=device, dtype=dtype)
        view_shape = [1] * len(batch_shape) + [self.dim, self.dim]
        R = R.view(*view_shape).repeat(*batch_shape, 1, 1)
        
        k = 0
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                theta = angles[..., k] 
                c = tn.cos(theta).unsqueeze(-1)
                s = tn.sin(theta).unsqueeze(-1)
                R_i = R[..., :, i].clone()
                R_j = R[..., :, j].clone()
                R[..., :, i] = c * R_i + s * R_j
                R[..., :, j] = -s * R_i + c * R_j
                k += 1
        return R

    def forward(self, x, params):
        n_pairs = self.dim * (self.dim - 1) // 2
        angles = params[..., :n_pairs]
        scales = tn.exp(params[..., n_pairs:n_pairs+self.dim])
        offsets = params[..., n_pairs+self.dim:]

        R = self._create_rotation_matrix(angles, params.device, params.dtype)
        x = x * scales
        z = tn.einsum('...ij,...j->...i', R, x) + offsets
        det_jac = tn.prod(scales, dim=-1)
        return z, det_jac


class Rank1Shear(Transform):
    """
    Volume-preserving rank-1 nonlinear shear:
    z = x + u * g(v^T x + c)
    
    To ensure volume preservation (det J = 1), u is projected orthogonally to v such that v^T u = 0.
    The scalar function g(t) is a polynomial of degree D with no constant term: g(t) = sum_{p=1}^D alpha_p t^p.
    If clipping is applied, the displacement is squashed to prevent numerical overflow: 
    g_clip(t) = clip * tanh(g(t) / clip).
    """
    def __init__(self, dim, degree=2, clip=None):
        super().__init__(dim)
        self.degree = degree
        self.clip = clip

    def input_requirement(self):
        return 2 * self.dim + 1 + self.degree

    @staticmethod
    def _eval_poly(coeffs, t):
        g = tn.zeros(tn.broadcast_shapes(coeffs.shape[:-1], t.shape), dtype=t.dtype, device=t.device)
        for p in range(coeffs.shape[-1] - 1, -1, -1):
            g = (g + coeffs[..., p]) * t
        return g

    def forward(self, x, params):
        u_raw = params[..., :self.dim]
        v = params[..., self.dim:2*self.dim]
        c = params[..., 2*self.dim]
        alpha = params[..., 2*self.dim+1:]

        u = u_raw - ((v * u_raw).sum(-1, keepdim=True) / ((v * v).sum(-1, keepdim=True) + 1e-12)) * v
        g = self._eval_poly(alpha, (v * x).sum(-1) + c)
        if self.clip is not None:
            g = self.clip * tn.tanh(g / self.clip)
        z = x + u * g.unsqueeze(-1)
        return z, 1.0


class TriangularShear(Transform):
    """
    Unit upper-triangular linear shear (completes the affine family).
    det J = 1.
    """
    def input_requirement(self):
        return self.dim * (self.dim - 1) // 2

    def forward(self, x, params):
        components = []
        k = 0
        for i in range(self.dim):
            xi = x[..., i]
            for j in range(i + 1, self.dim):
                xi = xi + params[..., k] * x[..., j]
                k += 1
            components.append(xi)
        z = tn.stack(tn.broadcast_tensors(*components), dim=-1)
        return z, 1.0


class TriangularPolyShear(Transform):
    """
    Knothe-Rosenblatt-style triangular polynomial shear:
    z_i = x_i + sum_{j>i} q_{ij}(x_j)

    Each q_{ij}(t) is a polynomial of degree D_poly with no constant term: q_{ij}(t) = sum_{p=1}^{D_poly} beta_{ij,p} t^p.
    det J = 1.
    """
    def __init__(self, dim, degree):
        super().__init__(dim)
        self.degree = degree

    def input_requirement(self):
        return (self.dim * (self.dim - 1) // 2) * self.degree

    def forward(self, x, params):
        n_pairs = self.dim * (self.dim - 1) // 2
        beta = params.reshape(*params.shape[:-1], n_pairs, self.degree)
        
        components = []
        k = 0
        for i in range(self.dim):
            xi = x[..., i]
            for j in range(i + 1, self.dim):
                xi = xi + Rank1Shear._eval_poly(beta[..., k, :], x[..., j])
                k += 1
            components.append(xi)
        z = tn.stack(tn.broadcast_tensors(*components), dim=-1)
        return z, 1.0


class SinhArcsinhWarp(Transform):
    """
    Elementwise sinh-arcsinh warp:
    z_i = sinh(e^{s_i} asinh(x_i) + b_i)
    """
    def input_requirement(self):
        return 2 * self.dim

    def forward(self, x, params):
        sa_log_scale = params[..., :self.dim]
        sa_offset = params[..., self.dim:]
        a = tn.exp(sa_log_scale)
        inner = a * tn.asinh(x) + sa_offset
        det_jac = tn.prod(a * tn.cosh(inner) / tn.sqrt(1.0 + x*x), dim=-1)
        z = tn.sinh(inner)
        return z, det_jac


class ComposedTransform(Transform):
    """
    Chains multiple transformations.
    """
    def __init__(self, transforms):
        dim = transforms[0].dim if transforms else 0
        super().__init__(dim)
        self.transforms = nn.ModuleList(transforms)

    def input_requirement(self):
        return sum(t.input_requirement() for t in self.transforms)

    def forward(self, x, params):
        det_jac = 1.0
        sofar = 0
        for t in self.transforms:
            req = t.input_requirement()
            p = params[..., sofar:sofar+req]
            x, dj = t(x, p)
            if isinstance(dj, tn.Tensor) or dj != 1.0:
                det_jac = det_jac * dj
            sofar += req
        return x, det_jac


class TTDensityLayer(nn.Module):
    """
    A TT Density Layer evaluating p(x) = p_ref(T(x)) * |det J_T(x)|.
    """
    def __init__(self, N, R, basis, transform=None, dtype=tn.float32):
        super().__init__()
        if R[0] != 1 or R[-1] != 1 and len(R) != len(N)+1:
            raise InvalidArguments("The rank and the number of modes do not match.")
        if len(basis) != len(N):
            raise InvalidArguments("The number of bases must match the number of modes.")
        for i, (n, b) in enumerate(zip(N, basis)):
            if n != b.n:
                raise InvalidArguments(f"Mode size N[{i}]={n} does not match basis dimension {b.n}")

        self.dim = len(N)
        self.N = N
        self.R = R
        self.transform = transform
        self.dtype = dtype
        
        self.basis = nn.ModuleList(basis)
        for i, b in enumerate(basis):
            self.register_buffer(f'integration_weight_{i}', b.integration_weights().to(self.dtype))

    @property
    def integration_weights(self):
        return [getattr(self, f'integration_weight_{i}') for i in range(self.dim)]

    @property
    def _tt_params_size(self):
        return sum(self.N[i]*self.R[i]*self.R[i+1] for i in range(self.dim))

    def input_requirement(self):
        total = self._tt_params_size
        if self.transform is not None:
            total += self.transform.input_requirement()
        return total

    # Backward compatibility
    input_requireemnt = input_requirement

    def marginalize(self, tts, x, keep_indices):
        pass
    
    def get_tt(self, tts):
        batch_shape = tts.shape[:-1]
        tts_flat = tts.reshape(-1, tts.shape[-1])
        tt_list = []
        for batch_idx in range(tts_flat.shape[0]):
            cores = []
            sofar = 0
            for i in range(self.dim):
                core_size = self.N[i] * self.R[i] * self.R[i+1]
                core_flat = tts_flat[batch_idx, sofar:sofar+core_size]
                core = core_flat.view(self.R[i], self.N[i], self.R[i+1])
                cores.append(core)
                sofar += core_size
            tt_list.append(torchtt.TT(cores))
        return tt_list

    def forward(self, tts, x):
        if not isinstance(x, tn.Tensor):
            x = tn.tensor(x, dtype=self.dtype, device=tts.device)
        else:
            x = x.to(dtype=self.dtype)
             
        if tts.shape[-1] != self.input_requirement():
            raise InvalidArguments("The shape of the tensor does not match the input requirement.")
        if self.dim != x.shape[-1]:
            raise InvalidArguments("The dimension of the tensor does not match the dimension of the input.")
        
        sofar = 0
        cores = []
        for i in range(self.dim):
            core_size = self.N[i]*self.R[i]*self.R[i+1]
            cores.append(tts[..., sofar:sofar+core_size].view(*tts.shape[:-1], self.R[i], self.N[i], self.R[i+1]).to(self.dtype))
            sofar += core_size
        
        det_jac = 1.0
        if self.transform is not None:
            transform_params = tts[..., sofar:]
            x, det_jac = self.transform(x, transform_params)

        Bevals = [self.basis[i](x[..., i]).to(self.dtype) for i in range(self.dim)]
        
        denominator = oe.contract("n,...nb,...nd->...bd", self.integration_weights[0], cores[0][...,0,:,:], cores[0][...,0,:,:])
        for i in range(1, self.dim):
            denominator = oe.contract("...ab,n,...anc,...bne->...ce", denominator, self.integration_weights[i], cores[i], cores[i])
        
        nominator = oe.contract("n...,...nb,...nd->...bd", Bevals[0], cores[0][...,0,:,:], cores[0][...,0,:,:])
        for i in range(1, self.dim):
            nominator = oe.contract("...ab,n...,...anc,...bne->...ce", nominator, Bevals[i], cores[i], cores[i])
        
        pdf_eval = nominator[..., 0, 0] / denominator[..., 0, 0]
        if isinstance(det_jac, tn.Tensor) or det_jac != 1.0:
            pdf_eval = pdf_eval * det_jac

        return pdf_eval

class LinearLayerTT(nn.Module):
    """
    Basic class for TT layers. See `Tensorizing Neural Networks <https://arxiv.org/abs/1509.06569>`_ for a detailed description.
    It can be used similarily to any layer from `torch.nn`.
    The output of the layer is :math:`\\mathcal{LTT}(\\mathsf{x}) =\\mathsf{Wx}+\\mathsf{b}`, where the tensor operator :math:`\\mathsf{W}` is represented in the TT format (with a fixed prescribed rank).

    """
    def __init__(self, size_in, size_out, rank, dtype = tn.float32, initializer = 'He'):
        """
        The constructor of the TT layer class takes as arguments the input shape and the output shape for the layer, the rank as well as the dtype and the initializer.
        
        Possible initializers are:
        
         * ``'He'`` for He Normal (He-et-al) initialization.
         * ``'Glo'`` for Glorot initialization.
            
        Args:
            size_in (list[int]): the size of the input tensor.
            size_out (list[int]): the size of the output tensor.
            rank (list[int]): the rank of the tensor operator.
            dtype (torch.dtype, optional): the dtype of the layer. Defaults to torch.float32.
            initializer (str, optional): the initializer for the weights and biases. Defaults to 'He'.
            
        Raises:
            InvalidArguments: Initializer not defined. Possible choices are 'He' and 'Glo'.
        """
        super().__init__()
        self.size_in, self.size_out, self.rank = size_in, size_out, rank
        if initializer=='He':
            t = torchtt.randn([(s2,s1) for s1,s2 in zip(size_in,size_out)], rank, dtype=dtype, var = 2/tn.prod(tn.tensor([s1 for s1 in size_in])))
            #self.cores = [nn.Parameter(tn.Tensor(c.clone())) for c in t.cores] 
            self.cores = nn.ParameterList([nn.Parameter(c) for c in t.cores])
            #bias
            bias = tn.zeros(size_out, dtype = dtype) 
            self.bias = nn.Parameter(bias)
        elif initializer=='Glo':
            t = torchtt.randn([(s2,s1) for s1,s2 in zip(size_in,size_out)], rank, dtype=dtype, var = 1/(tn.prod(tn.tensor([s1 for s1 in size_in]))+tn.prod(tn.tensor([s1 for s1 in size_out]))) )
            #self.cores = [nn.Parameter(tn.Tensor(c.clone())) for c in t.cores] 
            self.cores = nn.ParameterList([nn.Parameter(c) for c in t.cores])
            #bias
            bias = tn.zeros(size_out, dtype = dtype) 
            self.bias = nn.Parameter(bias)
        else:
            raise InvalidArguments('Initializer not defined. Possible choices are \'He\' and \'Glo\'.')

    @tn.jit.export 
    def forward(self, x):
        """
        Computes the output of the layer for the given input. 
        
        Supports trailing dimensiond broadcasting. If the input of the layer is set to ``[M1,...,Md]`` and a tensor od shape ``[...,M1,...,Md]`` is provided then the multiplication is performed along the last d dimensions.

        Args:
            x (torch.tensor): input of the layer.

        Returns:
            torch.tensor: output of the layer.
        """
        
        # return dense_matvec(self.cores,x) + self.bias
        
        result = tn.unsqueeze(x,-1)

        d = len(self.size_in)
        D = len(x.shape)

        for c in self.cores:
            result = tn.tensordot(result,c,([D-d,-1],[2,0]))
        result = tn.squeeze(result,-1)

        return result+self.bias


class CompressedTTLayer(nn.Module):
    """
    A Tensor Train (TT) neural network layer that operates directly on TT objects and applies nonlinear activation between TT cores during multiplication.
    
    This layer is inspired by Nonlinear Tensor Train formats for deep neural networks. Instead of computing the full dense tensor, it performs a fast matrix-vector-like multiplication of the layer's TTM weights with the input TT object. The layer natively applies a nonlinear activation and an optional bias to the intermediate core representations during the contraction sweep. The intermediate representations are orthogonalized and truncated to maintain the compression, ensuring the output TT object's ranks are strictly bounded by ``R_output``.
    """
    def __init__(self, N_in, N_out, R_layer, R_output, activation=tn.relu, bias=True, dtype=tn.float32):
        """
        Initialize the CompressedTTLayer.
        
        Args:
            N_in (list[int]): Mode sizes for the input dimensions.
            N_out (list[int]): Mode sizes for the output dimensions.
            R_layer (list[int]): TT ranks of the layer's TTM weight operator (length ``len(N_in)+1``).
            R_output (list[int]): TT ranks of the output tensor (length ``len(N_out)+1``). Must satisfy ``R_output[0] == R_output[-1] == 1``. Each entry ``R_output[k]`` bounds the rank at the k-th bond of the output TT, giving per-bond control over the compression.
            activation (callable, optional): The nonlinear activation function to apply between cores during multiplication. Defaults to ``torch.relu``.
            bias (bool, optional): If True, a trainable per-mode bias is added to the intermediate core representations. Defaults to True.
            dtype (torch.dtype, optional): Data type for the layer's weights. Defaults to ``torch.float32``.
        
        Raises:
            InvalidArguments: If ``R_output`` does not have the correct length or boundary conditions.
        """
        super().__init__()
        d = len(N_in)
        if len(R_output) != d + 1:
            raise InvalidArguments(f"R_output must have length {d+1} (len(N_in)+1), got {len(R_output)}.")
        if R_output[0] != 1 or R_output[-1] != 1:
            raise InvalidArguments("R_output[0] and R_output[-1] must be 1.")
        
        self.size_in = N_in
        self.size_out = N_out
        self.rank = R_layer
        self.R_output = list(R_output)
        self.activation = activation
        self.dtype = dtype

        t = torchtt.randn([(s2, s1) for s1, s2 in zip(N_in, N_out)], R_layer, dtype=dtype, var=2/tn.prod(tn.tensor([s1 for s1 in N_in])))
        self.cores = nn.ParameterList([nn.Parameter(c) for c in t.cores])
        
        if bias:
            b_cores = []
            for i in range(len(N_out)):
                b_cores.append(nn.Parameter(tn.zeros(1, N_out[i], 1, dtype=dtype)))
            self.bias = nn.ParameterList(b_cores)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        """
        Forward pass for the CompressedTTLayer.
        
        Computes the operation by multiplying the input TT with the layer's TTM using a right-to-left sweep. 
        During this sweep, the intermediate bias is added and the activation is applied to the core before 
        the core is orthogonalized and truncated (using SVD) to strictly enforce the per-bond rank limits 
        given by ``R_output``.

        Args:
            x (torchtt.TT): The input Tensor Train object.
            
        Returns:
            torchtt.TT: The output Tensor Train object representing the nonlinearly compressed forward pass, with ranks bounded by ``R_output``.
        """
        d = len(self.size_in)
        from ._fast_mult import swap_cores
        
        # The sweep processes cores right-to-left, so the internal "cores" list
        # is reversed: cores[j] in the sweep corresponds to bond (d-1-j) in the
        # original ordering. We need to map swap_cores calls to the correct
        # R_output entry for each bond.
        # After reversal, bond between sweep-cores[j] and sweep-cores[j+1]
        # corresponds to original bond index (d-1-j).
        
        cores = [tn.permute(c, [2, 1, 0]) for c in x.cores[::-1]]
        for i in range(d):
            cores[0] = oe.contract("mabk,kbn->man", self.cores[d-i-1], cores[0])
            
            if self.bias is not None:
                cores[0] = cores[0] + self.bias[d-i-1]
                
            if self.activation is not None:
                cores[0] = self.activation(cores[0])
                
            if i != d-1:
                for j in range(i, -1, -1):
                    # Bond between sweep-cores[j] and sweep-cores[j+1] maps to
                    # original bond index (d-1-j).
                    bond_idx = d - 1 - j
                    cores[j], cores[j+1] = swap_cores(cores[j], cores[j+1], 0.0, self.R_output[bond_idx])
                    
        cores = [tn.permute(c, [2, 1, 0]) for c in cores[::-1]]
        
        return torchtt.TT(cores)