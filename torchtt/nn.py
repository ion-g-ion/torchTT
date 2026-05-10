"""
Implements a basic TT layer for constructing deep TT networks.

"""
import torch as tn
import torch.nn as nn
import torchtt 
import opt_einsum as oe
from ._aux_ops import dense_matvec
from .errors import *

class TTDensityLayer(nn.Module):
    
    def __init__(self, N, R, basis, linear_transformation = False, dtype = tn.float32):
        """
        Initialize a TT Density Layer.
        
        Args:
            N (list[int]): mode sizes for each dimension.
            R (list[int]): TT ranks (must have R[0] = R[-1] = 1).
            basis (list[BaseBasis]): list of BaseBasis objects for each dimension.
            linear_transformation (bool, optional): if True, input is a linear transformation. Defaults to False.
            dtype (torch.dtype, optional): data type for the layer. Defaults to torch.float32.
        """
        super().__init__()
        
        if R[0] != 1 or R[-1] != 1 and len(R) != len(N)+1:
            raise InvalidArguments("The rank and the number of modes do not match.")
        if len(basis) != len(N):
            raise InvalidArguments("The number of bases must match the number of modes.")
        if len(N) < 1:
            raise InvalidArguments("The dimension of the tensor must be at least 1.")
        
        # Verify that N matches the basis dimensions
        for i, (n, b) in enumerate(zip(N, basis)):
            if n != b.n:
                raise InvalidArguments(f"Mode size N[{i}]={n} does not match basis dimension {b.n}")
        
        self.dim = len(N)
        self.N = N
        self.R = R
        self.linear_transformation = linear_transformation
        self.dtype = dtype
        
        # basis is a list of BaseBasis objects - get integration weights from them
        self.basis = nn.ModuleList(basis)
        
        # Register integration weights as buffers
        for i, b in enumerate(basis):
            self.register_buffer(f'integration_weight_{i}', b.integration_weights().to(self.dtype))

    @property
    def integration_weights(self):
        return [getattr(self, f'integration_weight_{i}') for i in range(self.dim)]
        
    @staticmethod
    def input_requireemnt(N, R, linear_transformation = False):
        """
        Computes the number of input features required for the given dimensions and ranks.

        Args:
            N (list[int]): mode size every domension.
            R (list[int]): the rank of the TT decomposition.
            linear_transformation (bool, optional): if True, the input is a linear transformation of the input. Defaults to False.

        Returns:
            int: the number of input features required.
        """
        n_cores = sum([N[i]*R[i]*R[i+1] for i in range(len(N))])
        if not linear_transformation:
            return n_cores
        else:
            dim = len(N)
            # Cores + Angles + Scales + Offsets
            return n_cores + dim * (dim - 1) // 2 + 2 * dim
        
    def marginalize(self, tts, x, keep_indices):
         
        pass
    
    def get_tt(self, tts):
        """
        Extract TT objects from flattened input.
        
        Args:
            tts (torch.Tensor): flattened TT cores of shape (..., total_params) where
                total_params = sum(N[i] * R[i] * R[i+1] for all i)
                
        Returns:
            list[torchtt.TT]: list of TT objects, one for each sample in the batch
        """
        if not tts.shape[-1] == sum([self.N[i]*self.R[i]*self.R[i+1] for i in range(len(self.N))]):
            raise InvalidArguments("The shape of the tensor does not match the number of modes and the ranks.")
        
        # Flatten all batch dimensions
        batch_shape = tts.shape[:-1]
        tts_flat = tts.reshape(-1, tts.shape[-1])
        
        tt_list = []
        for batch_idx in range(tts_flat.shape[0]):
            # Extract cores for this sample
            cores = []
            sofar = 0
            for i in range(len(self.N)):
                core_size = self.N[i] * self.R[i] * self.R[i+1]
                core_flat = tts_flat[batch_idx, sofar:sofar+core_size]
                core = core_flat.view(self.R[i], self.N[i], self.R[i+1])
                cores.append(core)
                sofar += core_size
            
            # Create TT object from cores
            tt_obj = torchtt.TT(cores)
            tt_list.append(tt_obj)
        
        # If input had batch dimensions, reshape the list accordingly
        if len(batch_shape) > 0:
            # Return nested list structure matching batch dimensions
            # For simplicity, return flat list - user can reshape if needed
            pass
        
        return tt_list
    
     
    def _create_rotation_matrix(self, angles, dim, device, dtype):
        batch_shape = angles.shape[:-1]
        # Start with identity
        R = tn.eye(dim, device=device, dtype=dtype)
        # Expand to batch
        view_shape = [1] * len(batch_shape) + [dim, dim]
        R = R.view(*view_shape).repeat(*batch_shape, 1, 1)
        
        k = 0
        for i in range(dim):
            for j in range(i + 1, dim):
                theta = angles[..., k] 
                c = tn.cos(theta)
                s = tn.sin(theta)
                
                # Expand for broadcasting
                c = c.unsqueeze(-1)
                s = s.unsqueeze(-1)
                
                R_i = R[..., :, i].clone()
                R_j = R[..., :, j].clone()
                
                # Update columns
                R[..., :, i] = c * R_i + s * R_j
                R[..., :, j] = -s * R_i + c * R_j
                
                k += 1
        return R

    def forward(self, tts, x):
        """
        Forward pass: evaluate the TT density at input points x.
        
        The evaluation uses the basis functions to compute:
        - Numerator: product of basis evaluations contracted with TT cores
        - Denominator: product of integration weights contracted with TT cores (normalization)
        - Output: normalized density value at each input point
        
        Args:
            tts (torch.Tensor): flattened TT parameters of shape (..., total_params)
            x (torch.Tensor): input points of shape (..., dim)
            
        Returns:
            torch.Tensor: density values at input points, shape (...)
        """
        if not isinstance(x, tn.Tensor):
            x = tn.tensor(x, dtype=self.dtype, device=tts.device)
        else:
             x = x.to(dtype=self.dtype)
             
        if not tts.shape[-1] == self.input_requireemnt(self.N, self.R, self.linear_transformation):
            raise InvalidArguments("The shape of the tensor does not match the number of modes and the ranks.")
        if self.dim != x.shape[-1]:
            raise InvalidArguments("The dimension of the tensor does not match the dimension of the input.")
        
        # Extract cores from flattened input
        sofar = 0
        cores = []
        for i in range(len(self.N)):
            core_size = self.N[i]*self.R[i]*self.R[i+1]
            # Reshape: (..., R[i], N[i], R[i+1])
            cores.append(tts[..., sofar:sofar+core_size].view(*tts.shape[:-1], self.R[i], self.N[i], self.R[i+1]).to(self.dtype))
            sofar += core_size
        
        det_jac = 1.0
        if self.linear_transformation:
            # Angles
            n_angles = self.dim * (self.dim - 1) // 2
            angles = tts[..., sofar:sofar+n_angles]
            sofar += n_angles
            
            # Scales
            n_scales = self.dim
            scales = tn.exp(tts[..., sofar:sofar+n_scales])
            sofar += n_scales
            
            # Offsets
            n_offsets = self.dim
            offsets = tts[..., sofar:sofar+n_offsets]
            sofar += n_offsets
            
            # Create A = R * S
            R = self._create_rotation_matrix(angles, self.dim, tts.device, tts.dtype)
            
            # Apply transformation z = R * S * x + b
            x = x * scales
            x = tn.einsum('...ij,...j->...i', R, x)
            x = x + offsets
            
            # Jacobian det = prod(scales)
            det_jac = tn.prod(scales, dim=-1)
            
        # Evaluate basis functions at input points
        # Bevals[i] has shape (N[i], ...) where ... is the shape of x[..., i]
        Bevals = [self.basis[i](x[..., i]).to(self.dtype) for i in range(self.dim)]
        
        # Compute denominator (normalization): contract with integration weights
        # Start with first mode: integrate over mode 0
        # "n,...nb,...nd->...bd" means: sum over n (mode index), contract cores
        denominator = oe.contract("n,...nb,...nd->...bd", self.integration_weights[0], cores[0][...,0,:,:], cores[0][...,0,:,:])
        
        for i in range(1, self.dim):
            # Continue contracting: "...ab,n,...anc,...bne->...ce"
            denominator = oe.contract("...ab,n,...anc,...bne->...ce", denominator, self.integration_weights[i], cores[i], cores[i])
        
        # Compute numerator: contract with basis evaluations at x
        # Bevals[0] has shape (N[0], ...) and cores[0] has shape (..., R[0], N[0], R[1])
        # We need to align dimensions properly
        nominator = oe.contract("n...,...nb,...nd->...bd", Bevals[0], cores[0][...,0,:,:], cores[0][...,0,:,:])
        
        for i in range(1, self.dim):
            nominator = oe.contract("...ab,n...,...anc,...bne->...ce", nominator, Bevals[i], cores[i], cores[i])
        
        # Extract final values (squeeze last dimensions which should be 1x1)
        pdf_eval = nominator[..., 0, 0] / denominator[..., 0, 0]
        
        if self.linear_transformation:
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
    
    This layer is inspired by Nonlinear Tensor Train formats for deep neural networks. Instead of computing the full dense tensor, it performs a fast matrix-vector-like multiplication of the layer's TTM weights with the input TT object. The layer natively applies a nonlinear activation and an optional bias to the intermediate core representations during the contraction sweep. The intermediate representations are orthogonalized and truncated to maintain the compression, ensuring the output TT object's rank is strictly bounded by `rmax`.
    """
    def __init__(self, N_in, N_out, R_layer, rmax, activation=tn.relu, bias=True, dtype=tn.float32):
        """
        Initialize the CompressedTTLayer.
        
        Args:
            N_in (list[int]): Mode sizes for the input dimensions.
            N_out (list[int]): Mode sizes for the output dimensions.
            R_layer (list[int]): TT ranks of the layer's TTM weight operator.
            rmax (int): The maximum TT rank allowed for the output TT object. The forward pass guarantees the output ranks will not exceed this value.
            activation (callable, optional): The nonlinear activation function to apply between cores during multiplication. Defaults to `torch.relu`.
            bias (bool, optional): If True, a trainable per-mode bias is added to the intermediate core representations. Defaults to True.
            dtype (torch.dtype, optional): Data type for the layer's weights. Defaults to torch.float32.
        """
        super().__init__()
        self.size_in = N_in
        self.size_out = N_out
        self.rank = R_layer
        self.rmax = rmax
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
        the core is orthogonalized and truncated (using SVD) to strictly enforce the `rmax` limit.

        Args:
            x (torchtt.TT): The input Tensor Train object.
            
        Returns:
            torchtt.TT: The output Tensor Train object representing the nonlinearly compressed forward pass, with maximum rank strictly bounded by `rmax`.
        """
        d = len(self.size_in)
        from ._fast_mult import swap_cores
        
        cores = [tn.permute(c, [2, 1, 0]) for c in x.cores[::-1]]
        for i in range(d):
            cores[0] = oe.contract("mabk,kbn->man", self.cores[d-i-1], cores[0])
            
            if self.bias is not None:
                cores[0] = cores[0] + self.bias[d-i-1]
                
            if self.activation is not None:
                cores[0] = self.activation(cores[0])
                
            if i != d-1:
                for j in range(i, -1, -1):
                    cores[j], cores[j+1] = swap_cores(cores[j], cores[j+1], 0.0, self.rmax)
                    
        cores = [tn.permute(c, [2, 1, 0]) for c in cores[::-1]]
        
        return torchtt.TT(cores)