"""
Implements a basic TT layer for constructing deep TT networks.

"""
import torch as tn
import torch.nn as nn
import torch.optim 
import torchtt 
from ._aux_ops import dense_matvec
from .errors import *
import math

class RiemannianAdam(torch.optim.Optimizer):
    r"""Implements Riemannian Adam (and AMSGrad) on a manifold, working entirely in real space.
    
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        manifold: a manifold object supporting the methods:
            - egrad_to_rgrad(point, egrad)
            - inner(point, vec1, vec2)
            - retraction_transport(point, tangent, search_dir)
        lr (float): learning rate (default: 0.05)
        betas (Tuple[float, float]): coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float): term added to denominator for numerical stability (default: 1e-8).
        amsgrad (bool): whether to use the AMSGrad variant (default: False).
    """
    def __init__(self, params, manifold, lr=0.05, betas=(0.9, 0.999), eps=1e-8, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, amsgrad=amsgrad)
        super(RiemannianAdam, self).__init__(params, defaults)
        self.manifold = manifold
        self._step = 0

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
            
        self._step += 1
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            amsgrad = group['amsgrad']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    if amsgrad:
                        state['v_hat'] = torch.zeros_like(p.data)
                
                # Retrieve state variables 
                momentum = state['momentum']
                v = state['v']
                
                # Compute Riemannian gradient 
                rgrad = self.manifold.egrad_to_rgrad(p.data, grad)
                
                # Update first moment estimate
                momentum = beta1 * momentum + (1 - beta1) * rgrad
                # Update second moment estimate using the manifold inner product
                v = beta2 * v + (1 - beta2) * self.manifold.inner(p.data, rgrad, rgrad)
                
                if amsgrad:
                    v_hat = state['v_hat']
                    # Update v_hat as elementwise maximum
                    v_hat = torch.max(v_hat, v)
                
                # Bias correction
                lr_corr = lr * math.sqrt(1 - beta2 ** self._step) / (1 - beta1 ** self._step)
                
                # Compute search direction using the (possibly AMSGrad-corrected) second moment
                if amsgrad:
                    denom = torch.sqrt(v_hat) + eps
                else:
                    denom = torch.sqrt(v) + eps
                search_dir = - lr_corr * momentum / denom
                
                # Compute new parameter value and transported momentum via retraction/transport
                new_p, new_momentum = self.manifold.retraction_transport(p.data, momentum, search_dir)
                
                # Update parameter and state
                p.data.copy_(new_p)
                state['momentum'] = new_momentum
                state['v'] = v
                if amsgrad:
                    state['v_hat'] = v_hat
                    
        return loss
    
class TTParameter(tn.nn.Parameter):
    def __new__(cls, tt_object, requires_grad=True):
       
        param = super().__new__(cls, tn.concatenate([c.flatten() for c in tt_object.cores]), requires_grad)

        param.M = tt_object.M if tt_object.is_ttm else None 
        param.N = tt_object.N 
        param.R = tt_object.R
        param.idx_core = []
        s = 0
        for c in tt_object.cores:
            param.idx_core.append(s)
            s += tn.numel(c)
        param.idx_core.append(s)
            
        return param

    def to_tt(self):
        pass
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
            #self.cores = nn.ModuleDict({"ttm": nn.ParameterList([nn.Parameter(c) for c in t.cores])})
            self.ttm = TTParameter(t)
            #bias
            bias = tn.zeros(size_out, dtype = dtype) 
            self.bias = nn.Parameter(bias)
        elif initializer=='Glo':
            t = torchtt.randn([(s2,s1) for s1,s2 in zip(size_in,size_out)], rank, dtype=dtype, var = 1/(tn.prod(tn.tensor([s1 for s1 in size_in]))+tn.prod(tn.tensor([s1 for s1 in size_out]))) )
            #self.cores = [nn.Parameter(tn.Tensor(c.clone())) for c in t.cores] 
            #self.cores = nn.ModuleDict({"ttm": nn.ParameterList([nn.Parameter(c) for c in t.cores])})
            self.ttm = TTParameter(t)
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

        for k in range(len(self.ttm.N)):
            c = self.ttm.data[self.ttm.idx_core[k]:self.ttm.idx_core[k+1]].view([self.ttm.R[k], self.ttm.M[k], self.ttm.N[k], self.ttm.R[k+1]])
            result = tn.tensordot(result,c,([D-d,-1],[2,0]))
        #for c in self.cores:
        #    result = tn.tensordot(result,c,([D-d,-1],[2,0]))
        result = tn.squeeze(result,-1)

        return result+self.bias


 
