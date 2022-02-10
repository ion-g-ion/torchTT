"""
Implements a basic TT layer for constructing deep TT networks.

"""
import torch as tn
import torch.nn as nn
import torchtt
from ._aux_ops import dense_matvec
from .errors import *
class LinearLayerTT(nn.Module):
    """
    Basic class for TT layers. See [Tensorizing Neural Networks](https://arxiv.org/abs/1509.06569) for a detailed description.
    It can be used similarily to any layer from `torch.nn`.
    The output of the layer is \(\\mathcal{LTT}(\\mathsf{x}) =\\mathsf{Wx}+\\mathsf{b}\), where the tensor operator \(\\mathsf{W}\) is represented in the TT format (with a fixed prescribed rank).

    """
    def __init__(self, size_in, size_out, rank, dtype = tn.float32, initializer = 'He'):
        """
        Constructor of the TT layer class. 
        
        Possible initializers are:
        
         * `'He'` for He Normal (He-et-al) initialization.
         * `'Glo'` for Glorot initialization.
            
        Args:
            size_in (list[int]): the size of the input tensor.
            size_out (list[int]): the size of the output tensor.
            rank (list[int]): the rank of the tensor operator.
            dtype (torch.dtype, optional): the dtype of the layer. Defaults to tn.float32.
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


    def forward(self, x):
        """
        Computes the output of the layer for the given input. 
        
        Supports trailing dimensiond broadcasting. If the input of the layer is set to `[M1,...,Md]` and a tensor od shape `[...,M1,...,Md]` is provided then the multiplication is performed along the last d dimensions.

        Args:
            x (torch.tensor): input of the layer.

        Returns:
            torch.tensor: output of the layer.
        """
        
        return dense_matvec(self.cores,x) + self.bias


 
