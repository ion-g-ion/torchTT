"""
Implements a basic TT layer for constructing deep TT networks.

Todo: 

 * finish it
    
"""
import torch as tn
import torch.nn as nn
import torchtt
from ._aux_ops import dense_matvec

class LinearLayerTT(nn.Module):
    """
    Implements a basic Tt linear layer.

    Args:
        size_in (list[int]): the input shape of the layer.shape
        size_out (list[int]): the output of the layer. Has to be of equal length with `size_in`.
        rank (list[int]): the rank of the TT layer. Has to be `len(rank)==len(size_in)+1`.
    """
    def __init__(self, size_in, size_out, rank, dtype = tn.float32):
        super().__init__()
        self.size_in, self.size_out, self.rank = size_in, size_out, rank
        t = torchtt.randn([(s2,s1) for s1,s2 in zip(size_in,size_out)], rank, dtype=dtype)
        #self.cores = [nn.Parameter(tn.Tensor(c.clone())) for c in t.cores] 
        self.cores = nn.ParameterList([nn.Parameter(c) for c in t.cores])
        #bias
        bias = tn.zeros(size_out, dtype = dtype) 
        self.bias = nn.Parameter(bias)



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


 
