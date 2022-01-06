import torch as tn
import torch.nn as nn
from torchtt import TT, randn

class LinearLayerTT(nn.Module):
    """ Tensor Train layer """
    def __init__(self, size_in, size_out, rank, dtype = tn.float32):
        super().__init__()
        self.size_in, self.size_out, self.rank = size_in, size_out, rank
        self.cores = [None] * len(size_in)
        t = randn([(s2,s1) for s1,s2 in zip(size_in,size_out)], rank, dtype=dtype)
        for i in range(len(size_in)):
            core = t.cores[i][:] 
            self.cores[i] = nn.Parameter(core) 
        #bias
        bias = tn.zeros(size_out, dtype = dtype) 
        self.bias = nn.Parameter(bias)



    def forward(self, x):
        
        W = TT(self.cores)
        return W @ x + self.bias


 
