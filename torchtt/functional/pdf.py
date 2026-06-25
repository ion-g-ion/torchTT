
import torchtt
from torchtt.functional import BaseBasis
import torch
from typing import Union, List
import torchtt

class TTPDF:

    def __init__(self, basis: list[BaseBasis], coeffs: torchtt.TT):

        self._basis = basis
        self._coeffs = coeffs
        self._n = coeffs.N
        self._is_normalized = False
        self._ws =  [b.integration_weights().to(device=coeffs.device, dtype=coeffs.dtype) for b in self._basis]
        self._Bint = 
        self.normalize()


    @property
    def n(self) -> int:
        return self._n

    @property
    def coeffs(self) -> TT:
        return self._coeffs

    @property
    def basis(self) -> list[BaseBasis]:
        return self._basis

    @property
    def conditional_dims(self) -> list[int]:
        return self._conditional_dims

    def normalize(self):

        if self._is_normalized:
            return 

        norm = torchtt.dot(self._coeffs, torchtt.rank1TT(self._ws))

        self._coeffs = self._coeffs / norm

        self._is_normalized = True    

    
    def marginalize(self, dims: list[int]) -> TTPDF:
        pass

    def __call__(self, x: Union[torch.Tensor, list[torch.Tensor]], derivative: bool = False) -> torch.Tensor:
        pass

    def mutual_info(self, dim1: int, dim2: int) -> float:
        """
        Compute the mutual information between two dimensions of the probability distribution.
        """
        if not self._is_normalized:
            self.normalize()

    def 
        
    def __pow__(self, other: TTPDF) -> TTPDF:

        pass 

    def to(self, device: torch.device = None, dtype: torch.dtype = None) -> TTPDF:
        
        new_coeffs = self._coeffs.to(device, dtype)
        return TTPDF(self._basis, new_coeffs)
        


    @staticmethod
    def uniform(lower_bounds: list[float], upper_bounds: list[float], N: list[int], degs: list[int] ) -> TTPDF:

        
        pass

    @staticmethod
    def normal(means: list[float], stds: list[float], N: list[int], degs: list[int] ) -> TTPDF:
        pass
        
        
        