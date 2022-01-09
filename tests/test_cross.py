"""
Test the cross approximation method.
"""
import unittest
import torchtt as tntt
import torch as tn
import numpy as np

err_rel = lambda t, ref :  tn.linalg.norm(t-ref).numpy() / tn.linalg.norm(ref).numpy() if ref.shape == t.shape else np.inf


class TestLinalgAdvanced(unittest.TestCase):
    
    def test_dmrg_corss_interpolation(self):
        """
        Test the DMRG cross interpolation method.
        """
        func1 = lambda I: 1/(2+tn.sum(I+1,1).to(dtype=tn.float64))
        N = [20]*4
        x = tntt.interpolate.dmrg_cross(func1, N, eps = 1e-7)
        Is = tntt.meshgrid([tn.arange(0,n,dtype=tn.float64) for n in N])
        x_ref = 1/(2+Is[0].full()+Is[1].full()+Is[2].full()+Is[3].full()+4)
       
        self.assertLess(err_rel(x.full(),x_ref),1e-6,"Error dmrg cross interpolation") 
    
    def test_function_interpolate_multivariable(self):   
        """
        Test the DMRG cross interpolation method for funcxtion approximation.
        """
        func1 = lambda I: 1/(2+tn.sum(I+1,1).to(dtype=tn.float64))
        N = [20]*4
        
        Is = tntt.meshgrid([tn.arange(0,n,dtype=tn.float64) for n in N])
        x_ref = 1/(2+Is[0].full()+Is[1].full()+Is[2].full()+Is[3].full()+4)
       
        y = tntt.interpolate.function_interpolate(func1, Is, 1e-8)
        self.assertLess(err_rel(y.full(),x_ref),1e-7,"Error dmrg cross interpolation") 
        
    def test_function_interpolate_univariate(self):   
        """
        Test the DMRG cross interpolation method for funcxtion approximation.
        """
        N = [20]*4
        Is = tntt.meshgrid([tn.arange(0,n,dtype=tn.float64) for n in N])
        x_ref = 1/(2+Is[0].full()+Is[1].full()+Is[2].full()+Is[3].full()+4)
        x = tntt.TT(x_ref)
        
       
        
        y = tntt.interpolate.function_interpolate(lambda x : tn.log(x), x, eps = 1e-7)
        
       
        self.assertLess(err_rel(y.full(),tn.log(x_ref)),1e-6,"Error dmrg cross interpolation") 