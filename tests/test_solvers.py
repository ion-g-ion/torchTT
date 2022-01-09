"""
Test the basic multilinear algebra operations between torchtt.TT objects.
"""
import unittest
import torchtt 
import torch as tn
import numpy as np

err_rel = lambda t, ref :  tn.linalg.norm(t-ref).numpy() / tn.linalg.norm(ref).numpy() if ref.shape == t.shape else np.inf
   
            
class TestSolvers(unittest.TestCase):

    def test_amen_solve(self):
        """
        Thest the AMEN solve on a small example.
        """
        A = torchtt.random([(4,4),(5,5),(6,6)],[1,2,3,1]) 
        x = torchtt.random([4,5,6],[1,2,3,1]) 
        b = A @ x 
        xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner='c') 
        err = (A@xx-b).norm()/b.norm() # error residual
        self.assertLess(err.numpy(),5*1e-10,"AMEN solve failed.")

if __name__ == '__main__':
    unittest.main()

