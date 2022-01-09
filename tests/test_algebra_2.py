"""
Test the advanced multilinear algebra operations between torchtt.TT objects.
Some operations (matvec for large ranks and elemntwise division) can be only computed using optimization (AMEN and DMRG).
"""
import unittest
import torchtt as tntt
import torch as tn
import numpy as np

err_rel = lambda t, ref :  tn.linalg.norm(t-ref).numpy() / tn.linalg.norm(ref).numpy() if ref.shape == t.shape else np.inf
   
            
class TestLinalgAdvanced(unittest.TestCase):
    
    def test_dmrg_matvec(self):
        """
        Test the fast matrix vector product using DMRG iterations.
        """
        n = 32
        A = tntt.random([(n,n)]*8,[1]+7*[3]+[1])
        A = A + A 
        
        x = tntt.random([n]*8,[1]+7*[5]+[1])
        x = x + x
        x = x + x
 
        # conventional method 
        y = (A @ x).round(1e-12)

        # dmrg matvec
        yf = A.fast_matvec(x)

        rel_error = (y-yf).norm().numpy()/y.norm().numpy()
        
        self.assertLess(rel_error,1e-12,"DMRG matrix vector problem.")
      
    def test_amen_division(self):
        """
        Test the division between tensors performed with AMEN optimization.
        """
        N = [7,8,9,10]
        xs = tntt.meshgrid([tn.linspace(0,1,n, dtype = tn.float64) for n in N])
        x = xs[0]+xs[1]+xs[2]+xs[3]+xs[1]*xs[2]+(1-xs[3])*xs[2]+1
        x = x.round(0)
        y = tntt.ones(x.N)
        
        a = y/x
        b = 1/x
        c = tn.tensor(1.0)/x
        
        self.assertLess(err_rel(a.full(),y.full()/x.full()),1e-11,"AMEN division problem: TT and TT.")
        self.assertLess(err_rel(b.full(),1/x.full()),1e-11,"AMEN division problem: scalar and TT.")
        self.assertLess(err_rel(c.full(),1/x.full()),1e-11,"AMEN division problem: scalar and TT part 2.")

if __name__ == '__main__':
    unittest.main()