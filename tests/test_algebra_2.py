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
    
    basic_dtype = tn.float64
    
    def test_dmrg_matvec(self):
        """
        Test the fast matrix vector product using DMRG iterations.
        """
        n = 32
        A = tntt.random([(n,n)]*8,[1]+7*[3]+[1], dtype = tn.complex128)
        Am = A + A 
        
        x = tntt.random([n]*8,[1]+7*[5]+[1], dtype = tn.complex128)
        xm = x + x
        xm = xm + xm
 
        # conventional method 
        y = 8 * (A @ x).round(1e-12)

        # dmrg matvec
        yf = Am.fast_matvec(xm)

        rel_error = (y-yf).norm().numpy()/y.norm().numpy()
        
        self.assertLess(rel_error,1e-12,"DMRG matrix vector problem: square matrix.")
        
        n = 32
        A = tntt.random([(n+2,n)]*8,[1]+7*[3]+[1], dtype = tn.complex128)
        Am = A + A 
        
        x = tntt.random([n]*8,[1]+7*[5]+[1], dtype = tn.complex128)
        xm = x + x
        xm = xm + xm
 
        # conventional method 
        y = 8 * (A @ x).round(1e-12)

        # dmrg matvec
        yf = Am.fast_matvec(xm)

        rel_error = (y-yf).norm().numpy()/y.norm().numpy()
        
        self.assertLess(rel_error,1e-12,"DMRG matrix vector problem: not square matrix.")
      
    def test_amen_division(self):
        """
        Test the division between tensors performed with AMEN optimization.
        """
        N = [7,8,9,10]
        xs = tntt.meshgrid([tn.linspace(0,1,n, dtype = self.basic_dtype) for n in N])
        x = xs[0]+xs[1]+xs[2]+xs[3]+xs[1]*xs[2]+(1-xs[3])*xs[2]+1
        x = x.round(0)
        y = tntt.ones(x.N, dtype = self.basic_dtype)
        
        a = y/x
        b = 1/x
        c = tn.tensor(1.0)/x
        
        self.assertLess(err_rel(a.full(),y.full()/x.full()),1e-11,"AMEN division problem: TT and TT.")
        self.assertLess(err_rel(b.full(),1/x.full()),1e-11,"AMEN division problem: scalar and TT.")
        self.assertLess(err_rel(c.full(),1/x.full()),1e-11,"AMEN division problem: scalar and TT part 2.")
        
    def test_amen_division_preconditioned(self):
        """
        Test the elemntwise division using AMEN (use preconditioner for the local subsystem).
        """
        N = [7,8,9,10]
        xs = tntt.meshgrid([tn.linspace(0,1,n, dtype = self.basic_dtype) for n in N])
        x = xs[0]+xs[1]+xs[2]+xs[3]+xs[1]*xs[2]+(1-xs[3])*xs[2]+1
        x = x.round(0)
        y = tntt.ones(x.N)
        
        a = tntt.elementwise_divide(y,x,preconditioner = 'c')
       
        
        self.assertLess(err_rel(a.full(),y.full()/x.full()),1e-11,"AMEN division problem (preconditioner): TT and TT.")
        
        
if __name__ == '__main__':
    unittest.main()