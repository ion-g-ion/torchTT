"""
Test the multilinear solvers.
"""
import unittest
import torchtt 
import torch as tn
import numpy as np

err_rel = lambda t, ref :  tn.linalg.norm(t-ref).numpy() / tn.linalg.norm(ref).numpy() if ref.shape == t.shape else np.inf
   

            
class TestSolvers(unittest.TestCase):

    basic_dtype = tn.complex128
    
    @unittest.skipUnless(torchtt.solvers.cpp_enabled(), "C++ extension must be present.")
    def test_amen_solve(self):
        """
        Test the AMEN solve on a small example.
        """
        A = torchtt.random([(4,4),(5,5),(6,6)],[1,2,3,1], dtype = tn.float64) 
        x = torchtt.random([4,5,6],[1,2,3,1], dtype = tn.float64) 
        b = A @ x 
        xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner=None, use_cpp=True) 
        err = (A@xx-b).norm()/b.norm() # error residual
        self.assertLess(err.numpy(),5*1e-8,"AMEN solve failed.")


    @unittest.skipUnless(torchtt.solvers.cpp_enabled(), "C++ extension must be present.")
    def test_amen_solve_cprec(self):
        """
        Test AMEN with central Jacobi preconditioner.
        """
        A = torchtt.random([(4,4),(5,5),(6,6)],[1,2,3,1], dtype = tn.float64) 
        x = torchtt.random([4,5,6],[1,2,3,1], dtype = tn.float64) 
        b = (A @ x).round(1e-16) 
        
        xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner='c', use_cpp=True) 
        err = (A@xx-b).norm()/b.norm() # error residual
        self.assertLess(err.numpy(),5*1e-8,"AMEN solve failed (c preconditioner).")

    @unittest.skipUnless(torchtt.solvers.cpp_enabled(), "C++ extension must be present.")
    def test_amen_solve_rprec(self):
        """
        Test AMEN with the right Jacobi reconditioner.
        """
        A = torchtt.random([(4,4),(5,5),(6,6)],[1,2,3,1], dtype = tn.float64) 
        x = torchtt.random([4,5,6],[1,2,3,1], dtype = tn.float64) 
        b = A @ x 
        xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner='r', use_cpp=True) 
        err = (A@xx-b).norm()/b.norm() # error residual
        self.assertLess(err.numpy(),5*1e-8,"AMEN solve failed (right preconditioner).") 
    
    def test_amen_solve_cprec_nocpp(self):
        """
        Test AMEN with central Jacobi preconditioner without the C++.
        """
        A = torchtt.random([(4,4),(5,5),(6,6)],[1,2,3,1], dtype = tn.float64) 
        x = torchtt.random([4,5,6],[1,2,3,1], dtype = tn.float64) 
        b = A @ x 
        
        xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner='c', use_cpp=False) 
        err = (A@xx-b).norm()/b.norm() # error residual
        self.assertLess(err.numpy(),5*1e-8,"AMEN solve failed (c preconditioner, without C++).")
        
    def test_amen_solve_rprec_nocpp(self):
        """
        Test AMEN with the right Jacobi reconditioner without the C++.
        """
        A = torchtt.random([(4,4),(5,5),(6,6)],[1,2,3,1], dtype = tn.float64) 
        x = torchtt.random([4,5,6],[1,2,3,1], dtype = tn.float64) 
        b = A @ x 
        xx = torchtt.solvers.amen_solve(A, b, verbose = False, eps=1e-10, nswp = 40, preconditioner='r', use_cpp=False) 
        err = (A@xx-b).norm()/b.norm() # error residual
        self.assertLess(err.numpy(),5*1e-8,"AMEN solve failed (right preconditioner).")
        
    def test_amen_solve_nocpp(self):
        """
        Test the AMEN solve on a small example (disable C++).
        """
        A = torchtt.random([(4,4),(5,5),(6,6)],[1,2,3,1], dtype = tn.float64) 
        x = torchtt.random([4,5,6],[1,2,3,1], dtype = tn.float64) 
        b = A @ x 
        xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner=None, use_cpp = False) 
        err = (A@xx-b).norm()/b.norm() # error residual
        self.assertLess(err.numpy(),5*1e-8,"AMEN solve failed.")

        xx = torchtt.solvers.amen_solve(A,b,verbose = False, eps=1e-10, preconditioner='c', use_cpp = False) 
        err = (A@xx-b).norm()/b.norm() # error residual
        self.assertLess(err.numpy(),5*1e-8,"AMEN solve failed (c preconditioner).")

if __name__ == '__main__':
    unittest.main()


