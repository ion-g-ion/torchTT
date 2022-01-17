"""
Test the basic multilinear algebra operations between torchtt.TT objects.
"""
import unittest
import torchtt as tntt
import torch as tn
import numpy as np

err_rel = lambda t, ref :  tn.linalg.norm(t-ref).numpy() / tn.linalg.norm(ref).numpy() if ref.shape == t.shape else np.inf
   
            
class TestLinalg(unittest.TestCase):

    def test_add(self):
        '''
        Test the addition operator 
        '''
        N = [10,8,6,9,12]
       
        x = tntt.random(N,[1,3,4,5,6,1])
        y = tntt.random(N,[1,2,4,5,4,1])
        z = tntt.random(N,[1,2,2,2,2,1])
        const = 3.1415926535
      
        X = x.full()
        Y = y.full()
        Z = z.full()
      
        w = x+y+z
        t = const+(const+x)+const
      
        W = X+Y+Z
        T = const+(const+X)+const
      
        self.assertTrue(err_rel(w.full(),W)<1e-14,'Addition error 1')
        self.assertTrue(err_rel(t.full(),T)<1e-14,'Addition error 2')

        M = tntt.random([(5,6),(7,8),(9,10)],[1,5,5,1])
        P = tntt.random([(5,6),(7,8),(9,10)],[1,2,20,1])
      
        Q = M+P+P+M
        Qr = M.full()+P.full()+P.full()+M.full()
      
        self.assertTrue(err_rel(Q.full(),Qr)<1e-14,'Addition error 2: TT-matrix')
      
    def test_sub(self):
        '''
        Test the subtraction operator. 
        '''
        N = [10,8,6,9,12]
       
        x = tntt.random(N,[1,3,4,5,6,1])
        y = tntt.random(N,[1,2,4,5,4,1])
        z = tntt.random(N,[1,2,2,2,2,1])
        const = 3.1415926535
      
        X = x.full()
        Y = y.full()
        Z = z.full()
      
        w = -x+y-z
        t = const+(const-x)-const
      
        W = -X+Y-Z
        T = const+(const-X)-const
      
        self.assertTrue(err_rel(w.full(),W)<1e-14,'Subtraction error 1')
        self.assertTrue(err_rel(t.full(),T)<1e-14,'Subtraction error 2')

        M = tntt.random([(5,6),(7,8),(9,10)],[1,5,5,1])
        P = tntt.random([(5,6),(7,8),(9,10)],[1,2,20,1])
      
        Q = -M+P-P-M
        Qr = -M.full()+P.full()-P.full()-M.full()
      
        self.assertTrue(err_rel(Q.full(),Qr)<1e-14,'Subtraction error 2: TT-matrix')

    def test_mult(self):
        """
        Test the pointwise multiplication between TT-objects.
        """
        A = tntt.random([(5,6),(7,8),(9,10),(4,5)],[1,5,5,3,1])
        B = tntt.random([(5,6),(7,8),(9,10),(4,5)],[1,5,5,3,1])
        Ar = A.full()
        Br = B.full()
        x = tntt.random([2,3,4,5,6],[1,2,4,8,4,1])
        y = tntt.random([2,3,4,5,6],[1,2,5,6,2,1])
        xr = x.full()
        yr = y.full()
        c = 2.5

        z = c*x*(-y*c)
        zr = c*xr*(-yr*c)

        self.assertLess(err_rel(z.full(),zr),1e-13,"Multiplication error: TT-tensors.")
        
        C = c*A*(B*c)
        Cr = c*Ar*(Br*c)

        self.assertLess(err_rel(C.full(),Cr),1e-13,"Multiplication error: Tt-matrices.")
          
    def test_matmult(self):
        self.assertTrue(True)

    def test_matvecdense(self):
        """
        Test the multiplication between a TT-matrix and a dense tensor

        """

        A = tntt.random([(5,6),(7,8),(9,10),(4,5)],[1,5,5,3,1])

        x = tn.rand([6,8,10,5], dtype = tn.float64)
        y = A @ x
        yr = tn.einsum('abcdijkl,ijkl->abcd', A.full(), x)
        self.assertLess(err_rel(y,yr),1e-14,'Dense matvec error 1.')

        x = tn.rand([32,4,33,6,8,10,5], dtype = tn.float64)
        y = A @ x
        yr = tn.einsum('abcdijkl,mnoijkl->mnoabcd',A.full(), x)
        self.assertEqual(y.shape,yr.shape,'Dense matvec shape mismatch.')
        self.assertLess(err_rel(y,yr),1e-14,'Dense matvec error 2.')

        x = tn.rand([1,22,6,8,10,5], dtype = tn.float64)
        y = A @ x
        yr = tn.einsum('abcdijkl,nmijkl->nmabcd',A.full(), x)
        self.assertEqual(y.shape,yr.shape,'Dense matvec shape mismatch.')
        self.assertLess(err_rel(y,yr),1e-14,'Dense matvec error 2.')



    def test_dot(self):
        '''
        Test the dot product between TT tensors.
        '''

        a = tntt.random([4,5,6,7,8,9],[1,2,10,16,20,7,1])
        b = tntt.random([4,5,6,7,8,9],[1,3,4,10,10,4,1])
        c = tntt.random([5,7,9],[1,2,7,1])
        d = tntt.random([4,5,9],[1,2,2,1])

        x = tntt.dot(a,b)
        y = tntt.dot(a,c,[1,3,5])
        z = tntt.dot(b,d,[0,1,5])
        
        self.assertLess(err_rel(x,tn.einsum('abcdef,abcdef->',a.full(),b.full())), 1e-12, 'Dot product error. Test: equal sized tensors.')
        self.assertLess(err_rel(y.full(),tn.einsum('abcdef,bdf->ace',a.full(),c.full())), 1e-12, 'Dot product error. Test: different sizes 1.')
        self.assertLess(err_rel(z.full(),tn.einsum('abcdef,abf->cde',b.full(),d.full())), 1e-12, 'Dot product error. Test: different sizes 2.')
        

    def test_kron(self):
        '''
        Test the Kronecker product.
        ''' 
        a = tntt.random([5,7,9],[1,2,7,1])
        b = tntt.random([4,5,9],[1,2,2,1])
        
        c = a**b
        self.assertLess(err_rel(c.full(),tn.einsum('abc,def->abcdef',a.full(),b.full())), 1e-12, 'Kronecker product error: 2 tensors.')

        A = tntt.random([(2,3),(4,5)],[1,2,1])
        B = tntt.random([(3,3),(4,2)],[1,3,1])
        
        C = A**B
        self.assertLess(err_rel(C.full(),tn.einsum('abcd,mnop->abmncdop',A.full(),B.full())), 1e-12, 'Kronecker product error: 2 tensor operators.')
        
        c = a**None
        self.assertLess(err_rel(a.full(),c.full()),1e-14,'Kronecker product error: tensor and None.')
        
        c = a**tntt.ones([])
        self.assertLess(err_rel(a.full(),c.full()),1e-14,'Kronecker product error: tensor and None.')
      
      
    def test_combination(self):
        '''
        Test sequence of linear algebra operations.
        '''
      
        x = tntt.random([4,7,13,14,19],[1,2,10,13,10,1])
        y = tntt.random([4,7,13,14,19],[1,2,4,2,4,1])
       
        x = x/x.norm()
        y = y/y.norm()
        
        z = x*x-2*x*y+y*y
        u = (x-y)*(x-y)
        norm = (z-u).norm()
        
        self.assertLess(norm.numpy(),1e-14,"Error: Multiple operations. Part 1 fails.")
  
    def test_slicing(self):
        '''
        Test the slicing operator.
        '''
      
        # print('Testing: Slicing of a tensor.')
      
        # TT-tensor
        cores = [tn.rand([1,9,3],dtype=tn.float64),tn.rand([3,10,4],dtype=tn.float64),tn.rand([4,15,5],dtype=tn.float64),tn.rand([5,15,1],dtype=tn.float64)]
        Att = tntt.TT(cores)
        A = Att.full()
      
      
      
        errs = []
        errs.append( err_rel(A[1,2,3,4],Att[1,2,3,4]) )
        errs.append( err_rel(A[1:3,2:4,3:10,4],Att[1:3,2:4,3:10,4].full()) )
        errs.append( err_rel(A[1,:,3,4],Att[1,:,3,4].full()) )
        # errs.append( err_rel(A[1,::-1,-1,4],Att[1,::-1,-1,4].full()) )
      
        # TT-matrix
        cores = [tn.rand([1,9,8,3],dtype=tn.float64),tn.rand([3,10,9,4],dtype=tn.float64),tn.rand([4,15,14,5],dtype=tn.float64),tn.rand([5,15,10,1],dtype=tn.float64)]
        Att = tntt.TT(cores)
        A = Att.full()
      
        errs.append( err_rel(A[1,2,3,4,5,4,3,2],Att[1,2,3,4,5,4,3,2]) )
        errs.append( err_rel(A[1:3,1:3,1:3,1:3,5:6,1:3,1:3,1:3],Att[1:3,1:3,1:3,1:3,5:6,1:3,1:3,1:3].full()) )
        errs.append( err_rel(A[1,1:3,1:3,1:3,2,1:3,1:3,1:3],Att[1,1:3,1:3,1:3,2,1:3,1:3,1:3].full()) )
      
        #### TODO: More testing for the TT-matrix case.
      
        self.assertFalse(max(errs) > 1e-15)
  
    def test_qtt(self):
        '''
        Test case for the QTT functions.
        '''
        N = [16,8,64,128]
        R = [1,2,10,12,1]
        x = tntt.random(N,R) 
        x_qtt = x.to_qtt()
        x_full = x.full()
      
        self.assertTrue(err_rel(tn.reshape(x_qtt.full(),x.N),x_full)<1e-12,'Tensor to QTT failed.')
      
        x = tntt.random([256,128,1024,128],[1,40,50,20,1])
        # x = tntt.random([16,8,4,16],[1,10,12,4,1])
        N = x.N
        xq = x.to_qtt()
        xx = xq.qtt_to_tens(N)

        self.assertTrue(np.abs((x-xx).norm(True)/x.norm(True))<1e-12,'TT to QTT and back not working.')
      
    def test_reshape(self):
        '''
        Test the reshape function.
        '''
      
        T = tntt.ones([3,2])
        Tf = T.full()
        Tr = tntt.reshape(T,[6])

        self.assertLess(tn.linalg.norm(tn.reshape(Tf,Tr.N)-Tr.full()).numpy(),1e-12,'TT-tensor reshape fail: test 1')


        T = tntt.random([6,8,9],[1,4,5,1])
        Tf = T.full()
        Tr = tntt.reshape(T,[2,6,12,3])

        self.assertLess(tn.linalg.norm(tn.reshape(Tf,Tr.N)-Tr.full()).numpy(),1e-12,'TT-tensor reshape fail: test 2')

        T = tntt.random([6,8,9],[1,4,5,1])
        Tf = T.full()
        Tr = tntt.reshape(T,[2,3,4,2,3,3])

        self.assertLess(tn.linalg.norm(tn.reshape(Tf,Tr.N)-Tr.full()).numpy(),1e-12,'TT-tensor reshape fail: test 3')


        T = tntt.random([2,3,4,2,3,2,5],[1,2,3,4,4,5,2,1])
        Tf = T.full()
        Tr = tntt.reshape(T,[6,24,10])

        self.assertLess(tn.linalg.norm(tn.reshape(Tf,Tr.N)-Tr.full()).numpy(),1e-11,'TT-tensor reshape fail: test 4')
      
        # test TT-matrix
      
        A = tntt.random([(9,4),(16,6)],[1,4,1])
        Af = A.full()
        Ar = tntt.reshape(A,[(3,2),(3,2),(4,2),(4,3)])

        self.assertLess(tn.linalg.norm(tn.reshape(Af,Ar.M+Ar.N)-Ar.full()).numpy(),1e-12,'TT-matrix reshape fail: test 1')

        A = tntt.random([(9,4),(16,6),(3,5)],[1,4,5,1])
        Af = A.full()
        Ar = tntt.reshape(A,[(3,2),(6,6),(24,10)])

        self.assertLess(err_rel(Ar.full(),tn.reshape(Af,Ar.M+Ar.N)),1e-13,'TT-matrix reshape fail: test 2')
      
        A = tntt.random([(4,8),(16,12),(2,8),(6,4)],[1,4,7,2,1])
        T = tntt.random([8,12,8,4],[1,3,9,3,1])
        Ar = tntt.reshape(A,[(2,4),(4,6),(4,2),(8,32),(3,2)])
        Tr = tntt.reshape(T,[4,6,2,32,2])
        Af = A.full()
        Tf = T.full()
        Ur = Ar@Tr
        U = A@T
        self.assertLess(err_rel(Ur.full(),tn.reshape(U.full(),Ur.N)),1e-13,'TT-matrix reshape fail: test 3')

    def test_mask(self):
        """
        Test the apply_mask() method.
        """
        indices = tn.randint(0,20,(1000,4))

        x = tntt.random([21,22,23,21],[1,10,10,10,1])
        xf = x.full()

        vals = x.apply_mask(indices)
        vals_ref = 0*vals
        for i in range(len(indices)):
            vals_ref[i] = xf[tuple(indices[i])]
        
        self.assertLess(tn.linalg.norm(vals-vals_ref), 1e-12, "Mask method error.")

        
if __name__ == '__main__':
    unittest.main()