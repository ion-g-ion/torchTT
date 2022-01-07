import unittest
import torchtt as tntt
import torch as tn
import numpy as np

err_rel = lambda t, ref :  tn.linalg.norm(t-ref).numpy() / tn.linalg.norm(ref).numpy() if ref.shape == t.shape else np.inf


class TestDecomposition(unittest.TestCase):

    def test_init(self):
        """
        Checks the constructor and the TT.full() function. 
        A list of cores is passed and is checked if the recomposed tensor is correct.
    
        Returns
        -------
        bool
            True if test is passed.
    
        """
        
        # print('Testing: Initialization from list of cores.')
        cores = [tn.rand([1,20,3],dtype=tn.float64),tn.rand([3,10,4],dtype=tn.float64),tn.rand([4,5,1],dtype=tn.float64)]
        
        T = tntt.TT(cores)
        
        Tfull = T.full()
        T_ref = tn.squeeze(tn.einsum('ijk,klm,mno->ijlno',cores[0],cores[1],cores[2]))
        
        self.assertTrue(err_rel(Tfull,T_ref) < 1e-14)
       
    
    def test_decomposition_random(self):
        '''
        Perform a TT decomposition of a random full random tensor and check if the decomposition is accurate.

        Returns
        -------
        None.

        '''
        # print('Testing: TT-decomposition from full (random tensor).')
        T_ref = tn.rand([10,20,30,5],dtype=tn.float64)
        
        T = tntt.TT(T_ref,eps = 1e-19,rmax = 1000)
        
        Tfull = T.full()
        
        self.assertTrue(err_rel(Tfull,T_ref) < 1e-12)
           
         
    def test_decomposition_lowrank(self):
        """
        Check the decomposition of a tensor which is already in the low rank format.
    
        Returns
        -------
        None.
        
        """
        # print('Testing: TT-decomposition from full (already low-rank).')
        cores = [tn.rand([1,200,30],dtype=tn.float64), tn.rand([30,100,4],dtype=tn.float64), tn.rand([4,50,1],dtype=tn.float64)]
        T_ref = tn.squeeze(tn.einsum('ijk,klm,mno->ijlno',cores[0],cores[1],cores[2]))
        
        T = tntt.TT(T_ref,eps = 1e-19)
        
        Tfull = T.full()
        
        self.assertTrue(err_rel(Tfull,T_ref)<1e-12)
        
    
         
    def test_decomposition_highd(self):
        """
        Decompose a 20d tensor with all modes 2.
    
        Returns
        -------
        None.
    
        """
        # print('Testing: TT-decomposition from full (long  20d TT).')
        cores = [tn.rand([1,2,16],dtype=tn.float64)] + [tn.rand([16,2,16],dtype=tn.float64) for i in range(18)] + [tn.rand([16,2,1],dtype=tn.float64)]
        T_ref = tntt.TT(cores).full()
        
        T = tntt.TT(T_ref,eps = 1e-12)
        Tfull = T.full()
        
        self.assertTrue(err_rel(Tfull,T_ref)<1e-12)
        
         
    def test_decomposition_ttm(self):
        """
        Decompose a TT-matrix.
    
        Returns
        -------
        bool
            True if test is passed.
    
        """
        
        T_ref = tn.rand([10,11,12,15,17,19],dtype=tn.float64)
        
        T = tntt.TT(T_ref, shape = [(10,15),(11,17),(12,19)], eps = 1e-19, rmax = 1000)
        Tfull = T.full()
        
        self.assertTrue(err_rel(Tfull,T_ref)<1e-12)
            
         
    def test_decomposition_orthogonal(self):
        """
        Checks the lr_orthogonal function. The reconstructed tensor should remain the same.
    
        Returns
        -------
        None.
    
        """
        # print('Testing: TT-orthogonalization.')
        cores = [tn.rand([1,20,3],dtype=tn.float64), tn.rand([3,10,4],dtype=tn.float64), tn.rand([4,5,20],dtype=tn.float64), tn.rand([20,5,2],dtype=tn.float64), tn.rand([2,10,1],dtype=tn.float64)]
        T = tntt.TT(cores)
        T = tntt.random([3,4,5,3,8,7,10,3,5,6],[1,20,12,34,3,50,100,12,2,80,1])
        T_ref = T.full()
        
        cores, R = tntt.decomposition.lr_orthogonal(T.cores, T.R, T.is_ttm)
        Tfull = tntt.TT(cores).full()
        
        self.assertTrue(err_rel(Tfull,T_ref)<1e-12,'Left to right ortho error too high.')
        
        for i in range(len(cores)):
            c = cores[i]
            L = tn.reshape(c,[-1,c.shape[-1]]).numpy()
            self.assertTrue(np.linalg.norm(L.T @ L - np.eye(L.shape[1])) < 1e-12 or i==len(cores)-1,'Cores are not left orthogonal after LR orthogonalization.')
                
        
        cores, R = tntt.decomposition.rl_orthogonal(T.cores, T.R, T.is_ttm)
        Tfull = tntt.TT(cores).full()
        
        self.assertTrue(err_rel(Tfull,T_ref)<1e-12,'Right to left ortho error too high.')

        
        for i in range(len(cores)):
            c = cores[i]
            R = tn.reshape(c,[c.shape[0],-1]).numpy()
            self.assertTrue(np.linalg.norm(R @ R.T - np.eye(R.shape[0])) < 1e-12 or i==0)
                
    
    def test_decomposition_rounding(self):
        """
        Testing the rounding of a TT-tensor.
        A rank-4tensor is constructed and successive approximations are performed.
    
        Returns
        -------
        None.
            
        """
        # print('Testing: TT-rounding.')
        
        T1 = tn.einsum('i,j,k->ijk',tn.rand([20],dtype=tn.float64),tn.rand([30],dtype=tn.float64),tn.rand([32],dtype=tn.float64))
        T2 = tn.einsum('i,j,k->ijk',tn.rand([20],dtype=tn.float64),tn.rand([30],dtype=tn.float64),tn.rand([32],dtype=tn.float64))
        T3 = tn.einsum('i,j,k->ijk',tn.rand([20],dtype=tn.float64),tn.rand([30],dtype=tn.float64),tn.rand([32],dtype=tn.float64))
        T4 = tn.einsum('i,j,k->ijk',tn.rand([20],dtype=tn.float64),tn.rand([30],dtype=tn.float64),tn.rand([32],dtype=tn.float64))
        
        T_ref = T1 / tn.linalg.norm(T1) + 1e-3*T2 / tn.linalg.norm(T2) + 1e-6*T3 / tn.linalg.norm(T3) + 1e-9*T4 / tn.linalg.norm(T4)
        T3 = T1 / tn.linalg.norm(T1) + 1e-3*T2 / tn.linalg.norm(T2) + 1e-6*T3 / tn.linalg.norm(T3) 
        T2 = T1 / tn.linalg.norm(T1) + 1e-3*T2 / tn.linalg.norm(T2) 
        T1 = T1 / tn.linalg.norm(T1) 
        
        
        T = tntt.TT(T_ref)
        T = T.round(1e-9)
        Tfull = T.full()
        self.assertEqual(T.R,[1,3,3,1],'Case 1: Ranks not equal')
        self.assertTrue(err_rel(Tfull,T_ref) < 1e-9,'Case 1: error too high')
        
        
        T = tntt.TT(T_ref)
        T = T.round(1e-6)
        Tfull = T.full()
        self.assertEqual(T.R,[1,2,2,1],'Case 2: Ranks not equal')
        self.assertTrue(err_rel(Tfull,T_ref) < 1e-6,'Case 1: error too high')

        T = tntt.TT(T_ref)
        T = T.round(1e-3)
        Tfull = T.full()
        self.assertEqual(T.R,[1,1,1,1],'Case 3: Ranks not equal')
        self.assertTrue(err_rel(Tfull,T_ref) < 1e-3,'Case 1: error too high')

if __name__ == '__main__':
    unittest.main()

