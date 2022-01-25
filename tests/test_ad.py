"""
Test all the AD related functions.

@author: ion
"""
import torch as tn
import torchtt 
import unittest

err_rel = lambda t, ref :  tn.linalg.norm(t-ref).numpy() / tn.linalg.norm(ref).numpy() if ref.shape == t.shape else np.inf


class TestAD(unittest.TestCase):

    def test_manifold(self):
        """
        Compare the result of the manifold projection and the manifold gradient computed using AD.
        """

        target = torchtt.randn([10,12,14,16],[1,8,8,7,1])

        func = lambda x: 0.5*(x-target).norm(True)

        R = [1,3,4,6,1]
        x = torchtt.randn(target.N,R.copy())

        gr_ad = torchtt.manifold.riemannian_gradient(x, func)

        gr_proj = torchtt.manifold.riemannian_projection(x, (x-target))

        self.assertListEqual(gr_ad.R,[2*r if r!=1 else 1 for r in R],"TT manifold: Riemannian gradient error: ranks mismatch.")
        self.assertListEqual(gr_proj.R,[2*r if r!=1 else 1 for r in R],"TT manifold: Riemannian projection error: ranks mismatch.")
        self.assertLess(err_rel(gr_ad.full(),gr_proj.full()),1e-12,"TT manifold: Riemannian gradient and projected gradient differ.")
    
    def test_ad(self):
        """
        Test the AD functionality.
        """
        N = [2,3,4,5]
        A = torchtt.randn([(n,n) for n in N],[1]+[2]*(len(N)-1)+[1])
        y = torchtt.randn(N,A.R)
        x = torchtt.ones(N)

        def f(x,A,y):
            z = torchtt.dot(A @ (x-y),(x-y))
            return z.norm()

        torchtt.grad.watch(x)

        val = f(x,A,y)
        grad_cores = torchtt.grad.grad(val, x)

        torchtt.grad.watch(A)

        val = f(x,A,y)
        grad_cores_A = torchtt.grad.grad(val, A)
             
        self.assertListEqual([c.shape for c in grad_cores],[c.shape for c in x.cores],"TT AD: problem for grad w.r.t. TT tensor.")
        self.assertListEqual([c.shape for c in grad_cores_A],[c.shape for c in A.cores],"TT AD: problem for grad w.r.t. TT matrix.")                   
        # h = 1e-7
        # x1 = x.clone()
        # x1.cores[1][0,0,0] += h
        # x2 = x.clone()
        # x2.cores[1][0,0,0] -= h
        # derivative = (f(x1,A,y)-f(x2,A,y))/(2*h)
        # print(tn.abs(derivative-grad_cores[1][0,0,0])/tn.abs(derivative))
        
if __name__ == '__main__':
    unittest.main()