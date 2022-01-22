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

        self.assertListEqual(gr_ad.R,[2*r if r!=1 else 1 for r in R],"Riemannian gradient error: ranks mismatch.")
        self.assertListEqual(gr_proj.R,[2*r if r!=1 else 1 for r in R],"Riemannian projection error: ranks mismatch.")