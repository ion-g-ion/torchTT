import torch as tn 
import torchtt as tntt 
import numpy as np

A = tntt.randn([(5,5),(6,6),(4,4),(4,4)],[1,2,3,3,1], dtype=tn.float64)
x = tntt.randn([5,6,4,4],[1,4,2,5,1], dtype = tn.float64)
b = (A@x).round(1e-18)

xs = tntt.solvers.amen_solve_cpp(A,b,eps = 1e-10, max_full=10000, verbose=True)