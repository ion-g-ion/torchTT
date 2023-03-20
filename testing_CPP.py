import torch as tn 
import torchtt as tntt 
import numpy as np
import datetime

tn.manual_seed(12345)

A = tntt.randn([(15,15),(16,16),(14,14),(14,14)],[1,2,3,3,1], dtype=tn.float64)
x = tntt.randn([15,16,14,14],[1,3,2,5,1], dtype = tn.float64)
A = A/A.norm()
x = x/x.norm()
b = (A@x).round(1e-18)
# A = A.cuda()
# b = b.cuda()
# x = x.cuda()

# xs = tntt.solvers.amen_solve(A,b,eps = 1e-10, max_full=23, verbose=True, preconditioner='c')
tme = datetime.datetime.now()
xs = tntt.solvers.amen_solve_cpp(A,b,eps = 1e-10, max_full=2003, verbose=True, preconditioner='c')
print(datetime.datetime.now()-tme)

print(xs)
print((A@xs-b).norm()/b.norm())
