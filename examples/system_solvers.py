"""
Linear solvers in the TT format

This tutorial addresses solving multilinear systems $\mathsf{Ax}=\mathsf{b}$ in the TT format.

"""

#%% Imports
import torch as tn
import torchtt as tntt 
import datetime


#%% Small example
# A random tensor operator $\mathsf{A}$ is created in the TT format. We create a random right-hand side $\mathsf{b} = \mathsf{Ax}$, where $\mathsf{x}$ is a random tensor in the TT format. 
# This way the solution of $\mathsf{Ax}=\mathsf{b}$ is known and we can compare it as a reference. This works only for small random tensors.
A = tntt.random([(4,4),(5,5),(6,6)],[1,2,3,1]) 
x = tntt.random([4,5,6],[1,2,3,1])
b = A @ x

# Solve the multilinear system $\mathsf{Ax}=\mathsf{b}$ using the method torchtt.solvers.amen_solve().
xs = tntt.solvers.amen_solve(A,b, x0 = b, eps = 1e-7)

# The relative residual norm and the relative error of the solution are reported:
print(xs)
print('Relative residual error ',(A@xs-b).norm()/b.norm())
print('Relative error of the solution  ',(xs-x).norm()/x.norm())


#%%  Finite differences
# We now solve the problem $\Delta u = 1$ in $[0,1]^d$ with $ u = 0 $ on the entire boundary using finite differences.
# First, set the size of the problem (n is the mode size and d is the number of dimensions):
dtype = tn.float64 
n =  64
d = 8

# Create the finite differences matrix corresponding to the problem. The operator is constructed directly in the TT format as it follows
L1d = -2*tn.eye(n, dtype = dtype)+tn.diag(tn.ones(n-1,dtype = dtype),-1)+tn.diag(tn.ones(n-1,dtype = dtype),1)
L1d[0,1] = 0
L1d[-1,-2] = 0
L1d *= (n-1)**2
L1d = tntt.TT(L1d, [(n,n)])

L_tt = tntt.zeros([(n,n)]*d)
for i in range(1,d-1):
    L_tt = L_tt+tntt.eye([n]*i)**L1d**tntt.eye([n]*(d-1-i))
L_tt = L_tt + L1d**tntt.eye([n]*(d-1)) +  tntt.eye([n]*(d-1))**L1d
L_tt = L_tt.round(1e-14)

# The right hand site of the finite difference system is also computed in the TT format
b1d = tn.ones(n, dtype=dtype)
#b1d[0] = 0
#b1d[-1] = 0
b1d = tntt.TT(b1d)
b_tt = b1d
for i in range(d-1):
    b_tt = b_tt**b1d
   
# Solve the system 
time = datetime.datetime.now()
x = tntt.solvers.amen_solve(L_tt, b_tt ,x0 = b_tt, nswp = 20, eps = 1e-7, verbose = True, preconditioner='c', use_cpp = True)
time = datetime.datetime.now() - time
print('Relative residual: ',(L_tt@x-b_tt).norm()/b_tt.norm())
print('Solver time: ',time)

# Display the structure of the TT
print(x)


#%% Try one more time on the GPU (if available).
if tn.cuda.is_available():
    time = datetime.datetime.now()
    x = tntt.solvers.amen_solve(L_tt.cuda(), b_tt.cuda() ,x0 = b_tt.cuda(), nswp = 20, eps = 1e-8, verbose = True, preconditioner='c')
    time = datetime.datetime.now() - time
    x = x.cpu()
    print('Relative residual: ',(L_tt@x-b_tt).norm()/b_tt.norm())
    print('Solver time: ',time)
else:
    print('GPU not available...')