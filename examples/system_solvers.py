#%% Imports
import tnt
import tnt.dmrg_solver
import tnt.solvers
import torch as tn

#%% Define the system matrix and rhs in TT
dtype = tn.float64 # default dtype
n =  300 # mode size

# define 1d matrix
A1d = -2*tn.eye(n, dtype = dtype)+tn.diag(tn.ones(n-1,dtype = dtype),-1)+tn.diag(tn.ones(n-1,dtype = dtype),-1)
A1d[0,1] = 0
A1d[-1,-2] = 0
A1d /= (n-1)
A1d = tnt.TT(A1d, [(n,n)])
# construct 4d TT-matrix
A = A1d**tnt.eye([n,n,n]) +  tnt.eye([n])**A1d**tnt.eye([n,n]) + tnt.eye([n,n])**A1d**tnt.eye([n]) + tnt.eye([n,n,n])**A1d

# define 1d rhs
b1d = tn.ones(n, dtype=dtype)
b1d[0] = 0
b1d[-1] = 0
b1d = tnt.TT(b1d)
# construct 4d rhs in TT
b = b1d**b1d**b1d**b1d

#%% Solve using AMEn

x = tnt.solvers.amen_solve(A,b)


#%% Solve using TT-GMRES