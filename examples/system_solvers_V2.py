"""
Linear solvers in the TT format

This tutorial addresses solving multilinear systems $\mathsf{Ax}=\mathsf{b}$ in the TT format.
For this we solve the 3 dimensional and 6 dimensional poisson equation with dirichlet boundary conditions.

"""

#%% Imports
import torch as tn
import torchtt as tntt 
import datetime
from torchtt.finitediff import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.linalg import toeplitz

#%%  Finite differences
# We now solve the problem $\Delta u = b$ in $[-1,1]^d$ with $ u = u_{ref} $ on the entire boundary using finite differences.
# First, set the size of the problem (n is the mode size and d is the number of dimensions):

dtype = tn.float64 

#defining Borders, Gridpoints and dimensions
a = -1
b = 1
n = 128
d = 3
N = [n]*d

#Calculating the Grid Tensos
xs = tntt.meshgrid([tn.linspace(a,b,n,dtype=dtype) for n in N])


#cross function approximation of function 1 $...$
func = lambda I: 2*((tn.sum((I**2),1).to(dtype=tn.float64))-3)/((tn.sum((I**2),1).to(dtype=tn.float64)+1)**3)
f_cross = tntt.interpolate.function_interpolate(func, xs)
f=f_cross


#cross function approximation of function 2 $...$
#func = lambda I: -d*tn.pi**2*1/tn.sum((I**2)+1,1).to(dtype=tn.float64)
#f_cross = tntt.interpolate.function_interpolate(func, xs)
#f=f_cross


#reference function 1
u_ref = 1/(1+xs[0].full()**2+xs[1].full()**2+xs[2].full()**2)


#reference function 1
#u_ref =  tn.sin(tn.pi*xs[0].full())* tn.sin(tn.pi*xs[1].full())*tn.sin(tn.pi*xs[2].full())


#calculation of the Laplacian
L = laplacian(xs)

#calculation of the righthandside
b_tt = righthandside(xs, inner_solution = f,  boundary_solution = tntt.TT(u_ref) )


# Solve the system 
time = datetime.datetime.now()
u = tntt.solvers.amen_solve(L, b_tt ,x0 = b_tt, nswp = 20, eps = 1e-10, verbose = True, preconditioner=None)
time = datetime.datetime.now() - time
#print('Relative residual: ',(L_tt@x-b_tt).norm()/b_tt.norm())
print('Solver time: ',time)


#plotting
ax = plt.subplot()

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

im = ax.imshow(u[:,:,1].full())

plt.colorbar(im, cax=cax)
plt.show()

ax = plt.subplot()

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

im = ax.imshow(u_ref[:,:,1])

plt.colorbar(im, cax=cax)
plt.show()


L22 = tntt.dot(u-tntt.TT(u_ref),u-tntt.TT(u_ref))*(1/(n**d)) 

print(L22)

L2 = tn.sqrt((tntt.TT(u_ref)-u).full()**2)#u_ref


plt.colorbar(im, cax=cax)
plt.show()

ax = plt.subplot()

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)


im = ax.imshow(L2[:,:,1])


plt.colorbar(im, cax=cax)
plt.show()

