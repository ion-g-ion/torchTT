""""
Linear solvers in the TT format

This tutorial addresses solving multilinear systems $\mathsf{Ax}=\mathsf{b}$ in the TT format.
For this we solve the 3 dimensional and 6 dimensional poisson equation with dirichlet boundary conditions.

"""

#%% Imports
import torch as tn
import timeit
import torchtt as tntt 
import datetime
#from torchtt.finitediff_old2 import *
from torchtt.finitediff import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.linalg import toeplitz

#%%  Finite differences
# We now solve the problem $\Delta u = b$ in $[-1,1]^d$ with $ u = u_{ref} $ on the entire boundary using finite differences.
# First, set the size of the problem (n is the mode size and d is the number of dimensions):

dtype = tn.float64 

#defining Borders, Gridpoints and dimensions
a = 0
b = 1
n = 128
d = 3
N = [n]*d
dtype=tn.float64
device=None
#%%  creating uniform Mesh

#Calculating the Grid Tensos
xn = tntt.meshgrid([tn.linspace(a,b,n,dtype=dtype) for n in N])
#%%  creating non-uniform Mesh
vec_list=[]
for i in range(d):
    vector = 2 * tn.rand(n, dtype = tn.float64) - 1
    sorted_vector , _ = vector.sort()
    vec_list.append(sorted_vector)
#xn = tntt.meshgrid(vec_list)


#%%  creating f and u_ref
f=tntt.ones([n]*d, dtype=dtype)*6


if d == 1:
    u_ref = xn[0].full()**2
    u_ref = tntt.TT(u_ref)
else:   
   
    func_ref  = lambda I: 27*tn.prod((I),1)*tn.prod((I-1),1)
    u_ref = tntt.interpolate.function_interpolate(func_ref, xn)
#%%  creating operators
#calculation of the Laplacian
L = laplacian(xn, boundarycondition = "Dirichlet")

#calculation of the righthandside
zeros=tntt.zeros([n]*d, dtype=dtype)
b_tt = righthandside(xn, inner_solution = f,  boundary_solution = zeros, boundarycondition="Dirichlet")

#%%  Solve the system 
time = datetime.datetime.now()
u = tntt.solvers.amen_solve(L.cuda(), b_tt.cuda() ,x0 = b_tt.cuda(), nswp = 20, eps = 1e-8, verbose = True, preconditioner=None)
time = datetime.datetime.now() - time
#print('Relative residual: ',(L_tt@x-b_tt).norm()/b_tt.norm())
print('Solver time: ',time)
#calculate error
L2 = tntt.dot(u-u_ref.cuda(),u-u_ref.cuda())*(1/(n**d)) 
print(L2)

#%%  plotting

slices = [1] * d
slices[0] = slice(None)
if d > 1:
    slices[1] = slice(None)
#plotting solution
ax = plt.subplot()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
if d > 1:
    im = ax.imshow(u[tuple(slices)].cpu().full())
else:
    im = ax.imshow(u[tuple(slices)].cpu().full().reshape(n,1))

plt.colorbar(im, cax=cax)
plt.show()

#%%  plotting reference solution
ax = plt.subplot()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
if d > 1:
    im = ax.imshow(u_ref[tuple(slices)].cpu().full())
else:
    im = ax.imshow(u_ref[tuple(slices)].cpu().full().reshape(n,1))

plt.colorbar(im, cax=cax)
plt.show()

#%%  plotting error
error = tn.sqrt((u_ref.cuda()-u)[tuple(slices)].cpu().full()**2)
ax = plt.subplot()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
im = ax.imshow(error)
plt.colorbar(im, cax=cax)
plt.show()


