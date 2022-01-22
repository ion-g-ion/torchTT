"""
# Automatic differentiation

Being based on `pytorch`, `torchtt` can handle automatic differentiation with respect to the TT cores. 
"""

#%% Imports
import torch as tn
import torchtt as tntt

#%% First, a function to differentiate is created and some tensors:
N = [2,3,4,5]
A = tntt.randn([(n,n) for n in N],[1]+[2]*(len(N)-1)+[1])
y = tntt.randn(N,A.R)
x = tntt.ones(N)

def f(x,A,y):
    z = tntt.dot(A @ (x-y),(x-y))
    return z.norm()

#%% In order to compute the derivative of a scalar with respect to all cores of a TT object, the AD graph recording has to be started:
tntt.grad.watch(x)

#%% Using the `torchtt.grad.grad()` method, the gradient is computed:
val = f(x,A,y)
grad_cores = tntt.grad.grad(val, x)

#%% The variable `grad_cores` is a list of tensors representing the derivatives of `f()` with resect to the individual core entries.
# For checking, we compute the derivative of teh function with respect to one element of the core
h = 1e-7
x1 = x.clone()
x1.cores[1][0,0,0] += h
x2 = x.clone()
x2.cores[1][0,0,0] -= h
derivative = (f(x1,A,y)-f(x2,A,y))/(2*h)
print(tn.abs(derivative-grad_cores[1][0,0,0])/tn.abs(derivative))

# The functions `torchtt.grad.grad()` and `torchtt.grad.watch()` can take an additional list of modes `core_indices` as argument which decides which cores are watched and differentiaated with respect to.
