"""
# Cross approximation in the TT format

Using the `torchtt.TT` constructor, a TT decomposition of a given tensor can be obtained. 
However, in the cases where the entries of the tensor are computed using a given function, building full tensors becomes unfeasible. 
It is possible to construct a TT decomposition using only a part of the entries of the full tensor. 
This is called the cross approximation method.
"""

#%% Imports
import torch as tn
import torchtt as tntt


#%% Cross interpolation of a tensor in TT format
# We want to approximate the tensor $\mathsf{x}_{i_1...i_d}=\frac{1}{2+i_1+\cdots+i_d}$. 
# Since the passed indices are integers of type torch.int64, casting is used.
func1 = lambda I: 1/(2+tn.sum(I+1,1).to(dtype=tn.float64))

# Call the torchtt.interpolate.dmrg_cross() method.
N = [20]*4
x = tntt.interpolate.dmrg_cross(func1, N, eps = 1e-7)

# Compute the full tensor and compare to the reference.
Is = tntt.meshgrid([tn.arange(0,n,dtype=tn.float64) for n in N])
x_ref = 1/(2+Is[0].full()+Is[1].full()+Is[2].full()+Is[3].full()+4)
print('Relative error ',tn.linalg.norm(x.full()-x_ref)/tn.linalg.norm(x_ref))

# We consider the case $d=10$, $n_i=32$. the full tensor would contain $32^{10}$ entries.
# The total number of functions calls is in this case 25000000 compared to $32^{10}$ of the total number of entries.
N = [32]*10
x = tntt.interpolate.dmrg_cross(func1, N, eps = 1e-10, verbose=True)

# The adaptive cross method used only a fraction of function calls from the original tensor.
# Check some entries (full tensor cannot be computed this time) and show the rank and the storage requirements.
print(x[1,2,3,4,5,6,7,8,9,11], ' reference ', func1(tn.tensor([[1,2,3,4,5,6,7,8,9,11]])))
print(x[12,23,17,25,30,0,7,8,9,11], ' reference ', func1(tn.tensor([[12,23,17,25,30,0,7,8,9,11]])))
print(x)


#%% Element wise application of an univariate function on a TT tensor.
# Let $f:\mathbb{R}\rightarrow\mathbb{R}$ be a function and $\mathsf{x}\in\mathbb{R}^{N_1\times\cdots\times N_d}$ be a tensor with a known TT approximation. The goal is to determine the TT approximation of $\mathsf{y}_{i_1...i_d}=f(\mathsf{x}_{i_1...i_d})$ within a prescribed relative accuracy $\epsilon$ (passed as argument).
# In this case the function is torchtt.interpoalte.function_interpolate() and takes as arguments a function handle, the tensor $\mathsf{x}$, the accuracy epsilon, a initial tensor (starting point), number of sweeps (nswp) and the size of the rank enrichment (kick).
# Further arguments are the dtype of the result and the verbose flag.
# The function handle as argument gets as arguments torch vectors and has to return torch vectors of the same size.
# The following example computes the elemntwise natural logarithm of a tensor. The relative error of the result is also reported.
x = tntt.TT(x_ref)
func = lambda t: tn.log(t)
y = tntt.interpolate.function_interpolate(func, x, 1e-9)
print('Relative error ',tn.linalg.norm(y.full()-func(x_ref))/tn.linalg.norm(func(x_ref)))


#%% Element wise application of an multivariate function on a TT tensor.
# Let $f:\mathbb{R}\rightarrow\mathbb{R}$ be a function and $\mathsf{x}^{(1)},...,\mathsf{x}^{(d)}\in\mathbb{R}^{N_1\times\cdots\times N_d}$ be a tensor with a known TT approximation. 
# The goal is to determine the TT approximation of $\mathsf{y}_{i_1...i_d}=f(\mathsf{x}_{i_1...i_d}^{(1)},...,\mathsf{x}^{(d)})$ within a prescribed relative accuracy $\epsilon$ (passed as argument). 
# The function is the same as in the previous case tochtt.interpoalte.function_interpolate(), but the second argument in this case is a list of torchtt.TT tensors. The function handle takes as argument a $M\times d$ torch.tensor and every of the $M$ lines corresponds to an evaluation of the function $f$ at a certain tensor entry. The function handle returns a torch tensor of length $M$.

# The following example computes the same tensor as in the previous case, but with the tochtt.interpoalte.function_interpolate() method.
z = tntt.interpolate.function_interpolate(func1, Is)
print('Relative error ',tn.linalg.norm(z.full()-x_ref)/tn.linalg.norm(x_ref))
