"""
# AMEn and DMRG for fast TT operations

The torchtt package includes density matrix renormalization group (DMRG; [Oseledets, 2011](https://doi.org/10.2478/cmam-2011-0021)) and alternating minimal energy (AMEn; [Dolgov and Savostyanov, 2014](https://doi.org/10.1137/140953289)) schemes for fast matrix vector products and elementwise inversion in the tensor-train (TT) format.
"""

#%% Imports
import torch as tn
import torchtt as tntt
import datetime


#%% Efficient matrix vector product
# When performing the multiplication between a a TT matrix and a TT tensor the rank of the result is the product of the ranks of the inputs. 
# Therefore rank rounding has to be performed. This increases the complexity to $\mathcal{O}(Ndr^6)$. 
# In order to overcome this, Oseledets proposed in ["DMRG Approach to Fast Linear Algebra in the TT-Format"](https://doi.org/10.2478/cmam-2011-0021) the DMRG optimization scheme to reduce the complexity. 
# This feature is implemented in torchtt by the member function fast_matvec() of the TT class. An example is showed in the following.

# Create a random TT object and a TT matrix.
n = 4 # mode size
A = tntt.random([(n,n)]*8,[1]+7*[4]+[1]) # random array
x = tntt.random([n]*8,[1]+7*[5]+[1]) # random tensor 

# Increase the rank without adding redundant information. 
# The multiplication performed in this case is actually equivalent to $32\mathbf{\mathsf{Ax}}$. 
A = A + A + A + A - A + A - A + A
x = x + x + x + x + x + x + x + x - x + x - x + x 
print(A)
print(x)

# Perform the TT matvec directly and round the result. The runtime is reported.
tme = datetime.datetime.now()
y = (A @ x).round(1e-12) 
tme = datetime.datetime.now() - tme 
print('Time classic ', tme)

# This time run the fast matvec routine.
tme = datetime.datetime.now()
yf = A.fast_matvec(x)
tme = datetime.datetime.now() - tme 
print('Time DMRG    ', tme)

# Check if the error is the same (debugging purpose).
print('Relative error ',(y-yf).norm().numpy()/y.norm().numpy())

# A second routine is `torchtt.fast_mv()`. The method is described in [Michailidis, Fenton and Kiffner, 2025](https://doi.org/10.1137/24M1714149) (preprint: [arXiv:2410.19747](https://arxiv.org/abs/2410.19747)). It works well for tensors in the quantized tensor-train (QTT) format,
# i.e. tensors reshaped so that every mode has size 2 (see `torchtt.TT.to_qtt()`; [Oseledets, 2010](https://doi.org/10.1137/090757861) and [Khoromskij, 2011](https://doi.org/10.1007/s00365-011-9131-1)).
# In the example below, `x` has 8 modes of size 2 (a vector of length 2^8 = 256) and `A` is a matching 256 x 256 operator with modes of size (2,2).
A = tntt.random([(2,2)]*8,[1]+7*[6]+[1]) # random array
x = tntt.random([2]*8,[1]+7*[5]+[1]) # random tensor 
for _ in range(8): A+=A
for _ in range(8): x+=x

tme = datetime.datetime.now()
yf2 = tntt.fast_mv(A, x)
tme = datetime.datetime.now() - tme 
print('Time fast 2  ', tme)

#%% Elementwise division in the TT format
# One other basic linear algebra function that cannot be done without optimization is the elementwise division of two tensors in the TT format.
# In contrast to the elemntwise multiplication (where the resulting TT cores can be explicitly computed), the elementwise inversion has to be solved by means of an optimization problem (the method of choice is AMEn). 
# The operator "/" can be used  for elemntwise division between tensors. Moreover one can use "/" between a scalar and a  torchtt.TT instance.

# Create 2 tensors:
# - $\mathsf{x}_{i_1i_2i_3i_4} = 2 + i_1$
# - $\mathsf{y}_{i_1i_2i_3i_4} = i_1^2+i_2+i_3+1$
# and express them in the TT format. For both of them a TT decomposition of the elemmentwise inverse cannot be explicitly formed.
N = [32,50,44,64]
I = tntt.meshgrid([tn.arange(n,dtype = tn.float64) for n in N])
x = 2+I[0]
x = x.round(1e-15)
y = I[0]*I[0]+I[1]+I[2]+I[3]+1
y = y.round(1e-15)

# Perform $\mathsf{z}_{\mathbf{i}} = \frac{\mathsf{x}_{\mathbf{i}}}{\mathsf{z}_{\mathbf{i}}}$ and report the relative error.
z = x/y
print('Relative error', tn.linalg.norm(z.full()-x.full()/y.full())/tn.linalg.norm(z.full()))

# Perform $\mathsf{u}_{\mathbf{i}} = \frac{1}{\mathsf{z}_{\mathbf{i}}}$ and report the relative error.
u = 1/y
print('Relative error', tn.linalg.norm(u.full()-1/y.full())/tn.linalg.norm(u.full()))

# Following are also possible:
# - scalar (float, int) divided elementwise by a tensor in the TT format.
# - torch.tensor with 1 element divided elementwise by a tensor in the TT format.
w = 1.0/y
a = tn.tensor(1.0)/y
