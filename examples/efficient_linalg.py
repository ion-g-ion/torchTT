"""
# AMEn and DMRG for fast TT operations

The torchtt package includes density matrix renormalization group (DMRG; [Oseledets, 2011](https://doi.org/10.2478/cmam-2011-0021)) and alternating minimal energy (AMEn; [Dolgov and Savostyanov, 2014](https://doi.org/10.1137/140953289)) schemes for fast matrix vector products and elementwise inversion in the tensor-train (TT) format.
"""

#%% Imports
import torch as tn
import torchtt as tntt
import datetime


#%% Efficient products
# When performing the multiplication between a TT matrix and a TT tensor, the rank of the result is the product of the ranks of the inputs.
# Therefore rank rounding has to be performed. This increases the complexity to $\mathcal{O}(dnr^6)$.
# If the result has a much smaller rank than the product of the ranks, the rounded result can be computed without forming the exact product, for example with the DMRG scheme proposed by Oseledets in ["DMRG Approach to Fast Linear Algebra in the TT-Format"](https://doi.org/10.2478/cmam-2011-0021).
# The functions `torchtt.matvec()`, `torchtt.matmat()` and `torchtt.hadamard()` compute the rounded products and the argument `method` chooses the algorithm:
#  - `'direct'`: form the exact product and round it,
#  - `'dmrg'`: DMRG ([Oseledets, 2011](https://doi.org/10.2478/cmam-2011-0021)),
#  - `'amen'`: AMEn ([Dolgov and Savostyanov, 2014](https://doi.org/10.1137/140953289)),
#  - `'swap'`: swapping of neighbouring cores ([Michailidis, Fenton and Kiffner, 2025](https://doi.org/10.1137/24M1714149)).
# The documentation of `torchtt.methods` describes which method to use when. The following function compares them.
def compare(A, x, methods):
    """
    Compute the matrix vector product with each method and print the runtime, the rank of the result and the relative error.
    """
    y_ref = (A @ x).round(1e-12)
    for method in methods:
        tme = datetime.datetime.now()
        y = tntt.matvec(A, x, eps=1e-10, method=method)
        tme = datetime.datetime.now() - tme
        print('%-7s time %s   rank %3d   relative error %.1e' % (method, tme, max(y.R), ((y - y_ref).norm() / y_ref.norm()).item()))

# Create a random TT matrix and a random TT tensor.
n = 4 # mode size
A = tntt.random([(n,n)]*8,[1]+7*[4]+[1]) # random TT matrix
x = tntt.random([n]*8,[1]+7*[5]+[1]) # random TT tensor

# Increase the ranks without adding information.
# The product of the ranks is now 24 * 40 = 960, but the result can be compressed to rank 20 (the product is equivalent to $8\mathbf{\mathsf{Ax}}$).
# The methods that do not form the exact product (DMRG and AMEn) are faster than `'direct'` in this case.
A_large = A + A + A + A - A - A
x_large = x + x + x + x + x + x - x - x
print('Ranks of A:', A_large.R)
print('Ranks of x:', x_large.R)
compare(A_large, x_large, ['direct', 'dmrg', 'amen', 'swap'])

# If the result cannot be compressed, forming the exact product is the fastest.
# For random tensors, the product of the ranks is also the rank of the result (here 4 * 5 = 20).
compare(A, x, ['direct', 'dmrg', 'amen', 'swap'])

# The parameters of a method are changed by passing an instance of its class from `torchtt.methods` instead of its name.
# The iterative methods also accept an initial guess of the result.
y = tntt.matvec(A_large, x_large, eps=1e-10, method=tntt.methods.DMRG(nswp=10, kickrank=2))
y = tntt.matvec(A_large, x_large, eps=1e-10, method=tntt.methods.AMEn(kickrank=8), initial=y)

# The same holds for the elementwise product and the product of TT matrices.
z = tntt.hadamard(x_large, x_large, eps=1e-10, method='dmrg')
B = tntt.matmat(A_large, A_large, eps=1e-10, method='direct')

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
