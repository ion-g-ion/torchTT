"""
Basic tutorial 

This notebook is a tutorial on how to use the basic functionalities of the `torchtt` package. 
"""

#%% Imports

import torch as tn
import torchtt as tntt 


#%% Decomposition of a full tensor in TT format
# We now create a 4d `torch.tensor` which we will use later
tens_full = tn.reshape(tn.arange(32*16*8*10, dtype = tn.float64),[32,16,8,10])

# The TT approximation of a given tensor is $\mathsf{x}_{i_1i_2...i_d} \approx \sum\limits_{r_1,...,r_{d-1}=1}^{R_1,...,R_{d-1}} \mathsf{g}^{(1)}_{1i_1r_1}\cdots\mathsf{g}^{(d)}_{r_{d-1}i_d1} $. Using the constructor `torchtt.TT()` a full tensor can be decomposed in the TT format. The `dtype` of the input will be passed to the TT cores.
tens_tt = tntt.TT(tens_full)

# The newly instantiated object contains the cores as a list, the mode sizes and the rank.
print('TT cores', tens_tt.cores)
print('Mode size ', tens_tt.N)
print('TT rank ', tens_tt.R)

# Since the TT decomposition is not exact in most of the cases, an approximation is made. If the argument `eps` is provided to the `torchtt.TT()` function the decomposition can be performed upt to the given relative accuracy.
# Moreover the maximum rank can also be provided as the argument `rmax`.
tens_full2 = tens_full+1e-5*tn.randn(tens_full.shape, dtype=tens_full.dtype)
tens_tt2 = tntt.TT(tens_full2, eps = 1e-4)
print(tens_tt2.R)

# The original tensor can be recovered using the `torchtt.TT.full()` method (also check if it equals the original full tensor):
tens_full_rec = tens_tt.full()
print(tn.linalg.norm(tens_full-tens_full_rec)/tn.linalg.norm(tens_full))

# Using the print() function, information about the newly created torchtt.TT instance can be displayed:
print(tens_tt)

#%% Tensor operators
# As a generalization of the matrix vector algebra, one can define tensor operators that act on tensors. If the tensor si $d$-dimensional, the tensor operator will be $2d$-dimensional.
# The goal is to perform a product $\mathsf{Ax}\in\mathbb{M_1\times \cdots \times M_d}$ between a tensor $\mathsf{x}\in\mathbb{R}^{N_1\times \cdots \times N_d}$ and the operator $\mathsf{A}\in \mathbb{R}^{(M_1\times \cdots \times M_d)\times(N_1\times \cdots \times N_d)}$. For the operators the following TT matrix format is used $\mathsf{A}_{i_1...i_d,j_1...j_d}\approx \sum\limits_{r_1,...,r_{d-1}=1}^{R_1,...,R_{d-1}} \mathsf{g}^{(1)}_{1i_1j_1r_1}\cdots\mathsf{g}^{(d)}_{r_{d-1}i_dj_d1}$.

# If a tensor operator needs to be decomposed from full, the additional argument `shape` of the `torchtt.TT()` constructor has to be used to provide the shape.
# If the tensor operator has the shape $(M_1\times \cdots \times M_d)\times(N_1\times \cdots \times N_d)$ the argument must be passed as `[(M1,N1),(M2,N2),(M3,N2),...]`.

A_full = tn.reshape(tn.arange(8*4*6*3*7*9, dtype = tn.float64),[8,4,6,3,7,9])
# create an instance of torchtt.TT
A_ttm = tntt.TT(A_full, eps = 1e-12, shape = [(8,3),(4,7),(6,9)])


#%% Slicing
# Slicing operation can be performed on a tensor in TT format. 
# If all the dimensions are indexed with an integer and the multiindices are valid, a torch.tensor with the corresponding value is returned.
# Slices can be also used, however the returned object in this case is again a torchtt.TT instance.

print(tens_tt[1,2,3,4])
print(tens_tt[1,1:4,2,:])


#%% TT rank rounding
# In some cases the TT rank becomes too large and a reduction is desired. The goal is to perform a reduction of the rank while maintaining an accuracy.
# The problem statement of the rounding operation is: given a tensor $\mathsf{x}$ in the TT format with the TT rank $\mathbf{R}$ and an $\epsilon>0$, find a tensor $\tilde{\mathsf{x}}$ with TT rank $\tilde{\mathbf{R}}\leq \mathbf{R}$ such that $ ||\mathsf{x}-\tilde{\mathsf{x}}||_F\leq \epsilon || \mathsf{x} ||_F$.
# This is implemented using the member method of a TT object `torchtt.TT.round()`. The argument `epsilon` is passed to the function as well as the optional argument `rmax` which also restricts the rank of the rounding.

#We will create a tensor of TT rank $(1,6,6,6,1)$ in the TT format.

t1 = tntt.randn([10,20,30,40],[1,2,2,2,1])
t2 = tntt.randn([10,20,30,40],[1,2,2,2,1])
t3 = tntt.randn([10,20,30,40],[1,2,2,2,1])
t1, t2, t3 = t1/t1.norm(), t2/t2.norm(), t3/t3.norm()
tt = t1+1e-3*t2+1e-6*t3
t_full = tt.full()
print(tt)

# Rounding the tensor to a relative `epsilon` of 1e-5 yields.
tt1 = tt.round(1e-5)
print(tt1)
print('Error ',tn.linalg.norm(tt1.full()-tt.full())/tn.linalg.norm(tt.full()))

# This is equivalent to removing the t3 from tt and the error will be less than 1e-6.
# If a truncation with epsilon=1e-2 is done, the resulting tensor will have the rank [1,2,2,2,1].
tt1 = tt.round(1e-2)
print(tt1)
print('Error ',tn.linalg.norm(tt1.full()-tt.full())/tn.linalg.norm(tt.full()))

# The maximum rank of a truncation can also be provided as argument.
tt3 = tt.round(1e-12,2)
print(tt3)
tt4 = tt.round(1e-12,[1,2,3,2,1])
print(tt4)


#%% Special tensors
# Some tensors can be directly constructed in the TT format: the one tensor, the zeros tensor, the identity tensor operator adn random tensors with a given rank.

# The one tensor can be created directly in the TT format using torchtt.ones().
print(tntt.ones([2,3,4]).full())

# The zero tensor ca be created in the TT format using torchtt.zeros().
print(tntt.zeros([2,3,4]).full())

# The identity tensor operator is created using torchtt.eye().
print(tntt.eye([10,20,30]))

# Tensors with random TT cores and a given rank can be created with torchtt.random().
print(tntt.random([3,4,5,6,7],[1,2,5,5,2,1]))
print(tntt.random([(3,7),(4,6),(5,5),(6,10),(7,2)],[1,2,5,5,2,1]))

# Random tensors with a given rank and random entries with expected value 0 and given variance can be created using torchtt.randn().
# Variance 1.0
x = tntt.randn([30]*5,[1,8,16,16,8,1])
x_full = x.full()
print('Var = ',tn.std(x_full).numpy()**2,' (has to be comparable to 1.0)')

# Variance 4.0
x = tntt.randn([30]*5,[1,8,16,16,8,1],var = 4.0)
x_full = x.full()
print('Var = ',tn.std(x_full).numpy()**2,' (has to be comparable to 4.0)')

# Variance 0.01
x = tntt.randn([30]*5,[1,8,16,16,8,1], var = 0.001)
x_full = x.full()
print('Var = ',tn.std(x_full).numpy()**2,' (has to be comparable to 0.001)')

# Variance 1.0 (longer train)
x = tntt.randn([10]*7, [1,4,4,4,4,4,4,1], var = 1.0)
x_full = x.full()
print('Var = ',tn.std(x_full).numpy()**2,' (has to be comparable to 1.0)')