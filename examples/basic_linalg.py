"""
# Basic linear algebra in torchTT


This notebook is an introduction into the basic linar algebra operations that can be perfromed using the `torchtt` package.
The basic operations such as +,-,*,@,norm,dot product can be performed between `torchtt.TT` instances without computing the full format by computing the TT cores of the result.
One exception is the elementwise division between TT objects. For this, no explicit form of the resulting TT cores can be derived and therefore optimization techniques have to be employed (see the notebook `fast_tt_operations.ipynb`).
"""

#%% Imports
import torch as tn
import torchtt as tntt 


#%% We will create a couple of tensors for the opperations that follow
N = [10,10,10,10]
o = tntt.ones(N)
x = tntt.randn(N,[1,4,4,4,1])
y = tntt.TT(tn.reshape(tn.arange(N[0]*N[1]*N[2]*N[3], dtype = tn.float64),N))
A = tntt.randn([(n,n) for n in N],[1,2,3,4,1])
B = tntt.randn([(n,n) for n in N],[1,2,3,4,1])


#%% Addition
# The TT class has the "+" operator implemeted. It performs the addition between TT objects (must have compatible shape and type) and it returns a TT object. One can also add scalars to a TT object (float/int/torch.tensor with 1d).
# The TT rank of the result is the sum of the ranks of the inputs. This is usually an overshoot and rounding can decrease the rank while maintaining the accuracy.
# Here are a few examples:
z = x+y 
print(z)
# adding scalars is also possible
z = 1+x+1.0
z = z+tn.tensor(1.0)
# it works for the TT amtrices too
M = A+A+1 
print(M)

# Broadcasting is also available and is similar to the `PyTorch` [broadcasting](https://pytorch.org/docs/stable/notes/broadcasting.html). 
# Tensors in the TT-format can be added even if their shapes are different. The rule is that the number of dimensions of the first operand must be greater or equal to the number of dimensions of the second operand. 
# In the following example a `(4,5)` tensor is added to a `(2,3,4,5)` tensor:
xx = tntt.random([2,3,4,5],[1,2,3,4,1])
yy = tntt.random([4,5],[1,2,1])
print(xx+yy)
#The mode sizes should match starting from the end or the mode size of the second tensor can be 1:
xx = tntt.random([2,3,4,5],[1,2,3,4,1])
yy = tntt.random([1,1,4,5],[1,2,2,2,1])
print(xx+yy)

#%% Subtraction
# The "-" operator is also implemented in  the `torchtt.TT` class. It can be used similarily to "+" between 2 `torchtt.TT` objects and between a `torchtt.TT` and a scalar.
# It can also be used as a negation.
v = x-y-1-0.5
C = A-B-3.14
w = -x+x
print(tn.linalg.norm(w.full()))
# Broadcasting is available for the "-" operation as well.

#%% Multiplication (elementwise)¶
# One can perform the elementwise multiplication $\mathsf{u}_{i_1...i_d} = \mathsf{x}_{i_1...i_d} \mathsf{y}_{i_1...i_d}$ between 2 tensors in the TT format without goin to full format. 
# The main issues of this is that the rank of the result is the product of the ranks of the input TT tensors.
u = x*y
print(u)
M2 = A*A
# Broadcasting is available for the "*" operation as well.


#%% Matrix vector product and matrix matrix product
# * TT matrix and TT tensor: $(\mathsf{Ax})_{i_1...i_d} = \sum\limits_{j_1...j_d}\mathsf{A}_{i_1...i_d,j_1...j_d} \mathsf{x}_{j_1...j_d}$
# * TT matrix and TT matrix: $(\mathsf{AB})_{i_1...i_d,k_1...k_d} = \sum\limits_{j_1...j_d}\mathsf{A}_{i_1...i_d,j_1...j_d} \mathsf{B}_{j_1...j_d,k_1...k_d}$
print(A@x)
print(A@B)
print(A@B@x)

# Multiplication can be performed between a TT operator and a full tensor (in torch.tensor format) the result in this case is a full tn.tensor
print(A@tn.rand(A.N, dtype = tn.float64))


#%% Kronecker product
# For computing the Kronecker product one can either use the "**" operator or the method torchtt.kron().
print(x**y)
print(A**A)


#%% Norm
# Frobenius norm of a tensor $||\mathsf{x}||_F^2 = \sum\limits_{i_1,...,i_d} \mathsf{x}_{i_1...i_d}$ can be directly domputed from a TT decomposition.
print(y.norm())
print(A.norm())


#%% Dot product and summing along modes
# One can sum alonf dimensions in torchtt. The function is torchtt.TT.sum() and can be used without arguments to sum along all dimensions, returning a scalar:
print('sum() result ', y.sum())
print('Must be equal to ', tn.sum(y.full()))

# If a list of modes is additionally provided, the summing will be performed along the given modes and a torchtt.TT object is returned.
print(x.sum(1))
print(x.sum([0,1,3]))
print(A.sum([1,2]))

# Dot product between 2 tensors is also possible using the function tirchtt.dot().
print(tntt.dot(y,y))

# Dot product can be performed between 2 tensors of different mode lengths. The modes alonnd the dot product is performed must be equal. And they are given as a list of integers as an additional argument. The modes given are relative to the first tensor. The returned value is a torchtt.TT instance.
t1 = tntt.randn([4,5,6,7,8,9],[1,2,4,4,4,4,1])
t2 = tntt.randn([5,7,9],[1,3,3,1])
print(tntt.dot(t1,t2,[1,3,5]))


#%% Reshaping 
# Given a tensor in the TT format, one can reshape it similarily as in pytorch or numpy. 
# The method is torchtt.reshape() and it taks as argument a torchtt.TT object, the new shape, the relative accuracy epsilon and a maximum rank. The last 2 are optional. 
# The method also performs rounding up to the desired accuracy.
q = tntt.TT(tn.reshape(tn.arange(2*3*4*5*7*3, dtype = tn.float64),[2,3,4,5,7,3]))
# perform a series of reshapes
w = tntt.reshape(q,[12,10,21])
print(w)
w = tntt.reshape(w,[360,7])
print(w)
w = tntt.reshape(w,[2,3,4,5,7,3])
print('Error ',(w-q).norm()/q.norm())

# Reshape works also for TT matrices. However there are some restrictions such as the merging or spliting of the dimensions must happen within the same core for both row/column indices.
A = tntt.randn([(4,8),(6,4),(5,6),(8,8)],[1,2,3,2,1])
B = tntt.reshape(A,[(2,4),(6,4),(10,12),(8,8)])
print(B)
B = tntt.reshape(B,[(60,32),(16,48)])
print(B)
B = tntt.reshape(B,[(4,8),(6,4),(5,6),(8,8)])
print('Error ',(B-A).norm()/A.norm())

# this will not work: tntt.reshape(A,[(24,4),(5,16),(8,24)])

