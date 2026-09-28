r"""
# Random tensors in the TT format

`torchtt.randn(N, R, var)` returns a random tensor in the tensor-train (TT) format with shape `N` and TT rank `R`.
The entries of the TT cores are independent and normally distributed, and they are scaled so that every entry of the full tensor has mean 0 and variance `var`.
An entry of the full tensor is a sum of products of the core entries, so it is only approximately normally distributed.

This notebook checks the variance for a few shapes and ranks by computing the full tensors.
"""

#%% Imports
import torch as tn
import torchtt as tntt

#%% Variance 1
# A 5-dimensional tensor of shape $30 \times \cdots \times 30$ with TT rank $(1,8,16,16,8,1)$:
x = tntt.randn([30]*5, [1,8,16,16,8,1])
x_full = x.full()
print('Var = ', tn.std(x_full).numpy()**2, ' (has to be comparable to 1.0)')

#%% Variance 4
# The same shape and rank with `var = 4.0`:
x = tntt.randn([30]*5, [1,8,16,16,8,1], var = 4.0)
x_full = x.full()
print('Var = ', tn.std(x_full).numpy()**2, ' (has to be comparable to 4.0)')

#%% Variance 0.01
# The same shape and rank with `var = 0.01`:
x = tntt.randn([30]*5, [1,8,16,16,8,1], var = 0.01)
x_full = x.full()
print('Var = ', tn.std(x_full).numpy()**2, ' (has to be comparable to 0.01)')

#%% Longer train
# A 7-dimensional tensor of shape $10 \times \cdots \times 10$ with all inner ranks equal to 4:
x = tntt.randn([10]*7, [1,4,4,4,4,4,4,1], var = 1.0)
x_full = x.full()
print('Var = ', tn.std(x_full).numpy()**2, ' (has to be comparable to 1.0)')

#%% Remark
# The variance of the entries equals `var` in expectation, but the empirical variance of a single tensor fluctuates around it.
# All entries of a TT tensor are built from the same cores, so they are not independent, and the fluctuations are larger than for a dense tensor with independent entries.
