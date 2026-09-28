r"""
# Tensor completion with Riemannian optimization

Tensor completion recovers a tensor from a small subset of its entries, assuming that the tensor has a low tensor-train (TT) rank.
Given noisy observations $y_m \approx \mathsf{x}_{\mathbf{i}_m}$ of the entries at the multi-indices $\mathbf{i}_1, \dots, \mathbf{i}_M$, we minimize the misfit

$$ f(\mathsf{x}) = \sum_{m=1}^{M} \big( \mathsf{x}_{\mathbf{i}_m} - y_m \big)^2 $$

over the set $\mathcal{M}_{\mathbf{r}}$ of TT tensors with a fixed TT rank $\mathbf{r}$.
This set is a smooth manifold, so we use Riemannian gradient descent ([Steinlechner, 2016](https://doi.org/10.1137/15M1010506)): the gradient is projected onto the tangent space at the current iterate, a step is taken along it, and TT rounding to rank $\mathbf{r}$ brings the result back onto the manifold.
The [manifold notebook](manifold.ipynb) introduces the Riemannian gradient in more detail.
"""

#%% Imports
import datetime
import numpy as np
import torch as tn
import torchtt as tntt

#%% Test problem
# The tensor to recover samples the function $g(x_1, x_2, x_3, x_4) = 1 + x_1 + x_2 + x_3 + x_4 + x_1 x_2 + x_2 x_3 + \sin(x_1)$ on a uniform $20 \times 20 \times 20 \times 20$ grid of $[0,1]^4$.
# It has 160,000 entries but a small TT rank.
tn.manual_seed(0)  # reproducible observations and initial guess

N = 20
Xs = tntt.meshgrid([tn.linspace(0, 1, N, dtype=tn.float64)]*4)
target = Xs[0]+1+Xs[1]+Xs[2]+Xs[3]+Xs[0]*Xs[1]+Xs[1]*Xs[2]+tntt.TT(tn.sin(Xs[0].full()))
target = target.round(1e-10)
print('TT rank of the target:', target.R)

#%% Observations
# We observe $M = 10000$ randomly chosen entries (about 6% of the tensor), perturbed by Gaussian noise with standard deviation $10^{-3}$.
# The method `apply_mask()` evaluates a TT tensor at a list of multi-indices.
M = 10000  # number of observations
indices = tn.randint(0, N, (M, 4))

# the observations are noisy
sigma_noise = 0.001
obs = tn.normal(target.apply_mask(indices), sigma_noise)

# squared misfit on the observed entries
loss = lambda x: (x.apply_mask(indices)-obs).norm()**2

#%% Riemannian gradient descent
# We start from a random TT tensor of rank $(1,3,3,3,1)$.
# In every iteration, `torchtt.manifold.riemannian_gradient()` computes the Riemannian gradient $\xi$ of the loss using automatic differentiation.
# Since the loss is quadratic, the step size along $-\xi$ that minimizes it can be computed exactly:
# $$ \alpha = \frac{\langle P_\Omega \xi,\, P_\Omega \mathsf{x} - y \rangle}{\| P_\Omega \xi \|^2}, $$
# where $P_\Omega$ keeps only the observed entries.
# Rounding the updated tensor to the rank of the iterate retracts it back onto the manifold.
x = tntt.randn([N]*4, [1, 3, 3, 3, 1])  # initial guess

tme = datetime.datetime.now()
for i in range(1000):
    # Riemannian gradient of the loss
    grad = tntt.manifold.riemannian_gradient(x, loss)

    # exact line search along -grad (the loss is quadratic)
    residual = x.apply_mask(indices) - obs
    grad_obs = grad.apply_mask(indices)
    step_size = (grad_obs @ residual) / (grad_obs @ grad_obs)

    # step and retraction onto the manifold of tensors with rank x.R
    x = (x - step_size * grad).round(0, x.R)

    if (i+1) % 100 == 0:
        print('Iteration %4d  loss %e  relative error %e' % (i+1, loss(x), (x-target).norm()/target.norm()))
tme = datetime.datetime.now() - tme
print('Time elapsed', tme)

#%% Results
# The loss levels off at about $M \sigma^2 = 10^{-2}$, the contribution of the noise, and the completed tensor matches the whole target, including the 94% of entries that were never observed.
# The number of unknowns (the entries of the TT cores) is much smaller than the number of observations.
print('Number of observations %d, tensor shape %s, percentage of entries observed %.2f%%' % (M, str(x.N), 100*M/np.prod(x.N)))
print('Number of unknowns %d, number of observations %d, unknowns/observations %.3f' % (tntt.numel(x), M, tntt.numel(x)/M))
print('Relative error of the completed tensor %e' % ((x-target).norm()/target.norm()))
