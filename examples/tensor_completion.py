#%% Imports
import torchtt as tntt
import torch as tn
import numpy as np
import datetime

#%% Preparation
# create a random tensor 
N = 20
target = tntt.random([N]*4,[1,4,5,3,1])

M = 10000 # number of observations 
indices = tn.randint(0,N,(M,4))

# observations are considered to be noisy
sigma_noise = 0.001
obs = tn.normal(target.apply_mask(indices), sigma_noise)

# define the loss function
loss = lambda x: (x.apply_mask(indices)-obs).norm()**2

#%% Manifold learning
print('Riemannian gradient descent\n')
# starting point
x = tntt.random([N]*4,[1,5,5,5,1])

tme = datetime.datetime.now()
# iterations
for i in range(25):
    # manifold gradient 
    gr = tntt.manifold.riemannian_gradient(x,loss)

    step_size = 1.0
    R = x.R
    # step update
    x = (x - step_size * gr).round(0,R)

    # compute loss value
    if (i+1)%10 == 0:
        loss_value = loss(x)
        print('Iteration %4d loss value %e error %e'%(i+1,loss_value.numpy(),(x-target).norm()/target.norm()))

tme = datetime.datetime.now() - tme
print('')
print('Time elapsed',tme)
print('Number of observations %d, tensor shape %s, percentage of entries observed %6.4f'%(M,str(x.N),100*M/np.prod(x.N)))
print('Number of unknowns %d, number of observations %d, DoF/observations %.6f'%(tntt.numel(x),M,tntt.numel(x)/M))

print('Rank after rounding',x.round(1e-6))

#%% Classical gradient descent w.r.t. TT-cores
# x = tnt.random([N]*4,[1,5,5,5,1])
# 
# for i in range(100):
#     tnt.grad.watch(x)
#     loss_val =loss(x)
#     cores_update = tnt.grad.grad(loss_val,x)
#     tnt.grad.unwatch(x)
#     x = tnt.TT([c1-0.015*c2 for c1,c2 in zip(x.cores,cores_update)])
# 
#     print('Iteration %4d loss value %e error %e'%(i+1,loss_val.detach().numpy(),(x-target).norm()/target.norm()))


