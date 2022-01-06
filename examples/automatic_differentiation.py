import torch as tn
import tnt
import tnt.manifold 
import datetime 
# import matplotlib.pyplot as plt 
import numpy as np
import tnt.grad

n = 64
A1d = -2*np.eye(n)+np.eye(n,k=-1)+np.eye(n,k=1)
A1d = tnt.TT([tn.tensor(A1d.reshape([1,n,n,1]),dtype=tn.float64)])


x = tnt.ones([n,n])



tnt.grad.watch(x)
y = ((A1d**tnt.eye([n]))@x)
y = y.sum()
dy_dx = tnt.grad.grad(y,x)



x = tnt.random([40]*5, [1,9,9,9,9,1])
# x = random([10,11,12,13],[1,2,3,4,1])
# x = random([5,6], [1,2,1])

N = [5,6,5]
R = [1,4,4,1]
R2 = [1,4,4,1]

# N = [10,12,13,14,10,9]
N = [32,31,32,33,34,35]
R = [1,3,4,6,3,4,1]
R2 = [1,3,4,6,3,4,1]





target = tnt.random(N,R)#+1e-2*tnt.random(N,[1]*(len(N)+1))

func = lambda x: 0.5*(x-target).norm(True)
# func = lambda x: 0.5*(x+(-1)*one).norm()**2
# func = lambda x : TT.dot(x,x)
x0 = tnt.random(N,R).round(1e-19)
eta = 0.5
beta = 0.01
log = []

x = x0.round(0)
xn = x0.round(0)



print('Value ' , func(x).numpy())
log.append([func(x).numpy(),func(xn).numpy()])

for i in range(20):
    # compute riemannian gradient using AD    
    gr = tnt.manifold.riemannian_gradient(x,func)
    
    # gradient
    gn = (x-target)
    # project gradient n riemannian manifold
    grr = tnt.manifold.riemannian_projection(x,gn)
   
    # print((grr-gr).norm())
    # stepsize
    p = -grr
    gr = grr
    #stepsize length
    alpha = 1.0
        
    # while func(xn+alpha2*(-1)*gn)>func(xn)-alpha2*beta*TT.dot(gn,gn) and alpha2 > 1e-8:
    #     alpha2 *= eta
    # alpha = grr.norm()/np.sqrt(len(grr.N))*0.1
    
    # update step
    x = (x-alpha*gr).round(0,R)
    # xn = (xn+(-1)*alpha2*gn).round_tt(0,R)
    
    print()
    print('Value ' , func(x).numpy(),' step size ',alpha)
    log.append([func(x).numpy(),func(xn).numpy()])
    
    
    
#log = np.array(log)
#plt.figure()
#plt.semilogy(np.arange(log.shape[0]),log[:,0])
#plt.semilogy(np.arange(log.shape[0]),log[:,1])


