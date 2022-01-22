
import torch as tn
import torchtt as tntt


N = [10,11,12,13,14]
Rt = [1,3,4,5,6,1]
Rx = [1,6,6,6,6,1]
target = tntt.randn(N,Rt).round(0)
func = lambda x: 0.5*(x-target).norm(True)


x0 = tntt.randn(N,Rx)
x =x0.clone()
for i in range(20):
    # compute riemannian gradient using AD    
    gr = tntt.manifold.riemannian_gradient(x,func)
    
    #stepsize length
    alpha = 1.0
    
    # update step
    x = (x-alpha*gr).round(0,Rx)    
    print('Value ' , func(x).numpy())

y = x0.detach().clone()


for i in range(10000):
    tntt.grad.watch(y)
    fval = func(y)
    deriv = tntt.grad.grad(fval,y)

    
    alpha = 0.00001 # for stability
    y = tntt.TT([y.cores[i].detach()-alpha*deriv[i] for i in range(len(deriv))])
    print(func(y))