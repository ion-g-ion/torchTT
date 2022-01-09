#%% Imports 
import torch as tn
import torchtt as tntt

#%% Variance 1.0
x = tntt.randn([30]*5,[1,8,16,16,8,1])
x_full = x.full()
print('Var = ',tn.std(x_full).numpy()**2,' (has to be comparable to 1.0)')

#%% Variance 4.0
x = tntt.randn([30]*5,[1,8,16,16,8,1],var = 4.0)
x_full = x.full()
print('Var = ',tn.std(x_full).numpy()**2,' (has to be comparable to 4.0)')

#%% Variance 0.01
x = tntt.randn([30]*5,[1,8,16,16,8,1], var = 0.01)
x_full = x.full()
print('Var = ',tn.std(x_full).numpy()**2,' (has to be comparable to 0.01)')

#%% Variance 1.0 (longer train)
x = tntt.randn([10]*7, [1,4,4,4,4,4,4,1], var = 1.0)
x_full = x.full()
print('Var = ',tn.std(x_full).numpy()**2,' (has to be comparable to 1.0)')
