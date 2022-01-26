"""
# GPU acceleration

The package `torchtt` can use the built-in GPU acceleration from `pytorch`.
"""

#%% Imports and check if any CUDA device is available.
import datetime
import torch as tn
try: 
    import torchtt as tntt
except:
    print('Installing torchTT...')
    # %pip install git+https://github.com/ion-g-ion/torchTT
    
print('CUDA available:',tn.cuda.is_available())
print('Device name: ' + tn.cuda.get_device_name())

#%% Define a function to test. It performs 2 matrix vector products in TT-format and a rank rounding. 
# The return result is a scalar.

def f(x,A,y):
    """
    fonction that performs operations of tensors in TT.

    Args:
        x (tnt.TT): input TT tensor
        A (tnt.TT): input TT matrix
        y (tnt.TT): input TT tensor

    Returns:
        torch.tensor: result
    """
    z = A @ y + A @ y # operatio that grows the rank
    z = z.round(1e-12) # rank rounding (contains QR and SVD decomposition)
    z += z*x # some other operation
    return tntt.dot(x,z) # contract the tensor

#%% Generate random tensors in the TT-format (on the CPU).
x = tntt.random([200,300,400,500],[1,10,10,10,1])
y = tntt.random([200,300,400,500],[1,8,8,8,1])
A = tntt.random([(200,200),(300,300),(400,400),(500,500)],[1,8,8,8,1])

#%% Run the function f() and report the time.
tme_cpu = datetime.datetime.now()
f(x,A,y) 
tme_cpu = datetime.datetime.now() - tme_cpu
print('Time on CPU: ',tme_cpu)

#%% Move the defined tensors on GPU. Similarily to pytorch tensors one can use the function cuda() to return a copy of a TT instance on the GPU. 
# All the cores of the returned TT object are on the GPU.
x = x.cuda()
y = y.cuda()
A = A.cuda()

#%% The function is executed once without timing to "warm-up" the CUDA.
f(x*0,A*0,0*y).cpu()

#%% Run the function again. This time the runtime is reported. 
# The return value is moved to CPU to assure blocking until all computations are done.
tme_gpu = datetime.datetime.now()
f(x,A,y).cpu()
tme_gpu = datetime.datetime.now() - tme_gpu
print('Time with CUDA: ',tme_gpu)

#%% The speedup is reported
print('Speedup: ',tme_cpu.total_seconds()/tme_gpu.total_seconds(),' times.')

#%% This time we perform the same test without using the rank rounding. 
# The expected result is better since the rank rounding contains QR and SVD which are not that parallelizable.

def g(x,A,y):
    """
    fonction that performs operations of tensors in TT.

    Args:
        x (tnt.TT): input TT tensor
        A (tnt.TT): input TT matrix
        y (tnt.TT): input TT tensor

    Returns:
        torch.tensor: result
    """
    z = A @ y + A @ y # operatio that grows the rank
    z += z+x # some other operation
    return tntt.dot(x,z) # contract the tensor

# put tensors on CPU
x, y, A = x.cpu(), y.cpu(), A.cpu()
# perform the test
tme_cpu = datetime.datetime.now()
g(x,A,y) 
tme_cpu = datetime.datetime.now() - tme_cpu
print('Time on CPU: ',tme_cpu)
# move the tensors back to GPU
x, y, A = x.cuda(), y.cuda(), A.cuda()
# execute the function
tme_gpu = datetime.datetime.now()
g(x,A,y).cpu()
tme_gpu = datetime.datetime.now() - tme_gpu
print('Time with CUDA: ',tme_gpu)

print('Speedup: ',tme_cpu.total_seconds()/tme_gpu.total_seconds(),' times.')

#%% A tensor can be copied to a differenct device using the to() method. Usage is similar to torch.tensor.to()
dev = tn.cuda.current_device()
x_cuda = x.to(dev)
x_cpu = x_cuda.to(None)