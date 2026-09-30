"""
# GPU acceleration

The package `torchtt` can use the built-in GPU acceleration from `pytorch`.
"""

#%% Imports and check if any CUDA device is available.
import time
import torch as tn
try: 
    import torchtt as tntt
except:
    print('Installing torchTT...')
    # %pip install git+https://github.com/ion-g-ion/torchTT
    
print('CUDA available:',tn.cuda.is_available())
print('Device name: ' + tn.cuda.get_device_name())

#%% Define a function to test. It performs 2 matrix vector products in the tensor-train (TT) format and a rank rounding. 
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
    z = z.round(1e-12) # rank rounding (contains QR and singular value decompositions)
    z += z*x # some other operation
    return tntt.dot(x,z) # contract the tensor

#%% Generate random tensors in the TT-format (on the CPU).
x = tntt.random([200,300,400,500],[1,10,10,10,1])
y = tntt.random([200,300,400,500],[1,8,8,8,1])
A = tntt.random([(200,200),(300,300),(400,400),(500,500)],[1,8,8,8,1])

#%% Warm up and report the average time over five runs.
n_runs = 5
f(x,A,y) # warm up
tme_cpu = time.perf_counter()
for _ in range(n_runs):
    f(x,A,y)
tme_cpu = (time.perf_counter() - tme_cpu) / n_runs
print('Average time on CPU: ',tme_cpu,' seconds')

#%% Move the defined tensors on GPU. Similarily to pytorch tensors one can use the function cuda() to return a copy of a TT instance on the GPU. 
# All the cores of the returned TT object are on the GPU.
cuda_name = 'cuda:0'
x = x.to(cuda_name)
y = y.to(cuda_name)
A = A.to(cuda_name)

#%% Warm up CUDA. Synchronization below waits for completion.
result_gpu = f(x,A,y)

#%% Synchronize once before the loop and at the end of each iteration.
# Report the average runtime.
# Input transfers, warm-up, and the final CPU copy are outside the timer.
tn.cuda.synchronize(cuda_name)
tme_gpu = time.perf_counter()
for _ in range(n_runs):
    result_gpu = f(x,A,y)
    tn.cuda.synchronize(cuda_name)
tme_gpu = (time.perf_counter() - tme_gpu) / n_runs
result_cpu = result_gpu.cpu()
print('Average time with CUDA: ',tme_gpu,' seconds')

#%% The speedup is reported
print('Speedup: ',tme_cpu/tme_gpu,' times.')

#%% This time we perform the same test without using the rank rounding. 
# The expected result is better since the rank rounding contains QR decompositions and singular value decompositions (SVD), which are not that parallelizable.

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
g(x,A,y) # warm up
tme_cpu = time.perf_counter()
for _ in range(n_runs):
    g(x,A,y)
tme_cpu = (time.perf_counter() - tme_cpu) / n_runs
print('Average time on CPU: ',tme_cpu,' seconds')
# move the tensors back to GPU
x, y, A = x.to(cuda_name), y.cuda(cuda_name), A.cuda(cuda_name)
# execute the function
result_gpu = g(x,A,y) # warm up
tn.cuda.synchronize(cuda_name)
tme_gpu = time.perf_counter()
for _ in range(n_runs):
    result_gpu = g(x,A,y)
    tn.cuda.synchronize(cuda_name)
tme_gpu = (time.perf_counter() - tme_gpu) / n_runs
result_cpu = result_gpu.cpu()
print('Average time with CUDA: ',tme_gpu,' seconds')

print('Speedup: ',tme_cpu/tme_gpu,' times.')

#%% A tensor can be copied to a differenct device using the to() method. Usage is similar to torch.tensor.to()
dev = tn.cuda.current_device()
x_cuda = x.to(dev)
x_cpu = x_cuda.to(None)