#%% Imports
import torchtt as tntt
import torch as tn
import datetime

print('CUDA available:',tn.cuda.is_available())
print('Device name: ' + tn.cuda.get_device_name())

#%% Define the test function: it contains several linear algebra operations in the TT-format together with one rounding operation.

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
    z += z+x # some other operation
    return tntt.dot(x,z) # contract the tensor


#%% Without CUDA

# generate 3 random TT instances

x = tntt.random([200,300,400,500],[1,8,8,8,1])
y = tntt.random([200,300,400,500],[1,8,8,8,1])
A = tntt.random([(200,200),(300,300),(400,400),(500,500)],[1,8,8,8,1])

# call the function once 
f(x,A,y)

# call the function again. This time with timing.
tme_cpu = datetime.datetime.now()
f(x,A,y) 
tme_cpu = datetime.datetime.now() - tme_cpu
print('Time without CUDA: ',tme_cpu)

#%% CUDA

x = x.cuda()
y = y.cuda()
A = A.cuda()

# call the function once. For CUDA a warmup is necessary
f(x,A,y).cpu()

# do it one more time with timing
tme_gpu = datetime.datetime.now()
f(x,A,y).cpu()
tme_gpu = datetime.datetime.now() - tme_gpu
print('Time with CUDA:    ',tme_gpu)

# print the speedup
print('Speedup:           ',tme_cpu.total_seconds()/tme_gpu.total_seconds())

#%% New test
N = [2, 2, 2, 3, 5, 2, 2, 2, 3, 5, 2, 2, 2, 3, 3, 7, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5, 2, 5]
R = [1,2,4,8,9,9,10,11,11,12,12,15,18,23,40,42,50,54,54,51,55,61,45,52,36,42,26,32,18,22,12,10,1]
x = tntt.random(N,R)
y = tntt.random(N,R)

def g(x,y):
    """
    Another function to test (just multipleication)

    Args:
        x (tnt.TT): input TT tensor
        y (tnt.TT): input TT tensor

    Returns:
        torch.tensor: result
    """
    z =2*x*y+x+y
    return z.sum()

# call the function once 
# g(x,y)

# call the function again. This time with timing.
tme_cpu = datetime.datetime.now()
g(x,y) 
tme_cpu = datetime.datetime.now() - tme_cpu
print('Time without CUDA: ',tme_cpu)


#%% CUDA

x = x.cuda()
y = y.cuda()

# call the function once. For CUDA a warmup is necessary
# g(x,y).cpu()

# do it one more time with timing
tme_gpu = datetime.datetime.now()
g(x,y).cpu()
tme_gpu = datetime.datetime.now() - tme_gpu
print('Time with CUDA:    ',tme_gpu)

# print the speedup
print('Speedup:           ',tme_cpu.total_seconds()/tme_gpu.total_seconds())
