.. _tutorial-label:

Complete Tutorial
=================

This comprehensive tutorial covers all the essential features of the ``torchtt`` package, progressing from basic concepts to advanced applications. Each section builds upon the previous ones, providing a complete learning path for working with Tensor-Train decompositions.

.. contents:: Table of Contents
   :local:
   :depth: 2

1. Basic Concepts and Getting Started
-------------------------------------

Introduction to Tensor-Train Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Tensor-Train (TT) format is a powerful method for representing high-dimensional tensors efficiently. Instead of storing all :math:`n^d` entries of a d-dimensional tensor, the TT format uses a sequence of smaller 3D tensors called "cores" that can be combined to reconstruct the original tensor.

Basic Tensor Operations
~~~~~~~~~~~~~~~~~~~~~~~

Let's start with the fundamental operations in ``torchtt``:

.. code-block:: python

    import torch as tn
    import torchtt as tntt 

    # Create a full tensor and decompose it in TT format
    tens_full = tn.reshape(tn.arange(32*16*8*10, dtype=tn.float64), [32,16,8,10])
    tens_tt = tntt.TT(tens_full)

    # Inspect the TT decomposition
    print('TT cores:', tens_tt.cores)
    print('Mode sizes:', tens_tt.N)
    print('TT rank:', tens_tt.R)

    # Approximate decomposition with specified accuracy
    tens_full2 = tens_full + 1e-5*tn.randn(tens_full.shape, dtype=tens_full.dtype)
    tens_tt2 = tntt.TT(tens_full2, eps=1e-4)

    # Recover the original tensor
    tens_full_rec = tens_tt.full()
    error = tn.linalg.norm(tens_full-tens_full_rec)/tn.linalg.norm(tens_full)
    print(f'Reconstruction error: {error}')

Working with Tensor Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor operators generalize matrix-vector operations to higher dimensions:

.. code-block:: python

    # Create a tensor operator (TT matrix)
    A_full = tn.reshape(tn.arange(8*4*6*3*7*9, dtype=tn.float64), [8,4,6,3,7,9])
    A_ttm = tntt.TT(A_full, eps=1e-12, shape=[(8,3),(4,7),(6,9)])

Slicing and Indexing
~~~~~~~~~~~~~~~~~~~~

TT tensors support flexible slicing operations:

.. code-block:: python

    # Single element access
    print(tens_tt[1,2,3,4])
    
    # Slice operations (returns another TT tensor)
    print(tens_tt[1,1:4,2,:])

Rank Rounding
~~~~~~~~~~~~~

Controlling the rank is crucial for efficiency:

.. code-block:: python

    # Create tensors with different contributions
    t1 = tntt.randn([10,20,30,40], [1,2,2,2,1])
    t2 = tntt.randn([10,20,30,40], [1,2,2,2,1])
    t3 = tntt.randn([10,20,30,40], [1,2,2,2,1])
    
    # Normalize and combine
    t1, t2, t3 = t1/t1.norm(), t2/t2.norm(), t3/t3.norm()
    tt = t1 + 1e-3*t2 + 1e-6*t3

    # Round to specified accuracy
    tt_rounded = tt.round(1e-5)
    print(f'Original rank: {tt.R}')
    print(f'Rounded rank: {tt_rounded.R}')

Special Tensors
~~~~~~~~~~~~~~~

Create common tensor types directly in TT format:

.. code-block:: python

    # Basic tensors
    ones_tt = tntt.ones([2,3,4])
    zeros_tt = tntt.zeros([2,3,4])
    eye_tt = tntt.eye([10,20,30])
    
    # Random tensors with specified rank
    random_tt = tntt.random([3,4,5,6,7], [1,2,5,5,2,1])
    randn_tt = tntt.randn([30]*5, [1,8,16,16,8,1], var=1.0)

2. Basic Linear Algebra Operations
----------------------------------

The ``torchtt`` package supports all fundamental linear algebra operations directly on TT tensors without converting to full format.

Arithmetic Operations
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create sample tensors
    N = [10,10,10,10]
    x = tntt.randn(N, [1,4,4,4,1])
    y = tntt.TT(tn.reshape(tn.arange(N[0]*N[1]*N[2]*N[3], dtype=tn.float64), N))
    A = tntt.randn([(n,n) for n in N], [1,2,3,4,1])
    B = tntt.randn([(n,n) for n in N], [1,2,3,4,1])

    # Addition and subtraction
    z = x + y + 1.0  # Can add scalars
    v = x - y - 1 - 0.5
    w = -x + x  # Negation

    # Broadcasting is supported
    xx = tntt.random([2,3,4,5], [1,2,3,4,1])
    yy = tntt.random([4,5], [1,2,1])
    result = xx + yy  # Broadcasting works

Elementwise Operations
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Elementwise multiplication (Hadamard product)
    u = x * y
    print(f'Result rank: {u.R}')  # Rank is product of input ranks

Matrix-Vector and Matrix-Matrix Products
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # TT matrix @ TT tensor
    result1 = A @ x
    
    # TT matrix @ TT matrix  
    result2 = A @ B
    
    # Chain operations
    result3 = A @ B @ x
    
    # TT matrix @ full tensor (returns full tensor)
    result4 = A @ tn.rand(A.N, dtype=tn.float64)

Kronecker Product
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Using ** operator or tntt.kron()
    kron1 = x ** y
    kron2 = A ** A

Norms and Dot Products
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Frobenius norm
    norm_y = y.norm()
    norm_A = A.norm()
    
    # Sum operations
    total_sum = y.sum()  # Sum all elements
    partial_sum1 = x.sum(1)  # Sum along mode 1
    partial_sum2 = x.sum([0,1,3])  # Sum along multiple modes
    
    # Dot product
    dot_product = tntt.dot(y, y)
    
    # Dot product with mode specification
    t1 = tntt.randn([4,5,6,7,8,9], [1,2,4,4,4,4,1])
    t2 = tntt.randn([5,7,9], [1,3,3,1])
    partial_dot = tntt.dot(t1, t2, [1,3,5])

Reshaping
~~~~~~~~~

.. code-block:: python

    # Create and reshape tensors
    q = tntt.TT(tn.reshape(tn.arange(2*3*4*5*7*3, dtype=tn.float64), [2,3,4,5,7,3]))
    
    # Series of reshapes
    w = tntt.reshape(q, [12,10,21])
    w = tntt.reshape(w, [360,7])
    w = tntt.reshape(w, [2,3,4,5,7,3])  # Back to original
    
    error = (w-q).norm()/q.norm()
    print(f'Reshape error: {error}')

3. CUDA and GPU Acceleration
----------------------------

The ``torchtt`` package leverages PyTorch's GPU acceleration capabilities for significant speedups.

Basic GPU Operations
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import datetime
    
    # Check CUDA availability
    print('CUDA available:', tn.cuda.is_available())
    if tn.cuda.is_available():
        print('Device name:', tn.cuda.get_device_name())

    # Create test function
    def f(x, A, y):
        z = A @ y + A @ y  # Operation that grows rank
        z = z.round(1e-12)  # Rank rounding (contains QR and SVD)
        z += z * x
        return tntt.dot(x, z)

    # Generate test tensors on CPU
    x = tntt.random([200,300,400,500], [1,10,10,10,1])
    y = tntt.random([200,300,400,500], [1,8,8,8,1])
    A = tntt.random([(200,200),(300,300),(400,400),(500,500)], [1,8,8,8,1])

    # Benchmark CPU performance
    time_cpu = datetime.datetime.now()
    result_cpu = f(x, A, y)
    time_cpu = datetime.datetime.now() - time_cpu
    print('Time on CPU:', time_cpu)

Moving Tensors to GPU
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Move tensors to GPU
    x_gpu = x.cuda()
    y_gpu = y.cuda()
    A_gpu = A.cuda()
    
    # Alternative: use .to() method
    device = tn.cuda.current_device()
    x_gpu = x.to(device)
    
    # Warm-up CUDA
    f(x_gpu*0, A_gpu*0, 0*y_gpu).cpu()

    # Benchmark GPU performance
    time_gpu = datetime.datetime.now()
    result_gpu = f(x_gpu, A_gpu, y_gpu).cpu()  # Move result back to CPU
    time_gpu = datetime.datetime.now() - time_gpu
    
    print('Time with CUDA:', time_gpu)
    print('Speedup:', time_cpu.total_seconds()/time_gpu.total_seconds(), 'times')

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Test without rank rounding for better GPU performance
    def g(x, A, y):
        z = A @ y + A @ y
        z += z + x  # No rounding step
        return tntt.dot(x, z)
    
    # GPU operations without QR/SVD show better speedups
    # since these operations are less parallelizable

4. Efficient Linear Algebra
---------------------------

Advanced algorithms for efficient TT operations when standard methods become too expensive.

Fast Matrix-Vector Products (DMRG)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import datetime
    
    # Create large rank tensors
    n = 4
    A = tntt.random([(n,n)]*8, [1]+7*[4]+[1])
    x = tntt.random([n]*8, [1]+7*[5]+[1])
    
    # Artificially increase rank
    A = A + A + A + A - A + A - A + A
    x = x + x + x + x + x + x + x + x - x + x - x + x
    
    print(f'Matrix A: {A}')
    print(f'Vector x: {x}')

    # Standard matrix-vector product with rounding
    time_classic = datetime.datetime.now()
    y_classic = (A @ x).round(1e-12)
    time_classic = datetime.datetime.now() - time_classic
    print('Time classic:', time_classic)

    # Fast DMRG-based matrix-vector product
    time_dmrg = datetime.datetime.now()
    y_fast = A.fast_matvec(x)
    time_dmrg = datetime.datetime.now() - time_dmrg
    print('Time DMRG:', time_dmrg)
    
    # Verify accuracy
    error = (y_classic - y_fast).norm().numpy() / y_classic.norm().numpy()
    print('Relative error:', error)

Alternative Fast Matrix-Vector Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Alternative method for QTT tensors
    A_qtt = tntt.random([(2,2)]*8, [1]+7*[6]+[1])
    x_qtt = tntt.random([2]*8, [1]+7*[5]+[1])
    
    # Increase rank
    for _ in range(8): 
        A_qtt += A_qtt
        x_qtt += x_qtt

    time_fast2 = datetime.datetime.now()
    y_fast2 = tntt.fast_mv(A_qtt, x_qtt)
    time_fast2 = datetime.datetime.now() - time_fast2
    print('Time fast method 2:', time_fast2)

Elementwise Division (AMEN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create test tensors for division
    N = [32,50,44,64]
    I = tntt.meshgrid([tn.arange(n, dtype=tn.float64) for n in N])
    
    # x = 2 + i_1, y = i_1^2 + i_2 + i_3 + i_4 + 1
    x = 2 + I[0]
    x = x.round(1e-15)
    y = I[0]*I[0] + I[1] + I[2] + I[3] + 1
    y = y.round(1e-15)

    # Elementwise division z = x / y
    z = x / y
    error = tn.linalg.norm(z.full() - x.full()/y.full()) / tn.linalg.norm(z.full())
    print('Division error:', error)

    # Elementwise inversion u = 1 / y
    u = 1 / y
    error = tn.linalg.norm(u.full() - 1/y.full()) / tn.linalg.norm(u.full())
    print('Inversion error:', error)

5. Cross Interpolation
----------------------

Build TT decompositions from function evaluations without forming the full tensor.

Basic Cross Interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Define a function to approximate
    func1 = lambda I: 1/(2 + tn.sum(I+1, 1).to(dtype=tn.float64))

    # Cross interpolation for moderate dimensions
    N = [20]*4
    x = tntt.interpolate.dmrg_cross(func1, N, eps=1e-7)

    # Verify accuracy
    Is = tntt.meshgrid([tn.arange(0, n, dtype=tn.float64) for n in N])
    x_ref = 1/(2 + Is[0].full() + Is[1].full() + Is[2].full() + Is[3].full() + 4)
    error = tn.linalg.norm(x.full() - x_ref) / tn.linalg.norm(x_ref)
    print('Cross interpolation error:', error)

High-Dimensional Cross Interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # High-dimensional case where full tensor is infeasible
    N = [32]*10  # 32^10 entries would be impossible to store
    x = tntt.interpolate.dmrg_cross(func1, N, eps=1e-10, verbose=True)

    # Verify specific entries
    test_idx = tn.tensor([[1,2,3,4,5,6,7,8,9,11]])
    computed = x[1,2,3,4,5,6,7,8,9,11]
    reference = func1(test_idx)
    print(f'Computed: {computed}, Reference: {reference}')
    
    print(f'Final tensor: {x}')

Function Interpolation on TT Tensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Apply univariate function elementwise
    x_base = tntt.TT(x_ref)
    func = lambda t: tn.log(t)
    y = tntt.interpolate.function_interpolate(func, x_base, 1e-9)
    
    error = tn.linalg.norm(y.full() - func(x_ref)) / tn.linalg.norm(func(x_ref))
    print('Function interpolation error:', error)

Multivariate Function Interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Apply multivariate function to multiple TT tensors
    z = tntt.interpolate.function_interpolate(func1, Is)
    error = tn.linalg.norm(z.full() - x_ref) / tn.linalg.norm(x_ref)
    print('Multivariate interpolation error:', error)

6. System Solvers
-----------------

Solve large multilinear systems :math:`\mathsf{Ax} = \mathsf{b}` efficiently in TT format.

Basic Linear System Solving
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import datetime
    
    # Small example with known solution
    A = tntt.random([(4,4),(5,5),(6,6)], [1,2,3,1])
    x_exact = tntt.random([4,5,6], [1,2,3,1])
    b = A @ x_exact  # Create RHS with known solution

    # Solve the system
    x_solved = tntt.solvers.amen_solve(A, b, x0=b, eps=1e-7)
    
    # Check results
    residual_error = (A @ x_solved - b).norm() / b.norm()
    solution_error = (x_solved - x_exact).norm() / x_exact.norm()
    
    print(f'Residual error: {residual_error}')
    print(f'Solution error: {solution_error}')

Finite Difference Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Solve Δu = 1 in [0,1]^d with u = 0 on boundary
    dtype = tn.float64
    n = 64
    d = 8

    # Create 1D finite difference operator
    L1d = -2*tn.eye(n, dtype=dtype) + tn.diag(tn.ones(n-1, dtype=dtype), -1) + tn.diag(tn.ones(n-1, dtype=dtype), 1)
    L1d[0,1] = 0    # Boundary condition
    L1d[-1,-2] = 0  # Boundary condition
    L1d *= (n-1)**2
    L1d = tntt.TT(L1d, [(n,n)])

    # Build d-dimensional Laplacian using Kronecker products
    L_tt = tntt.zeros([(n,n)]*d)
    for i in range(1, d-1):
        L_tt = L_tt + tntt.eye([n]*i) ** L1d ** tntt.eye([n]*(d-1-i))
    L_tt = L_tt + L1d ** tntt.eye([n]*(d-1)) + tntt.eye([n]*(d-1)) ** L1d
    L_tt = L_tt.round(1e-14)

    # Create right-hand side
    b1d = tn.ones(n, dtype=dtype)
    b1d = tntt.TT(b1d)
    b_tt = b1d
    for i in range(d-1):
        b_tt = b_tt ** b1d

    # Solve the system
    time_start = datetime.datetime.now()
    x = tntt.solvers.amen_solve(L_tt, b_tt, x0=b_tt, nswp=20, eps=1e-7, 
                                verbose=True, preconditioner='c', use_cpp=True)
    solve_time = datetime.datetime.now() - time_start
    
    residual = (L_tt @ x - b_tt).norm() / b_tt.norm()
    print(f'Residual: {residual}')
    print(f'Solver time: {solve_time}')
    print(f'Solution: {x}')

GPU-Accelerated Solving
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Solve on GPU if available
    if tn.cuda.is_available():
        time_start = datetime.datetime.now()
        x_gpu = tntt.solvers.amen_solve(L_tt.cuda(), b_tt.cuda(), x0=b_tt.cuda(), 
                                        nswp=20, eps=1e-8, verbose=True, preconditioner='c')
        solve_time_gpu = datetime.datetime.now() - time_start
        x_gpu = x_gpu.cpu()
        
        residual_gpu = (L_tt @ x_gpu - b_tt).norm() / b_tt.norm()
        print(f'GPU residual: {residual_gpu}')
        print(f'GPU solver time: {solve_time_gpu}')

7. Automatic Differentiation
----------------------------

Compute gradients with respect to TT cores for optimization and machine learning applications.

Basic Automatic Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Create test tensors and function
    N = [2,3,4,5]
    A = tntt.randn([(n,n) for n in N], [1]+[2]*(len(N)-1)+[1])
    y = tntt.randn(N, A.R)
    x = tntt.ones(N)

    def f(x, A, y):
        z = tntt.dot(A @ (x-y), (x-y))
        return z.norm()

    # Start gradient recording
    tntt.grad.watch(x)

    # Compute function value and gradients
    val = f(x, A, y)
    grad_cores = tntt.grad.grad(val, x)

Gradient Verification
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Verify gradient using finite differences
    h = 1e-7
    x1 = x.clone()
    x1.cores[1][0,0,0] += h
    x2 = x.clone()
    x2.cores[1][0,0,0] -= h
    
    finite_diff = (f(x1, A, y) - f(x2, A, y)) / (2*h)
    ad_gradient = grad_cores[1][0,0,0]
    
    relative_error = tn.abs(finite_diff - ad_gradient) / tn.abs(finite_diff)
    print(f'Gradient verification error: {relative_error}')

Selective Core Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Differentiate with respect to specific cores only
    core_indices = [0, 2]  # Only differentiate w.r.t. cores 0 and 2
    tntt.grad.watch(x, core_indices=core_indices)
    val = f(x, A, y)
    partial_grad = tntt.grad.grad(val, x, core_indices=core_indices)

8. Manifold Optimization
------------------------

Optimize functions directly on the TT manifold using Riemannian gradients.

Riemannian Gradient Descent
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Define optimization problem
    N = [10,11,12,13,14]
    Rt = [1,3,4,5,6,1]    # Target rank
    Rx = [1,6,6,6,6,1]    # Search rank (higher than target)
    
    target = tntt.randn(N, Rt).round(0)  # Target tensor
    func = lambda x: 0.5*(x-target).norm(True)  # Objective function

    # Initialize and optimize
    x0 = tntt.randn(N, Rx)
    x = x0.clone()
    
    for i in range(20):
        # Compute Riemannian gradient
        gr = tntt.manifold.riemannian_gradient(x, func)
        
        # Update step
        alpha = 1.0  # Step size
        x = (x - alpha*gr).round(0, Rx)  # Project back to manifold
        
        print(f'Iteration {i+1}: Value = {func(x).numpy()}')

Standard Gradient Descent Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Compare with standard gradient descent
    y = x0.detach().clone()

    for i in range(100):  # More iterations needed for standard method
        tntt.grad.watch(y)
        fval = func(y)
        deriv = tntt.grad.grad(fval, y)
        
        alpha = 0.00001  # Much smaller step size for stability
        new_cores = [y.cores[j].detach() - alpha*deriv[j] for j in range(len(deriv))]
        y = tntt.TT(new_cores)
        
        if i % 10 == 0:
            print(f'Standard GD iteration {i}: Value = {func(y).numpy()}')

9. Neural Networks with TT Layers
---------------------------------

Build neural networks with TT-compressed linear layers for parameter efficiency.

Basic TT Neural Network
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch.nn as nn
    import datetime

    class BasicTT(nn.Module):
        def __init__(self):
            super().__init__()
            # TT layers with specified input/output shapes and ranks
            self.ttl1 = tntt.nn.LinearLayerTT([16,16,16,16], [8,8,8,8], [1,3,3,3,1])
            self.ttl2 = tntt.nn.LinearLayerTT([8,8,8,8], [4,4,4,4], [1,2,2,2,1])
            self.ttl3 = tntt.nn.LinearLayerTT([4,4,4,4], [2,4,2,4], [1,2,2,2,1])
            self.linear = nn.Linear(64, 10, dtype=tn.float32)

        def forward(self, x):
            x = self.ttl1(x)
            x = tn.relu(x)
            x = self.ttl2(x)
            x = tn.relu(x)
            x = self.ttl3(x)
            x = tn.relu(x)
            x = tn.reshape(x, [-1, 64])
            return self.linear(x)

Model Analysis
~~~~~~~~~~~~~~

.. code-block:: python

    # Create and analyze the model
    model = BasicTT()
    print(f'Number of trainable parameters: {len(list(model.parameters()))}')
    print(f'Model structure:\n{model}')

    # Test forward pass
    input_single = tn.rand((16,16,16,16), dtype=tn.float32)
    pred_single = model.forward(input_single)
    print(f'Single input shape: {input_single.shape}')
    print(f'Single output shape: {pred_single.shape}')

    # Batch processing
    input_batch = tn.rand((1000,16,16,16,16), dtype=tn.float32)
    pred_batch = model.forward(input_batch)
    print(f'Batch input shape: {input_batch.shape}')
    print(f'Batch output shape: {pred_batch.shape}')

Training Loop
~~~~~~~~~~~~~

.. code-block:: python

    # Setup training
    label_batch = tn.randint(0, 10, (1000,))  # Classification labels
    criterion = nn.CrossEntropyLoss()
    optimizer = tn.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(5):
        optimizer.zero_grad()
        
        outputs = model(input_batch)
        loss = criterion(outputs, label_batch)
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch+1}, loss: {loss.item():.6f}')

    print('Training completed!')

GPU Training
~~~~~~~~~~~~

.. code-block:: python

    # GPU acceleration for training
    if tn.cuda.is_available():
        model_gpu = BasicTT().cuda()
        input_batch_gpu = tn.rand((400,16,16,16,16)).cuda()
        
        # CPU timing
        input_cpu = tn.rand((400,16,16,16,16))
        time_start = datetime.datetime.now()
        pred_cpu = model(input_cpu)
        time_cpu = datetime.datetime.now() - time_start
        print(f'CPU inference time: {time_cpu}')
        
        # GPU timing  
        time_start = datetime.datetime.now()
        pred_gpu = model_gpu.forward(input_batch_gpu).cpu()
        time_gpu = datetime.datetime.now() - time_start
        print(f'GPU inference time: {time_gpu}')
        
        speedup = time_cpu.total_seconds() / time_gpu.total_seconds()
        print(f'GPU speedup: {speedup:.2f}x')

Advanced Applications
---------------------

Parameter Efficiency Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Compare TT layer vs standard linear layer parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Standard dense network
    class DenseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(16**4, 8**4, dtype=tn.float32)  # 65536 -> 4096
            self.linear2 = nn.Linear(8**4, 4**4, dtype=tn.float32)   # 4096 -> 256
            self.linear3 = nn.Linear(4**4, 64, dtype=tn.float32)     # 256 -> 64
            self.linear4 = nn.Linear(64, 10, dtype=tn.float32)       # 64 -> 10
            
        def forward(self, x):
            x = x.view(-1, 16**4)
            x = tn.relu(self.linear1(x))
            x = tn.relu(self.linear2(x))
            x = tn.relu(self.linear3(x))
            return self.linear4(x)

    dense_model = DenseNet()
    tt_model = BasicTT()
    
    dense_params = count_parameters(dense_model)
    tt_params = count_parameters(tt_model)
    
    print(f'Dense model parameters: {dense_params:,}')
    print(f'TT model parameters: {tt_params:,}')
    print(f'Parameter reduction: {dense_params/tt_params:.1f}x')

Best Practices and Tips
-----------------------

Memory Management
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Monitor memory usage during operations
    def print_memory_usage():
        if tn.cuda.is_available():
            print(f'GPU memory: {tn.cuda.memory_allocated()/1e9:.2f} GB')
        
    # Use appropriate data types
    x_float32 = tntt.randn([100]*5, [1,10,10,10,1], dtype=tn.float32)
    x_float64 = tntt.randn([100]*5, [1,10,10,10,1], dtype=tn.float64)
    
    print(f'Float32 tensor size: {sum(core.numel() for core in x_float32.cores)}')
    print(f'Float64 tensor size: {sum(core.numel() for core in x_float64.cores)}')

Rank Selection Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Analyze rank vs accuracy tradeoff
    original = tntt.randn([20]*6, [1,20,20,20,20,1])
    
    ranks_to_test = [[1,5,5,5,5,1], [1,10,10,10,10,1], [1,15,15,15,15,1]]
    
    for rank in ranks_to_test:
        approximation = original.round(1e-12, rank)
        error = (original - approximation).norm() / original.norm()
        storage = sum(r1*n*r2 for r1, n, r2 in zip(rank[:-1], original.N, rank[1:]))
        print(f'Rank {rank}: Error = {error:.2e}, Storage = {storage}')

Numerical Stability
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Handle numerical issues in optimization
    def safe_division(x, y, eps=1e-12):
        """Safe elementwise division with regularization"""
        y_reg = y + eps * tntt.ones(y.N)
        return x / y_reg
    
    # Use appropriate tolerances for different operations
    high_accuracy = 1e-12  # For exact operations
    medium_accuracy = 1e-8  # For iterative methods
    low_accuracy = 1e-4     # For approximations

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Batch operations when possible
    def efficient_batch_matvec(A, X_list):
        """Efficiently apply A to multiple vectors"""
        # Stack vectors and use batch operations
        if len(X_list) > 1:
            # Use fast methods for multiple applications
            return [A.fast_matvec(x) for x in X_list]
        else:
            return [A @ X_list[0]]
    
    # Reuse decompositions
    def cached_decomposition(tensor, cache={}):
        """Cache TT decompositions to avoid recomputation"""
        tensor_hash = hash(tensor.data_ptr())  # Simple hash
        if tensor_hash not in cache:
            cache[tensor_hash] = tntt.TT(tensor)
        return cache[tensor_hash]

This completes our comprehensive tutorial covering all major features of the ``torchtt`` package. Each section builds upon the previous ones, providing both theoretical understanding and practical implementation examples. The tutorial progresses from basic tensor operations to advanced applications in machine learning and scientific computing.

For more detailed examples and applications, refer to the individual example scripts in the ``examples/`` directory of the torchTT repository. 