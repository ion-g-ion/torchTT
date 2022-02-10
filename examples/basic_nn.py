#!/usr/bin/env python
# coding: utf-8

#%% Tensor Train layers for neural networks
# In this section, the TT layers are introduced.
# Imports:
import torch as tn
import torch.nn as nn
import datetime
import torchtt as tntt


#%% We consider a linear layer $\mathcal{LTT}(\mathsf{x}) = \mathsf{Wx}+\mathsf{b}$ acting on a tensor input $\mathsf{x}$ of shape $n_1 \times \cdots \times n_d$ and returning a tensor of shape $m_1\times\cdots\times m_d$. The corresponding weight matrix $\mathsf{W}$ would have the shape $(m_1\times\cdots\times m_d) \times (n_1 \times \cdots \times n_d)$. The goal is to represent the weights tensor operator in TT format and perform the learning with respect tot the cores of the TT decomposition (ranks have to be fixed a priori).
# Due to the AD functionality of `torchtt`, the gradient with respect tot the cores can be computed for any network structure.
# TT layers can be added using `torchtt.nn.LinearLayerTT()` class. 
# In the following, a neural netywork with 3 hidden layers and one linear layer is created.
# The shapes of the individual layers are 
# $\mathbb{R}^{16} \times\mathbb{R}^{16} \times\mathbb{R}^{16} \times\mathbb{R}^{16} \underset{}{\longrightarrow} \mathbb{R}^8 \times\mathbb{R}^8 \times\mathbb{R}^8 \times\mathbb{R}^8 \underset{}{\longrightarrow} \mathbb{R}^4 \times\mathbb{R}^4 \times\mathbb{R}^4 \times\mathbb{R}^4  \underset{}{\longrightarrow}  \mathbb{R}^2 \times\mathbb{R}^4 \times\mathbb{R}^2 \times\mathbb{R}^4 \underset{}{\longrightarrow} \mathbb{R}^{10}$.

class BasicTT(nn.Module):
    def __init__(self):
        super().__init__()
        self.ttl1 = tntt.nn.LinearLayerTT([16,16,16,16], [8,8,8,8], [1,3,3,3,1])
        self.ttl2 = tntt.nn.LinearLayerTT([8,8,8,8], [4,4,4,4], [1,2,2,2,1])
        self.ttl3 = tntt.nn.LinearLayerTT([4,4,4,4], [2,4,2,4], [1,2,2,2,1])
        self.linear = nn.Linear(64, 10, dtype = tn.float32)

    def forward(self, x):
        x = self.ttl1(x)
        x = tn.relu(x)
        x = self.ttl2(x)
        x = tn.relu(x)
        x = self.ttl3(x)
        x = tn.relu(x)
        x = tn.reshape(x,[-1,64])
        return self.linear(x)


#% Create the model and print the number of trainable parameters as well as the model structure.
model = BasicTT()
print('Number of trainable parameters:', len(list(model.parameters())))
print(model)


#%% A random input is created and passed as argument to the model. Batch evaluation is also possible by extending the dimensionality of the input before the leading mode.
input = tn.rand((16,16,16,16), dtype = tn.float32)
pred = model.forward(input)

input_batch = tn.rand((1000,16,16,16,16), dtype = tn.float32)
label_batch = tn.rand((1000,10), dtype = tn.float32)

#%% The obtained network can be trained similarily to other `torch` models.
# A loss function together with an optimizer are defined. 
criterion = nn.CrossEntropyLoss()
optimizer = tn.optim.Adam(model.parameters(), lr = 0.001)   


#%% A training loop is executed to exemplify the training parameters update procedure. An example where a true dataset is used is presented [here](https://github.com/ion-g-ion/torchTT/blob/main/examples/mnist_nn.ipynb).  
for epoch in range(5):  

    optimizer.zero_grad()

    outputs = model(input_batch)
    loss = criterion(outputs, label_batch)
    loss.backward()
    optimizer.step()

    # print statistics
    print('Epoch %d, loss %e'%(epoch+1,loss.item()))


print('Finished Training')


#%% If the GPU is available, the model can be run on it to get a speedup (should be run 2 times to see the speedup due to CUDA warm-up).
if tn.cuda.is_available():
    model_gpu = BasicTT().cuda()
    input_batch_gpu = tn.rand((400,16,16,16,16)).cuda()

    input_batch = tn.rand((400,16,16,16,16))
    tme = datetime.datetime.now()
    pred = model.forward(input_batch)
    tme = datetime.datetime.now() - tme
    print('Time on CPU ',tme)

    tme = datetime.datetime.now()
    pred_gpu = model_gpu.forward(input_batch_gpu).cpu()
    tme = datetime.datetime.now() - tme
    print('Time on GPU ',tme)

    

