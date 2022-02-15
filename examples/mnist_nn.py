#!/usr/bin/env python
# coding: utf-8

#%% Digit recognition using TT neural networks
# The TT layer is applied to the MNIST dataset.
# Imports:
import torch as tn
import torch.nn as nn
import torchtt as tntt
from torch import optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

device = tn.device('cuda' if tn.cuda.is_available() else 'cpu')


#%% Download the dataset and store it to a subfolder 'data'.
train_data = datasets.MNIST(root = 'downloads', train = True, transform = ToTensor(), download = True)
test_data = datasets.MNIST(root = 'downloads', train = False, transform = ToTensor())


#%% Create 2 dataloaders for the training set and the test set.
dataloader_train = tn.utils.data.DataLoader(train_data, batch_size=1000, shuffle=True, num_workers=10)
dataloader_test = tn.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=10)


#%% Define the neural network arhitecture. I contains 2 hidden TT layers (with RELU activation function) with a linear output layer. A sotmax is applied at the output.
class BasicTT(nn.Module):
    def __init__(self):
        super().__init__()
        self.ttl1 = tntt.nn.LinearLayerTT([1,7,4,7,4], [8,10,10,10,10], [1,4,2,2,2,1])
        self.ttl2 = tntt.nn.LinearLayerTT([8,10,10,10,10], [8,3,3,3,3], [1,2,2,2,2,1])
        self.linear = nn.Linear(81*8, 10, dtype = tn.float32)
        self.logsoftmax = nn.LogSoftmax(1)

    def forward(self, x):
        x = self.ttl1(x)
        x = tn.relu(x)
        x = self.ttl2(x)
        x = tn.relu(x)
        x = x.view(-1,81*8)
        x = self.linear(x)
        return self.logsoftmax(x)


#%% Instantiate the model and choose the optimizer and the loss function.
model = BasicTT().to(device)
loss_function = nn.CrossEntropyLoss()   
optimizer = optim.Adam(model.parameters(), lr = 0.001)   


#%% Start the training for 30 epochs
n_epochs = 30
 
for epoch in range(n_epochs):
    
    for i,(input,label) in enumerate(dataloader_train):
        
        input = tn.reshape(input.to(device),[-1,1,7,4,7,4])
        label = label.to(device)
        
        optimizer.zero_grad()
        output = model(input)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        print('Epoch %d/%d iteration %d/%d loss %e'%(epoch+1,n_epochs,i+1,len(dataloader_train),loss))
        
        
#%% Compute the accuracy over the test set.
n_correct = 0
n_total = 0
for (input,label) in dataloader_test:
    input = tn.reshape(input.to(device),[-1,1,7,4,7,4])
        
    output = model(input).cpu()
    
    n_correct += tn.sum(tn.max(output,1)[1] == label)   
    
    n_total += input.shape[0]
    
print('Test accuracy ',n_correct/n_total)

