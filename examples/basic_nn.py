import tnt
import tnt.nn
import torch as tn
import torch.nn as nn
import datetime


class ClassicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(256*256, 4096, dtype = tn.float32)
        self.l2 = nn.Linear(4096, 729, dtype = tn.float32)
        self.l3 = nn.Linear(729, 64, dtype = tn.float32)
        self.linear = nn.Linear(64, 10, dtype = tn.float32)

    def forward(self, x):
        x = self.l1(x)
        x = tn.relu(x)
        x = self.l2(x)
        x = tn.relu(x)
        x = self.l3(x)
        x = tn.relu(x)
        x = tn.reshape(x,[-1,64])
        return self.linear(x)

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Defining a 2D convolution layer
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.batch1 = nn.BatchNorm2d(4)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        # Defining another 2D convolution layer
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(4)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.batch3 = nn.BatchNorm2d(4)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.linear = nn.Linear(64, 10, dtype = tn.float32)

    def forward(self, x):
        show = False
        x = self.conv1(x)
        if show: print(x.shape)
        x = self.batch1(x)
        if show: print(x.shape)
        x = self.relu1(x)
        if show: print(x.shape)
        x = self.maxpool1(x)
        if show: print(x.shape)
        x = self.conv2(x)
        if show: print(x.shape)
        x = self.batch2(x)
        if show: print(x.shape)
        x = self.relu2(x)
        if show: print(x.shape)
        x = self.maxpool2(x)
        if show: print(x.shape)
        x = self.conv3(x)
        if show: print(x.shape)
        x = self.batch3(x)
        if show: print(x.shape)
        x = self.relu3(x)
        if show: print(x.shape)
        x = self.maxpool3(x)
        if show: print(x.shape)

        x = x.view(x.size(0), -1)
        # print(x.shape)

        return self.linear(x)


class BasicTT(nn.Module):
    def __init__(self):
        super().__init__()
        self.ttl1 = tnt.nn.LinearLayerTT([4,8,8,4,8,8], [4,4,4,4,4,4], [1,3,3,3,3,3,1])
        self.ttl2 = tnt.nn.LinearLayerTT([4,4,4,4,4,4], [2,2,4,2,2,4], [1,2,2,2,2,2,1])
        self.ttl3 = tnt.nn.LinearLayerTT([2,2,4,2,2,4], [2,2,2,2,2,2], [1,2,2,2,2,2,1])
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

class BasicTT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.ttl1 = tnt.nn.LinearLayerTT([16,16,16,16], [8,8,8,8], [1,3,3,3,1])
        self.ttl2 = tnt.nn.LinearLayerTT([8,8,8,8], [4,4,4,4], [1,2,2,2,1])
        self.ttl3 = tnt.nn.LinearLayerTT([4,4,4,4], [2,4,2,4], [1,2,2,2,1])
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

model = BasicTT()
model2 = ClassicModel()
model3 = CNNModel()
model4 = BasicTT2()

input = tn.rand((4,8,8,4,8,8), dtype = tn.float32)
pred = model.forward(input)
tme = datetime.datetime.now()
pred = model.forward(input)
tme = datetime.datetime.now() -tme
print('Time TT (one eval)', tme)

input = tn.rand((16,16,16,16), dtype = tn.float32)
pred = model4.forward(input)
tme = datetime.datetime.now()
pred = model4.forward(input)
tme = datetime.datetime.now() -tme
print('Time TT 2 (one eval)', tme)

input = tn.reshape(input,[-1])
tme = datetime.datetime.now()
pred = model2.forward(input)
tme = datetime.datetime.now() -tme
print('Time full (one eval)', tme)

input = tn.reshape(input,[1,1,256,256])
tme = datetime.datetime.now()
pred = model3.forward(input)
# print(pred.shape)
tme = datetime.datetime.now() -tme
print('Time CNN (one eval)', tme)

input = tn.rand((800,4,8,8,4,8,8), dtype = tn.float32)
tme = datetime.datetime.now()
pred = model.forward(input)
tme = datetime.datetime.now() -tme
print('Time TT (batch eval)', tme)

input = tn.reshape(input,[-1,256*256])
tme = datetime.datetime.now()
pred = model2.forward(input)
tme = datetime.datetime.now() -tme
print('Time full (batch eval)', tme)

input = tn.reshape(input,[-1,1,256,256])
tme = datetime.datetime.now()
pred = model3.forward(input)
tme = datetime.datetime.now() -tme
print('Time CNN (batch eval)', tme)

input = tn.reshape(input,[-1,16,16,16,16])
pred = model4.forward(input)
tme = datetime.datetime.now()
pred = model4.forward(input)
tme = datetime.datetime.now() -tme
print('Time TT 2 (batch eval)', tme)

