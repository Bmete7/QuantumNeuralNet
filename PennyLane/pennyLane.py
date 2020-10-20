# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 21:58:50 2020

@author: burak
"""

 

import pennylane as qml


import timeit

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function
from torchvision import datasets,transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit.visualization import *

from qiskit.circuit.library.standard_gates import HGate
'''

# state embedding is done with pennyLane

# features would be non-differentiable with pennylanes framework
#however we need gradients only up-to Unitary gate



n_samples = 100

X_train = datasets.MNIST(root='./data', train=True, download=True,
                         transform=transforms.Compose([transforms.Resize((8,8)),transforms.ToTensor() , 
                                                       transforms.Normalize((0.5,) , (1,))
                                                       ]))

# Leaving only labels 0 and 1 
idx = np.append(np.where(X_train.targets == 0)[0][:n_samples], 
                np.where(X_train.targets == 1)[0][:n_samples])

X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]

train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)


data,target = iter(train_loader).__next__()
normalized = nn.functional.normalize(data.view(1,-1)).numpy().reshape(-1)

numberOfQubits = int(np.ceil(np.log2(normalized.shape[0])))

dev = qml.device('default.qubit', wires=numberOfQubits)

@qml.qnode(dev)
def circuit():
    qml.templates.embeddings.AmplitudeEmbedding(normalized[i for i in range(numberOfQubits)])
    return qml.probs([i for i in range(numberOfQubits)]) # measurement
    # return qml.expval(qml.PauliZ([i for i in range(numberOfQubits)]))

circuit()





'''

import numpy as np
import pennylane as qml
import torch
import sklearn.datasets

# n_qubits = 2


# @qml.qnode(dev)
# def qnode(inputs, weights):
#     qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
#     print(weights)
#     qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
#     return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))



n_samples = 100

X_train = datasets.MNIST(root='./data', train=True, download=True,
                         transform=transforms.Compose([transforms.Resize((4,4)),transforms.ToTensor() , 
                                                       transforms.Normalize((0.5,) , (1,))
                                                       ]))

# Leaving only labels 0 and 1 
idx = np.append(np.where(X_train.targets == 0)[0][:n_samples], 
                np.where(X_train.targets == 1)[0][:n_samples])

X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]

train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)


data,target = iter(train_loader).__next__()
normalized = nn.functional.normalize(data.view(1,-1)).numpy().reshape(-1)

n_qubits = int(np.ceil(np.log2(normalized.shape[0])))

normalized = torch.Tensor(normalized).view(1,-1)

dev = qml.device("default.qubit", wires=n_qubits)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        @qml.qnode(dev)
        def my_qnode(inputs, weights_0):
            qml.templates.AmplitudeEmbedding(inputs,[ i for i in range(n_qubits) ])
            qml.RX(inputs[0], wires = 0)
            qml.Rot(*weights_0, wires = 2)
            return qml.probs([i for i in range(n_qubits)])
            # return qml.expval(qml.PauliZ([0])) , qml.expval(qml.PauliZ([1]))
        weight_shapes = {"weights_0": 3}
        
        
        
        self.qlayer = qml.qnn.TorchLayer(my_qnode, weight_shapes)
        self.clayer1 = torch.nn.Linear(16,12)
        self.clayer2 = torch.nn.Linear(12,10)
        self.softmax = torch.nn.Softmax(dim=1)
        self.seq = torch.nn.Sequential(self.clayer1, self.qlayer)

    def forward(self, x):
        x = self.qlayer(x)
        x = self.clayer1(x)
        x = self.clayer2(x)
        x = self.softmax(x)
        return x


class_model = Net()

print(class_model(normalized))
'''



clayer = torch.nn.Linear(16, 4)
clayer2 = torch.nn.Linear(4,1 )
model = torch.nn.Sequential(qlayer, clayer,clayer2)



samples = 100
x, y = sklearn.datasets.make_moons(samples)
y_hot = np.zeros((samples, 2))
y_hot[np.arange(samples), y] = 1

X = torch.tensor(x).float()
Y = torch.tensor(y_hot).float()

opt = torch.optim.SGD(class_model.parameters(), lr=0.5)
loss = torch.nn.L1Loss()


epochs = 8
batch_size = 5
batches = samples // batch_size

data_loader = torch.utils.data.DataLoader(list(zip(X, Y)), batch_size=batch_size,
                                          shuffle=True, drop_last=True)


for epoch in range(epochs):

    running_loss = 0

    for x, y in data_loader:
        opt.zero_grad()

        loss_evaluated = loss(class_model(x), y)
        loss_evaluated.backward()

        opt.step()

        running_loss += loss_evaluated

    avg_loss = running_loss / batches
    print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))
    
    '''