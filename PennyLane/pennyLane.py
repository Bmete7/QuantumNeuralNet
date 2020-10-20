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
    qml.templates.embeddings.AmplitudeEmbedding(normalized,[i for i in range(numberOfQubits)])
    return qml.probs([i for i in range(numberOfQubits)]) # measurement
    # return qml.expval(qml.PauliZ([i for i in range(numberOfQubits)]))

circuit()
