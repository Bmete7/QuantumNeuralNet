# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 23:03:42 2020

@author: burak
"""


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



n_samples = 1000

X_train = datasets.MNIST(root = './data', train=True, download=True, transform = transforms.Compose([transforms.Resize((4,4)), transforms.ToTensor()]))

X_train = datasets.MNIST(root='./data', train=True, download=True,
                         transform=transforms.Compose([
                                                       transforms.Resize((4,4)),
                                                        transforms.ToTensor(),
                                                       transforms.Normalize((0.5,),(1,)),
                                                       ]))

train_loader = torch.utils.data.DataLoader(X_train, batch_size = 1, shuffle = True)

numberOfQubits = 16

data_iter = iter(train_loader)

image, target = data_iter.__next__() 
image_flattened = image.view(-1)

q_circuit = qiskit.QuantumCircuit(16)
parameter_container = []
for i in range(16):
    parameter_container.append( qiskit.circuit.Parameter('theta' + str(i))    )

for i in range(16):
    q_circuit.ry(parameter_container[i],i)
for i in range(16):    
    q_circuit.assign_parameters({parameter_container[i]:image_flattened[i].item()}, inplace=True)

q_circuit.draw()
