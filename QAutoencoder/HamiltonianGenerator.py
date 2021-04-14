# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 20:08:21 2021

@author: burak
"""


import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from copy import deepcopy
# %% 
class HamiltonianGenerator(nn.Module):
    def __init__(self,dev):
    
        super(HamiltonianGenerator, self).__init__()
        observables = [qml.PauliX(0),qml.PauliY(0),qml.PauliZ(0) ]
        print(observables)
        @qml.qnode(dev)
        def q_circuit(coefs ,inputs = False):
            self.embedding(inputs)
            
            
            qml.Hamiltonian(*coefs, observables)
            return [qml.expval(qml.PauliZ(i)) for i in range(0,1)]
                        
        weight_shapes = {"coefs": (1,3)}        
        print(weight_shapes)
        self.qlayer = qml.qnn.TorchLayer(q_circuit, weight_shapes)
    
    
    @qml.template
    def embedding(self,inputs):
        qml.templates.AmplitudeEmbedding(inputs, wires = range(0,1), normalize = True,pad=(0.j))
        
    def forward(self, x):
        
        x =  self.qlayer(x)
        
        return x
# %% 
        
device = qml.device("default.qubit", wires = 2)
hamNet = HamiltonianGenerator(device)


# %%


res = hamNet(torch.Tensor(latents[0].astype('float64')))

H = qml.Hamiltonian(
    [1, 1, 0.5],
    [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)]
)
print(H)