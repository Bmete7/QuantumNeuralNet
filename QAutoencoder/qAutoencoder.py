# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:01:43 2020

@author: burak
"""

# %% 
import numpy as np
import torch
from torch.autograd import Function

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import pennylane as qml
import qiskit




import timeit
from torch.utils.tensorboard import SummaryWriter



import torchvision


from torch.utils.data import Dataset, DataLoader
import sys

sys.path.append('../dataLoad')
sys.path.append('../pennyLane')
sys.path.append('../representation')
from IrisLoad import importIRIS, IRISDataset
from MNISTLoad import importMNIST
from qCircuit import Net
from visualizations import visualize, visualize_state_vec

# %% Dataset + preprocessing 


n_samples = 4
img_shape = 4
batch_size = 1
latent_space_size = 1 # Latent space size will be in N - latent_space_size
auxillary_qubit_size = 1 # for the SWAP Test

dataset_type = 'MNIST'
#dataset_type = 'IRIS'


if(dataset_type == 'MNIST'):
    train_loader, n_qubits, training_qubits_size, pad = importMNIST(img_shape, n_samples, batch_size, latent_space_size, auxillary_qubit_size, True)
if(dataset_type == 'IRIS'):
    train_loader = importIRIS()
else:
    assert('Error')
# %%
def Fidelity_loss(measurements):
    
    fidelity = (2 * measurements[0] - 1.00)
    return torch.log(1- fidelity)

dev = qml.device("default.qubit", wires=n_qubits,shots = 1000)
model = Net(dev, latent_space_size, n_qubits, training_qubits_size)

learning_rate = 0.1
epochs = 15
loss_list = []

# opt = torch.optim.SGD(model.parameters() , lr = learning_rate )
opt = torch.optim.Adam(model.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# loss_func = torch.nn.CrossEntropyLoss() # TODO replaced with fidelity
loss_func = Fidelity_loss

test_loss = nn.MSELoss()


# %%  The Training part

for epoch in range(epochs):
    total_loss = []
    start_time = timeit.time.time()
    for i,datas in enumerate(train_loader):
        opt.zero_grad()
        # for iris dataseti
        # data = datas['data']
        
        data, target = datas
        
        # They do not have to be normalized since AmplitudeEmbeddings does that
        # But we do it anyways for visualization
        
        normalized = np.abs(nn.functional.normalize((data).view(1,-1)).numpy()).reshape(-1)
        normalized = torch.Tensor(normalized).view(1,-1)
        if(pad['padding_op']):
            
            new_arg = torch.cat((normalized[0], pad['pad_tensor']), dim=0)    
            normalized = torch.Tensor(new_arg).view(1,-1)
        
        
        out = model(normalized,True)

        
        loss = loss_func(out[0])
        loss.backward()
        
        if(i%10 == 0):
            print(out)
        opt.step()
        
    
        total_loss.append(loss.item())
    end_time = timeit.time.time()
    print('Time elapsed for the epoch' +  str(epoch)  + ' : {:.2f}'.format(end_time-start_time))
    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(100. * (epoch + 1) / epochs, loss_list[-1]))
    
# %% 
n_test_samples = 10
if(dataset_type == 'MNIST'):
    test_loader, _, _, _= importMNIST(img_shape, n_test_samples, batch_size, latent_space_size, auxillary_qubit_size, False)
    
    
# %%
with torch.no_grad():
    correct = 0
    for batch_idx, datas in enumerate(test_loader):
        
        # for iris dataset
        # data = data['data']
        data,target = datas
        
        normalized = np.abs(nn.functional.normalize((data ).view(1,-1)).numpy()).reshape(-1)
        normalized = torch.Tensor(normalized).view(1,-1)
        
        if(pad['padding_op']):
            new_arg = torch.cat((normalized[0], pad['pad_tensor']), dim=0)    
            normalized = torch.Tensor(new_arg).view(1,-1)
            
        output = model(normalized, training_mode = False, return_latent =False)             
        loss = test_loss((normalized**2).view(-1), output.view(-1))
            
        visualize(output, normalized ** 2 ,img_shape)
        visualize_state_vec(output , 'output' + str(batch_idx) , training_qubits_size)
        visualize_state_vec(normalized**2, 'data' + str(batch_idx),training_qubits_size)
        
        print((normalized**2).view(-1))
        print(output.view(-1))
        print(' - - - ')
        if(batch_idx == 5):
            break
    
        total_loss.append(loss.item())

    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / len(test_loader) * 100)
        )
        
