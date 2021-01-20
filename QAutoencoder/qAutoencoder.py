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




import timeit
from torch.utils.tensorboard import SummaryWriter



import torchvision


from torch.utils.data import Dataset, DataLoader
import sys

sys.path.append('../dataLoad')
sys.path.append('C:/Users/burak/OneDrive/Desktop/Quantum Machine Learning/CodeBank/QuantumNeuralNet/dataLoad')
sys.path.append('C:/Users/burak/OneDrive/Desktop/Quantum Machine Learning/CodeBank/QuantumNeuralNet/pennyLane')
sys.path.append('C:/Users/burak/OneDrive/Desktop/Quantum Machine Learning/CodeBank/QuantumNeuralNet/representation')
sys.path.append('../pennyLane')
sys.path.append('../representation')
from IrisLoad import importIRIS, IRISDataset
from MNISTLoad import importMNIST
from qCircuit import Net, latNet, genNet
from visualizations import visualize, visualize_state_vec
from copy import deepcopy
import seaborn
# %% Dataset + preprocessing 


# Couple of changes is done on torch.py and jacobian.py for grads in pennylane

n_samples = 2
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
def Fidelity_loss(mes):
    tot  =0
    for i in mes[0]:
        tot += i[0]
    fidelity = (2 * (tot) / len(mes[0])  - 1.00)
    
    
    return torch.log(1- fidelity)

dev = qml.device("default.qubit", wires=n_qubits,shots = 1000)
model = Net(dev, latent_space_size, n_qubits, training_qubits_size,auxillary_qubit_size)

learning_rate = 0.1
epochs = 9
loss_list = []

# opt = torch.optim.SGD(model.parameters() , lr = learning_rate )
opt = torch.optim.Adam(model.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# loss_func = torch.nn.CrossEntropyLoss() # TODO replaced with fidelity
loss_func = Fidelity_loss

test_loss = nn.MSELoss()



q1 = np.zeros(2)
q1[1] = 1 

hadamard = np.array(((1+0j,1), (1,-1)))
hadamard /= np.sqrt(2)
q2 = np.zeros(2)
q2[0] = 1 
q2 = np.array(hadamard @ q2, dtype = 'float64')
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
        # data = np.kron(np.kron(np.array(q1,dtype='float64'),q2), np.kron(np.array(q1,dtype='float64'),q2))
        # if(epoch%2 == 1):        
        #     data = np.kron(np.kron(q2,q2), np.kron(q2,q2))
        # data = torch.Tensor(data)
        
        # data = np.abs(nn.functional.normalize((data).view(1,-1)).numpy()).reshape(-1)
        # normalized = torch.Tensor(data).view(1,-1)
        
        
        if(pad['padding_op']):
            
            new_arg = torch.cat((normalized[0], pad['pad_tensor']), dim=0)    
            normalized = torch.Tensor(new_arg).view(1,-1)
        
        
        out = model(normalized,True)
        loss = loss_func(out)
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
dev2 = qml.device("default.qubit", wires=n_qubits,shots = 1000)
dev3 = qml.device("default.qubit", wires=n_qubits,shots = 1000)
paramserver = model.paramServer()
latModel = latNet(dev2, latent_space_size, n_qubits, training_qubits_size,paramserver,auxillary_qubit_size)
genModel = genNet(dev3, latent_space_size, n_qubits, training_qubits_size,paramserver,auxillary_qubit_size)
    

# %% 

def qubitsToStateVectorNew(vec, n):
    a = np.sum(vec[0,:(2**(n-1))].numpy())
    
    if(n == 3):
        b=  (vec[0,0] + vec[0,1] + vec[0,4] + vec[0,5]).numpy()
    elif(n==4):
        b=  (vec[0,0] + vec[0,1] + vec[0,2] +  vec[0,3]  + vec[0,8] + vec[0,9] + vec[0,10] + vec[0,11] ).numpy()
    
    
    if(n==3):
        c  = np.sum(vec[0,::2].numpy())
    else:
        c =  (vec[0,0] + vec[0,1] + vec[0,4] +  vec[0,5]  + vec[0,8] + vec[0,9] + vec[0,12] + vec[0,13] ).numpy()
        
    print(a,b,c)
    d = np.sum(vec[0,::2].numpy())
    s0 = np.zeros(2)
    s0[0] = 1
    
    s1 = np.zeros(2)
    s1[1] = 1
    
    lat1 = s0 * np.sqrt(a)  +s1 * np.sqrt((1-a))
    lat2 = s0 * np.sqrt(b)  +s1 * np.sqrt((1-b))
    lat3 = s0 * np.sqrt(c)  +s1 * np.sqrt((1-c))
    lat4 = s0 * np.sqrt(d)  +s1 * np.sqrt((1-d))
    if  ( n== 3):
        return lat1,lat2,lat3
    else:
        return lat1,lat2,lat3,lat4

# %% 


def qubitsToStateVector(vec, n):
        
    s0 = np.zeros(2)
    s0[0] = 1
    
    s1 = np.zeros(2)
    s1[1] = 1
    vals = ((vec + 1 )/2)
    state_1_vals = 1- deepcopy(vals)
    
    vals = np.sqrt(vals)
    state_1_vals = np.sqrt(state_1_vals)
    print(vals)
    lat1 = s0 * vals[0][0].numpy()  +s1 * (state_1_vals[0][0].numpy())
    lat2 = s0 * vals[0][1].numpy() + s1 * (state_1_vals[0][1].numpy())
    lat3 = s0 * vals[0][2].numpy() + s1 * (state_1_vals[0][2].numpy())
    if(n == 4):
        lat4 = s0 * vals[0][3].numpy() + s1 * (state_1_vals[0][3].numpy())
        return lat1,lat2,lat3,lat4
    else:
        return lat1,lat2,lat3
    
# %%
with torch.no_grad():
    correct = 0
    for batch_idx, datas in enumerate(test_loader):
        
        # for iris dataset
        # data = data['data']
        data,target = datas
        
        normalized = np.abs(nn.functional.normalize((data ).view(1,-1)).numpy()).reshape(-1)
        normalized = torch.Tensor(normalized).view(1,-1)
        data = np.kron(np.kron(np.array(q1,dtype='float64'),q2), np.kron(np.array(q1,dtype='float64'),q2))
        data = np.kron(np.kron(q2,q2), np.kron(q2,q2))
        data = torch.Tensor(data)
        data = np.abs(nn.functional.normalize((data).view(1,-1)).numpy()).reshape(-1)
        normalized = torch.Tensor(data).view(1,-1)
        
        if(pad['padding_op']):
            new_arg = torch.cat((normalized[0], pad['pad_tensor']), dim=0)    
            normalized = torch.Tensor(new_arg).view(1,-1)
        
        latoutput = model(normalized, training_mode = False, return_latent =True, return_latent_vec = False) 
        latvec = model(normalized, training_mode = False, return_latent =True, return_latent_vec = True)
        output = model(normalized, training_mode = False, return_latent =False) 
        
        
        lat1,lat2,lat3 = qubitsToStateVector(latvec, 3)
        
        #m = np.kron(np.kron(lat1,lat2), lat3)
        # output = model(new_normed2, training_mode = False, return_latent =False) 
        
        #output2 = model(normalized, training_mode = True, return_latent =True)            
        #loss = test_loss((normalized**2).view(-1), output.view(-1))
            
        visualize(output.detach(), normalized ** 2 ,img_shape)
        visualize_state_vec(output , 'output' + str(batch_idx) , training_qubits_size)
        visualize_state_vec(normalized**2, 'data' + str(batch_idx),training_qubits_size)
        
        lat_out = latModel(normalized)
        gen_out = genModel(np.sqrt(latoutput.detach()))
        # visualize_state_vec(gen_out, 'gen_out' + str(batch_idx),training_qubits_size)
        # #print(lat_out)
        # visualize(gen_out.detach(), normalized ** 2 ,img_shape)
        visualize(lat_out.detach(), normalized ** 2 ,img_shape)
        # print(gen_out)
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
        

# %% 
U_gate = np.kron(np.kron(pauli_y, pauli_z), np.kron(pauli_y, pauli_z))
data = np.kron(np.kron(np.array(q1,dtype='float64'),q2), np.kron(np.array(q1,dtype='float64'),q2))
data = torch.Tensor(data)
data = np.abs(nn.functional.normalize((data).view(1,-1)).numpy()).reshape(-1)
normalized = torch.Tensor(data).view(1,-1)
output = model(normalized, training_mode = False, return_latent =False) 
q1 = np.zeros(2)
q1[1] = 1 
q2
