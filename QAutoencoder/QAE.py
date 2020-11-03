# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:26:18 2020

@author: burak
"""



'''
TODO: 
    DONE + import MNIST DATASET, transformit to 4x4 (Check if 8x8 has eligible compute efficiency also)
    DONE + Do a Quantum State Embedding, Each pixel-a qubit(or think of a more intelligent solution)
    DONE + Build the programmable circuit represented on the paper, use the rotation gate approach
    - Train the network and get some results
    Exp. 21.10.2020

'''


# %% Imports
import numpy as np
import torch
from torch.autograd import Function
from torchvision import datasets,transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import pennylane as qml
import qiskit
from qiskit.visualization import *

import sklearn.datasets
import matplotlib.pyplot as plt
import timeit

# 

# features would be non-differentiable with pennylanes framework
# however we need gradients only up-to Unitary gate

# TODO ! 
# 1- Neural network layers will be replaced by swap gate
# 2- Loss function(CrossEntropy) will be replaced with fidelity (SWAP test)

# ?- Maybe we can optimize with pennyLane instead of Torch? Look into it.


# %% Dataset + preprocessing 
n_samples = 100

X_train = datasets.MNIST(root='./data', train=True, download=True,
transform=transforms.Compose([transforms.Resize((4,4)),transforms.ToTensor() , 
transforms.Normalize((0.5,) , (1,))
]))

# Leaving only labels 0 and 1, only for now
idx = np.append(np.where(X_train.targets == 0)[0][:n_samples], 
                np.where(X_train.targets == 1)[0][:n_samples])

X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]

train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)

# To get number of qubits dynamically
data,target = iter(train_loader).__next__()
normalized = data.view(1,-1).numpy().reshape(-1)
latent_space_size = 1 # TODO determined according to reference size
auxillary_qubit_size = 1

training_qubits_size = int(np.ceil(np.log2(normalized.shape[0])))
n_qubits = training_qubits_size + latent_space_size + auxillary_qubit_size
normalized = torch.Tensor(normalized).view(1,-1)

dev = qml.device("default.qubit", wires=n_qubits)
# Default qubit must be changed to gaussian, since it is very slow

# %% Whole network is defined within this class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # inputs shpould be a keyword arguement, for AmplitudeEmbedding
        # then it becomes non-differentiable and the network suits with autograd
        
        @qml.qnode(dev)
        def q_circuit(weights_r ,weights_cr ,inputs = False):
            self.embedding(inputs)
            
            # Add entangling layer ??
            
            #qml.Hadamard(wires= 0)
            #Definition of unitary gate of the programmable circuit
            for i in range(latent_space_size + auxillary_qubit_size , n_qubits):
                ind = i - (latent_space_size + auxillary_qubit_size)
                qml.Rot(*weights_r[0, ind], wires = i)
                ctr=0
                for j in range(latent_space_size+auxillary_qubit_size , n_qubits):
                    ind_contr = j - (latent_space_size + auxillary_qubit_size)
                    if(i==j):
                        continue
                    else:
                        qml.CRot( *weights_cr[ind_contr,ctr]  ,wires= [i,j])
                        ctr += 1
                qml.Rot(*weights_r[1, ind], wires = i)
                qml.SWAP(wires = range(auxillary_qubit_size, auxillary_qubit_size + 2*latent_space_size))
            # return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)),qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3)),qml.expval(qml.PauliZ(4))
            return qml.probs(range(n_qubits))
        
        weight_shapes = {"weights_r": (2 , training_qubits_size, 3),"weights_cr": (training_qubits_size,training_qubits_size-1 ,3)}
        
        self.qlayer = qml.qnn.TorchLayer(q_circuit, weight_shapes)
# =============================================================================
#         4 -Time elapsed: 0.50
#         5 -Time elapsed: 1.07
#         6 -Time elapsed: 1.92
#         7 -Time elapsed: 3.18
#         8- Time elapsed: 5.44
# =============================================================================
        # Those should be replaced with the swap test
        self.clayer1 = torch.nn.Linear(64,2)
        self.clayer2 = torch.nn.Linear(2,2)
        self.softmax = torch.nn.Softmax(dim=1)
        #self.seq = torch.nn.Sequential(self.qlayer, self.clayer1)
    @qml.template
    def embedding(self,inputs):
        # Most Significant qubit is the ancilla, then reference, then subsystem B then subystem A
        # - Ancilla
        # - Refernce
        # - B (Trash State)
        # - A (Latent Space)
        # When normalize flag is True, features act like a prob. distribution
        qml.templates.AmplitudeEmbedding(inputs, wires = range(latent_space_size+auxillary_qubit_size,n_qubits), normalize = True)
        #qml.QubitStateVector(inputs, wires = range(n_qubits))
        
    def forward(self, x):

        x =  self.qlayer(x)
        print(x)
        print('****')
        x = self.clayer1(x)
        x = self.clayer2(x)
        x = self.softmax(x)
        
        # For now, NN layers are just dummy
        return x

# %%

model = Net()
learning_rate = 0.5
samples = 100
epochs = 1
batch_size = 1
batches = samples // batch_size
loss_list = []

opt = torch.optim.SGD(model.parameters() , lr = learning_rate )
loss_func = torch.nn.CrossEntropyLoss() # TODO replaced with fidelity

# %%
for epoch in range(epochs):
    total_loss = []
    start_time = timeit.time.time()
    for i,datas in enumerate(train_loader):
        opt.zero_grad()
        
        data,target = datas
        # They do not have to be normalized since AmplitudeEmbeddings does that
        normalized = data.view(1,-1).numpy().reshape(-1) 
        normalized = torch.Tensor(normalized).view(1,-1)

        
        
        out = model(normalized)
        loss = loss_func(out, target)                
        loss.backward()
        opt.step()
        
        
    
        total_loss.append(loss.item())
        break
    end_time = timeit.time.time()
    print('Time elapsed: {:.2f}'.format(end_time-start_time))
    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(100. * (epoch + 1) / epochs, loss_list[-1]))
    



# %%
n_samples = 1

X_test = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.Compose([transforms.Resize((4,4)),transforms.ToTensor()]))

idx = np.append(np.where(X_test.targets == 0)[0][:n_samples], 
                np.where(X_test.targets == 1)[0][:n_samples])

X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]

test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)




# %%
with torch.no_grad():
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        normalized = data.view(1,-1).numpy().reshape(-1)
        normalized = torch.Tensor(normalized).view(1,-1)
        
        output = model(normalized)
        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        loss = loss_func(output, target)
        total_loss.append(loss.item())
        
    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / len(test_loader) * 100)
        )

    # %%
n_samples_show = 6
count = 0
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

model.eval()
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        if count == n_samples_show:
            break
        
        normalized = nn.functional.normalize(data.view(1,-1)).numpy().reshape(-1)
        normalized = torch.Tensor(normalized).view(1,-1)
        
        output = model(normalized)
        
        pred = output.argmax(dim=1, keepdim=True) 

        axes[count].imshow(data[0].numpy().squeeze(), cmap='gray')

        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title('Predicted {}'.format(pred.item()))
        
        count +=1  

