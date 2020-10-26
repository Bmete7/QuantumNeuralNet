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

## TODO ! 
# Neural network layers will be replaced by swap gate, and
# the autoencoder paper will be embodided


import numpy as np
import pennylane as qml
import torch
import sklearn.datasets

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
#n_qubits=7 
dev = qml.device("default.qubit", wires=n_qubits)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        
        # inputs shpould be a keyword arguement, then it becomes 
        #non-differentiable and the network suits with autograd
        
        @qml.qnode(dev)
        def my_circuit(weights,weights_1,inputs = False):
            self.embed(inputs)
            
            for i in range(n_qubits):
                
                qml.Rot(*weights[0, i], wires = i)
            for i in range(n_qubits):
                ctr=0
                for j in range(n_qubits):
                    
                    if(i==j):
                        continue
                    else:
                        qml.CRot( *weights_1[i,ctr]  ,wires= [i,j])
                        ctr += 1
            for i in range(n_qubits):
                qml.Rot(*weights[1, i], wires = i)
            #         print(i)
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1)),qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(3))
            # return qml.probs(range(n_qubits))
        
        weight_shapes = {"weights": ( 2 , n_qubits, 3),"weights_1": (n_qubits,n_qubits-1 ,3)}
        
        self.qlayer = qml.qnn.TorchLayer(my_circuit, weight_shapes)
        
        self.clayer1 = torch.nn.Linear(4,2)
        self.clayer2 = torch.nn.Linear(2,2)
        self.softmax = torch.nn.Softmax(dim=1)
        #self.seq = torch.nn.Sequential(self.qlayer, self.clayer1)
    @qml.template
    def embed(self,inputs):
        #qml.QubitStateVector(inputs, wires = range(n_qubits))
        qml.templates.AmplitudeEmbedding(inputs, wires = range(n_qubits), normalize = True)
        
    
    def forward(self, x):

        
        x =  self.qlayer(x)
        
        x = self.clayer1(x)
        
        x = self.clayer2(x)
        
        x = self.softmax(x)
        
        # For now, NN layers are just dummy
        return x



        
dev_embedding = qml.device("default.qubit", wires=n_qubits,shots = 1000)

@qml.qnode(dev)
def Embedding_circuit(f=None):
    qml.templates.AmplitudeEmbedding(features=f, wires=range(4),normalize = True)
    return qml.probs(range(n_qubits))



class_model = Net()

opt = torch.optim.SGD(class_model.parameters() , lr = 0.1)
loss_func = torch.nn.CrossEntropyLoss()

samples = 100
epochs = 50
batch_size = 1
batches = samples // batch_size
loss_list = []


for epoch in range(epochs):
    total_loss = []
    for i,datas in enumerate(train_loader):
        opt.zero_grad()
        data,target = datas
        normalized = data.view(1,-1).numpy().reshape(-1)
        #intermed_val = Embedding_circuit(normalized.detach().clone())
        normalized = torch.Tensor(normalized).view(1,-1)
        # res = amp_embedding(normalized.numpy().reshape(-1))     
        # res = res.astype('double') 
        # res = torch.tensor(res).view(1,-1).float()
        # print('--')
        # print(res)
        # out = class_model(res )
        start = timeit.time.time()
        out = class_model(normalized )
        
        end = timeit.time.time()
        print('Time elapsed: {:2.2f%}' ,end-start)
        
        start = timeit.time.time()
        loss = loss_func(out, target)
        end = timeit.time.time()
        print('Time elapsed: {:2.2f%}' ,end-start)
        print(out)
        start = timeit.time.time()
        loss.backward()
        end = timeit.time.time()
        print( 'Time elapsed: {:.2f}'.format(end-start))
        start = timeit.time.time()
        opt.step()
        end = timeit.time.time()
        print('Time elapsed: {:2.2f%}' ,end-start)
        total_loss.append(loss.item())
    break
    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(100. * (epoch + 1) / epochs, loss_list[-1]))
    




n_samples = 50

X_test = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.Compose([transforms.Resize((4,4)),transforms.ToTensor()]))

idx = np.append(np.where(X_test.targets == 0)[0][:n_samples], 
                np.where(X_test.targets == 1)[0][:n_samples])

X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]

test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)





with torch.no_grad():
    
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        normalized = nn.functional.normalize(data.view(1,-1)).numpy().reshape(-1)
        normalized = torch.Tensor(normalized).view(1,-1)
        output = class_model(normalized)
        print(output)
        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        loss = loss_func(output, target)
        total_loss.append(loss.item())
        
    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / len(test_loader) * 100)
        )

    
n_samples_show = 6
count = 0
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

class_model.eval()
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        if count == n_samples_show:
            break
        
        normalized = nn.functional.normalize(data.view(1,-1)).numpy().reshape(-1)
        normalized = torch.Tensor(normalized).view(1,-1)
        
        output = class_model(normalized)
        
        pred = output.argmax(dim=1, keepdim=True) 

        axes[count].imshow(data[0].numpy().squeeze(), cmap='gray')

        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title('Predicted {}'.format(pred.item()))
        
        count +=1  
    
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