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
import random
import pennylane as qml
import qiskit
from qiskit.visualization import *

import sklearn.datasets
import matplotlib.pyplot as plt
import timeit
from torch.utils.tensorboard import SummaryWriter



import torchvision





# 

# features would be non-differentiable with pennylanes framework
# however we need gradients only up-to Unitary gate

# TODO ! 
# 1+ Neural network layers will be replaced by swap gate
# 2+ Loss function(CrossEntropy) will be replaced with fidelity (SWAP test)
# + Maybe we can optimize with pennyLane instead of Torch? Look into it.


# In testing, you measure the training qubits, in training you only measure the ancilla

#Ask Irene if its meaningful to visualize encoded/trained latent space, and the output in the presentation?

## CHANGE DEVICE FROM DEFAULT TO GAUSSIAN

# %% Dataset + preprocessing 
n_samples = 2
# for4x4 

X_train = datasets.MNIST(root='./data', train=True, download=True,
transform=transforms.Compose([transforms.Resize((4,4)),transforms.ToTensor() , 
transforms.Normalize((0.5,) , (1,))
]))


# X_train = datasets.MNIST(root='./data', train=True, download=True,
# transform=transforms.Compose([transforms.Resize((6,6)),transforms.ToTensor() , 
# transforms.Normalize((0.5,) , (1,))
# ]))


# Leaving only labels 0 and 1, only for now, we dont need it anymore
# idx = np.append(np.where(X_train.targets == 0)[0][:n_samples], 
#                 np.where(X_train.targets == 1)[0][:n_samples])


X_train.data = X_train.data[:n_samples]
X_train.targets = X_train.targets[:n_samples]

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
        self.training_mode = True
        @qml.qnode(dev)
        def q_circuit(weights_r ,weights_cr,weights_st ,inputs = False):
            self.embedding(inputs)
            
            qml.templates.StronglyEntanglingLayers(weights_st, range(latent_space_size+auxillary_qubit_size,n_qubits))
            
            # Add entangling layer ??
            for i in range(latent_space_size + auxillary_qubit_size , n_qubits):
                ind = i - (latent_space_size + auxillary_qubit_size)
                qml.Rot(*weights_r[0, ind], wires = i)
            #Definition of unitary gate of the programmable circuit
            for i in range(latent_space_size + auxillary_qubit_size , n_qubits):
                ind = i - (latent_space_size + auxillary_qubit_size)
                ctr=0
                for j in range(latent_space_size+auxillary_qubit_size , n_qubits):
                    ind_contr = j - (latent_space_size + auxillary_qubit_size)
                    if(i==j):
                        continue
                    else:
                        qml.CRot( *weights_cr[ind_contr,ctr]  ,wires= [i,j])
                        ctr += 1
            for i in range(latent_space_size + auxillary_qubit_size , n_qubits):
                ind = i - (latent_space_size + auxillary_qubit_size)
                qml.Rot(*weights_r[1, ind], wires = i)

            if(self.training_mode==True):
                self.SWAP_Test()
                return qml.expval(qml.PauliZ(0))
            else:
                #qml.expval(qml.PauliZ(0))
                return qml.probs(range(auxillary_qubit_size+latent_space_size,n_qubits ))
    
        weight_shapes = {"weights_r": (2 , training_qubits_size, 3),"weights_cr": (training_qubits_size,training_qubits_size-1 ,3), "weights_st":  (3,training_qubits_size,3)}
        weights_st = torch.tensor(qml.init.strong_ent_layers_uniform(3, training_qubits_size), requires_grad=True)
        self.qlayer = qml.qnn.TorchLayer(q_circuit, weight_shapes)
        self.DRAW_CIRCUIT_FLAG = True
# =============================================================================
#         4 -Time elapsed: 0.50
#         5 -Time elapsed: 1.07
#         6 -Time elapsed: 1.92
#         7 -Time elapsed: 3.18
#         8- Time elapsed: 5.44
# =============================================================================
        # Those should be replaced with the swap test
        # self.clayer1 = torch.nn.Linear(64,2)
        # self.clayer2 = torch.nn.Linear(2,2)
        # self.softmax = torch.nn.Softmax(dim=1)
        #self.seq = torch.nn.Sequential(self.qlayer, self.clayer1)
    @qml.template
    def embedding(self,inputs):
        # Most Significant qubit is the ancilla, then reference, then subsystem B then subystem A
        # - Ancilla
        # - Refernce
        # - B (Trash State)
        # - A (Latent Space)
        # When normalize flag is True, features act like a prob. distribution
        
        # print(inputs)
        qml.templates.AmplitudeEmbedding(inputs, wires = range(latent_space_size+auxillary_qubit_size,n_qubits), normalize = True,pad=(0.j))
        
        #qml.QubitStateVector(inputs, wires = range(n_qubits))
        
    @qml.template
    def SWAP_Test(self):
        
        qml.Hadamard(wires = 0)
        for i in range(auxillary_qubit_size, latent_space_size + auxillary_qubit_size):
            qml.CSWAP(wires = [0, i, i + latent_space_size])
        qml.Hadamard(wires = 0)
        
    def forward(self, x, training_mode = True):
        self.training_mode = training_mode
        x =  self.qlayer(x)
        
        #printing once before training
        if(self.DRAW_CIRCUIT_FLAG):
            self.DRAW_CIRCUIT_FLAG = False
            
            # Within Torch Object, you reach the circuit with TorchObj.qnode
            print(self.qlayer.qnode.draw())

        
        return x


# %%

def Fidelity_loss(measurements):
    fidelity = (2 * measurements - 1.00)
    return torch.log(1-fidelity)


# %%

model = Net()
learning_rate = 0.05 
samples = 200
epochs = 10
batch_size = 1
batches = samples // batch_size
loss_list = []

opt = torch.optim.SGD(model.parameters() , lr = learning_rate )
# loss_func = torch.nn.CrossEntropyLoss() # TODO replaced with fidelity
loss_func = Fidelity_loss


# TODO : save the initial parameters, then look at optimized ones.

pad_amount = int(2 ** (np.ceil(np.log2(training_qubits_size ** 2))) - training_qubits_size ** 2)
padding_op = False
if(pad_amount > 0 ):
    padding_op = True

pad_tensor  = torch.zeros(pad_amount)



# %%
for epoch in range(epochs):
    total_loss = []
    start_time = timeit.time.time()
    for i,datas in enumerate(train_loader):
        opt.zero_grad()
        
        data,target = datas
        # normalized = nn.functional.normalize(data.view(1,-1)).numpy().reshape(-1)
        # normalized = torch.Tensor(normalized).view(1,-1)
        # They do not have to be normalized since AmplitudeEmbeddings does that
        # But maybe we need it for the loss
        
        normalized = np.abs(nn.functional.normalize(data.view(1,-1)).numpy()).reshape(-1)
        normalized = torch.Tensor(normalized).view(1,-1)
        if(padding_op):
            new_arg = torch.cat((normalized[0], pad_tensor), dim=0)    
        
            new_arg = torch.Tensor(new_arg).view(1,-1)
        
        
        if(i%10 == 0):
            start_time_in = timeit.time.time()
        if(padding_op):
            out = model(new_arg,True)
        else:
            out = model(normalized,True)
        loss = loss_func(out)    
        loss.backward()
        opt.step()
        if(i%10 == 9):
            end_time_in = timeit.time.time()
            # print(out)
            print('Time elapsed: {:.2f}'.format(end_time_in-start_time_in))
        
    
        total_loss.append(loss.item())
    end_time = timeit.time.time()
    print('Time elapsed: {:.2f}'.format(end_time-start_time))
    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(100. * (epoch + 1) / epochs, loss_list[-1]))
    


# %%
n_samples = 4

X_test = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.Compose([transforms.Resize((4,4)),transforms.ToTensor()]))

# X_test = datasets.MNIST(root='./data', train=False, download=True,
#                         transform=transforms.Compose([transforms.Resize((6,6)),transforms.ToTensor()]))
# We do not need it anymore
# idx = np.append(np.where(X_test.targets == 0)[0][:n_samples], 
#                 np.where(X_test.targets == 1)[0][:n_samples])

X_test.data = X_test.data[:n_samples]
X_test.targets = X_test.targets[:n_samples]

test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)


# %%

test_loss = nn.MSELoss()

    
def visualize(out,data):
    
    #unnormalizing the output:
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
    # if(padding_op):
    #     print((out[:pad_amount].shape))
    #     unnormed_out = (out[0][:(len(output[0]) - pad_amount)].view(1,-1)  * np.sqrt((data**2).sum().numpy())).view(1,1,training_qubits_size,-1)
    # else:
    #     unnormed_out = (out  * np.sqrt((data**2).sum().numpy())).view(1,1,training_qubits_size,-1)
    
    data = data.view(1,1,training_qubits_size,-1)
    
    count = 0
    axes[count].imshow(data[0].numpy().squeeze(), cmap='gray')

    axes[count].set_xticks([])
    axes[count].set_yticks([])
    out = out.view(1,1,training_qubits_size,-1)
    count+=1
    axes[count].imshow(out[0].numpy().squeeze(), cmap='gray')

    axes[count].set_xticks([])
    axes[count].set_yticks([])
    plt.show()

# %%
with torch.no_grad():
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        # normalized = data.view(1,-1).numpy().reshape(-1)
        # normalized = torch.Tensor(normalized).view(1,-1)
        
        normalized = nn.functional.normalize(data.view(1,-1)).numpy().reshape(-1)
        
        normalized = torch.Tensor(normalized).view(1,-1)
        if(padding_op):
            new_arg = torch.cat((normalized[0], pad_tensor), dim=0)    
            new_arg = torch.Tensor(new_arg).view(1,-1)
        
            output = model(new_arg, training_mode = False)        
            loss = test_loss((new_arg**2).view(-1), output.view(-1))
        else:
            output = model(normalized, training_mode = False)        
            loss = test_loss((normalized**2).view(-1), output.view(-1))
        visualize((output)**(1/2), normalized)
        visualize_state_vec(output , 'output' + str(batch_idx))
        visualize_state_vec(normalized**2, 'data' + str(batch_idx))
        if(batch_idx == 1):
            print((normalized**2).view(-1))
            print(output.view(-1))
        total_loss.append(loss.item())

        
        # print(normalized)
    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / len(test_loader) * 100)
        )

# %%

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
img_grid = torchvision.utils.make_grid(output.view(4,4))

# show images
matplotlib_imshow(img_grid, one_channel=True)

writer = SummaryWriter('runs/fashion_mnist_experiment_1')
writer.add_image('four_fashion_mnist_images', img_grid)

writer.add_graph(model, data)
writer.close()
# %% Some experiments, not project related



def visualize_state_vec(output , string):
    rep = output.view(1,-1)
    # xPoints = [ str(np.binary_repr(i)) for i in range(0,training_qubits_size**2)]
    xPoints = [ i for i in range(0,training_qubits_size**2)]
    yPoints = [rep[0,i] for i in range(0,training_qubits_size**2)]
    plt.bar(xPoints, yPoints)
    plt.suptitle(string)
    plt.show()

