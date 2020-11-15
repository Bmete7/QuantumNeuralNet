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
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# %%

dataset = pd.read_csv('../dataset/iris.csv')

# transform species to numerics
dataset.loc[dataset.species=='Iris-setosa', 'species'] = 0
dataset.loc[dataset.species=='Iris-versicolor', 'species'] = 1
dataset.loc[dataset.species=='Iris-virginica', 'species'] = 2


train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:4]].values,
                                                    dataset.species.values, test_size=0.8)

# wrap up with Variable in pytorch
traindata = (torch.Tensor(train_X).float())
testdata = (torch.Tensor(test_X).float())

class IRISDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, traindata,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.traindata = traindata
        self.traindata = torch.cat( ( torch.cat(  (traindata.clone().detach(),traindata), dim = 1) ,  torch.cat(  (traindata.clone().detach(),traindata),dim = 1 )) , dim = 1)
        # self.traindata  = traindata
        self.transform = transform

    def __len__(self):
        # return len(self.traindata)
        return 1
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        
        data = self.traindata[idx]
         
        
        
        sample = {'data': data}

        if self.transform:
            sample = self.transform(sample)

        return sample
myData = IRISDataset(traindata)


dataloader = DataLoader(myData)




# %% Dataset + preprocessing 

# setting1
# n_samples = 200

test_loss = nn.MSELoss()
# setting2
n_samples = 2
# for4x4 
img_shape = 4
# X_train = datasets.MNIST(root='./data', train=True, download=True,
# transform=transforms.Compose([transforms.Resize((4,4)),transforms.ToTensor() , 
# transforms.Normalize((0.5,) , (1,))
# ]))


X_train = datasets.MNIST(root='./data', train=True, download=True,
transform=transforms.Compose([transforms.Resize((img_shape,img_shape)),transforms.ToTensor() 
]))


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
training_qubits_size = 4
n_qubits = training_qubits_size + latent_space_size  + auxillary_qubit_size
normalized = torch.Tensor(normalized).view(1,-1)

dev = qml.device("default.qubit", wires=n_qubits,shots = 1000)
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
            
            # qml.templates.StronglyEntanglingLayers(weights_st, range(latent_space_size+auxillary_qubit_size,n_qubits))
            
            self.first_rots = []
            self.final_rots = []
            self.c_nots = []
            self.cnot = []
            self.wires_list = []
            # Add entangling layer ??
            for i in range(latent_space_size + auxillary_qubit_size , n_qubits):
                ind = i - (latent_space_size + auxillary_qubit_size)
                self.first_rots.append(np.matrix((qml.Rot(*weights_r[0, ind], wires = i)).matrix).H)
            #Definition of unitary gate of the programmable circuit
            for i in range(latent_space_size + auxillary_qubit_size , n_qubits):
                ind = i - (latent_space_size + auxillary_qubit_size)
                ctr=0
                for j in range(latent_space_size+auxillary_qubit_size , n_qubits):
                    ind_contr = j - (latent_space_size + auxillary_qubit_size)
                    if(i==j):
                        pass
                    else:
                        self.cnot.insert( len(self.cnot) , np.matrix(( qml.CRot( *weights_cr[ind,ctr]  ,wires= [i,j]).matrix )).H )
                        self.wires_list.insert( len(self.wires_list) , [i,j])
                        ctr += 1
            for i in range(latent_space_size + auxillary_qubit_size , n_qubits):
                ind = i - (latent_space_size + auxillary_qubit_size)
                self.final_rots.append(np.matrix(qml.Rot(*weights_r[1, ind], wires = i).matrix).H)
                
            if(self.training_mode==True):
                self.SWAP_Test()
                return qml.probs(0)
            else:
                #qml.expval(qml.PauliZ(0))
                # return qml.probs(range(2*latent_space_size + auxillary_qubit_size, n_qubits ))
                
                qml.SWAP(wires = [1,2])
                
                for i in range(latent_space_size + auxillary_qubit_size , n_qubits):
                    ind = i - (latent_space_size + auxillary_qubit_size)
                    qml.QubitUnitary(self.final_rots[ind], wires = i)
                    print('final', self.final_rots)
                for i in range(len(self.cnot)):
                    qml.QubitUnitary(self.cnot.pop() , wires = self.wires_list.pop())
                
                for i in range(latent_space_size + auxillary_qubit_size , n_qubits):
                    ind = i - (latent_space_size + auxillary_qubit_size)
                    qml.QubitUnitary(self.first_rots[ind], wires = i)
                    print('rots', self.first_rots)
                
                return qml.probs(range(auxillary_qubit_size+latent_space_size,n_qubits ))
    
        weight_shapes = {"weights_r": (2 , training_qubits_size, 3),"weights_cr": (training_qubits_size,training_qubits_size-1 ,3), "weights_st":  (3,training_qubits_size,3)}
        weights_st = torch.tensor(qml.init.strong_ent_layers_uniform(3, training_qubits_size), requires_grad=True)
        self.qlayer = qml.qnn.TorchLayer(q_circuit, weight_shapes)
        self.DRAW_CIRCUIT_FLAG = True
        self.matrix_container = np.ndarray((training_qubits_size))
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
        if(training_mode == False):
            print(self.qlayer.qnode.draw())

        
        return x


# %%

def Fidelity_loss(measurements):
    
    fidelity = (2 * measurements[0] - 1.00)
    return torch.log(1- fidelity)


# %%

model = Net()


# %%
# Setting1
# learning_rate = 0.05 
# epochs = 10


# Setting2


learning_rate = 0.1
learning_rate = 0.01

epochs = 70

loss_list = []

# opt = torch.optim.SGD(model.parameters() , lr = learning_rate )
opt = torch.optim.Adam(model.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# loss_func = torch.nn.CrossEntropyLoss() # TODO replaced with fidelity
loss_func = Fidelity_loss
# loss_func = nn.MSELoss()

# TODO : save the initial parameters, then look at optimized ones.

pad_amount = int(2 ** (np.ceil(np.log2(training_qubits_size ** 2))) - normalized.view(-1,1).shape[0])
padding_op = False
if(pad_amount > 0 ):
    padding_op = True

    pad_tensor  = torch.zeros(pad_amount)



# model.qlayer.qnode_weights


# %%

for epoch in range(epochs):
    total_loss = []
    start_time = timeit.time.time()
    for i,datas in enumerate(train_loader):
        opt.zero_grad()
        
        # data = datas['data']
        # for iris dataseti
        
        data, target = datas
        # normalized = nn.functional.normalize(data.view(1,-1)).numpy().reshape(-1)
        # normalized = torch.Tensor(normalized).view(1,-1)
        # They do not have to be normalized since AmplitudeEmbeddings does that
        # But maybe we need it for the loss
        
        normalized = np.abs(nn.functional.normalize((data ).view(1,-1)).numpy()).reshape(-1)
        normalized = torch.Tensor(normalized).view(1,-1)
        if(padding_op):
            new_arg = torch.cat((normalized[0], pad_tensor), dim=0)    
        
            new_arg = torch.Tensor(new_arg).view(1,-1)
        
        
        
        #     start_time_in = timeit.time.time()
        if(padding_op):
            out = model(new_arg,True)
        else:
            out = model(normalized,True)
        
        
        # su measurementi bi print ettir
        # loss = loss_func(out)    
        loss = loss_func(out[0])
        loss.backward()
        if(i%10 == 0):
            print(out)
        opt.step()
        # output = model(normalized, training_mode = False)        
        # loss_test = test_loss((normalized**2).view(-1), output.view(-1))
        
        # print('Test loss: ', loss_test)
        
        # if(i%10 == 9):
        #     end_time_in = timeit.time.time()
        #     #print(out)
        #     print('Time elapsed: {:.2f}'.format(end_time_in-start_time_in))
        
    
        total_loss.append(loss.item())
    end_time = timeit.time.time()
    print('Time elapsed: {:.2f}'.format(end_time-start_time))
    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(100. * (epoch + 1) / epochs, loss_list[-1]))
    


# %%
n_samples = 10

# X_test = datasets.MNIST(root='./data', train=False, download=True,
#                         transform=transforms.Compose([transforms.Resize((4,4)),transforms.ToTensor()]))

# X_test = datasets.MNIST(root='./data', train=False, download=True,
#                         transform=transforms.Compose([transforms.Resize((6,6)),transforms.ToTensor()]))
X_test = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.Compose([transforms.Resize((img_shape,img_shape)),transforms.ToTensor()]))
# We do not need it anymore
# idx = np.append(np.where(X_test.targets == 0)[0][:n_samples], 
#                 np.where(X_test.targets == 1)[0][:n_samples])

X_test.data = X_test.data[:n_samples]
X_test.targets = X_test.targets[:n_samples]

test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)


# %%


    
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




# # %%

# def matplotlib_imshow(img, one_channel=False):
#     if one_channel:
#         img = img.mean(dim=0)
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     if one_channel:
#         plt.imshow(npimg, cmap="Greys")
#     else:
#         plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
# img_grid = torchvision.utils.make_grid(output.view(4,4))

# # show images
# matplotlib_imshow(img_grid, one_channel=True)

# writer = SummaryWriter('runs/fashion_mnist_experiment_1')
# writer.add_image('four_fashion_mnist_images', img_grid)

# writer.add_graph(model, data)
# writer.close()
# %% Some experiments, not project related



def visualize_state_vec(output , string):
    rep = output.view(1,-1)
    # xPoints = [ str(np.binary_repr(i)) for i in range(0,training_qubits_size**2)]
    xPoints = [ i for i in range(0,training_qubits_size**2)]
    yPoints = [rep[0,i] for i in range(0,training_qubits_size**2)]
    plt.bar(xPoints, yPoints)
    plt.suptitle(string)
    plt.ylim(top = 1) #xmax is your value
    plt.ylim(bottom = 0.00) #xmax is your value
    plt.show()
    


# %%
with torch.no_grad():
    correct = 0
    for batch_idx, datas in enumerate(train_loader):
        # TEST LOADERA DEGISMELI!!!
        # normalized = data.view(1,-1).numpy().reshape(-1)
        # normalized = torch.Tensor(normalized).view(1,-1)
        # data = data['data']
        # for iris dataset
        data,target = datas
        
        normalized = np.abs(nn.functional.normalize((data ).view(1,-1)).numpy()).reshape(-1)
        normalized = torch.Tensor(normalized).view(1,-1)
        if(padding_op):
            new_arg = torch.cat((normalized[0], pad_tensor), dim=0)    
            new_arg = torch.Tensor(new_arg).view(1,-1)
        
            output = model(new_arg, training_mode = False)        
            # loss = test_loss((new_arg**2).view(-1), output.view(-1))
        else:
            output = model(normalized, training_mode = False)        
            # loss = test_loss((normalized**2).view(-1), output.view(-1))
        shape_val = output.shape
        visualize(output, normalized ** 2 )
        visualize_state_vec(output , 'output' + str(batch_idx))
        visualize_state_vec(normalized**2, 'data' + str(batch_idx))
        
        print((normalized**2).view(-1))
        print(output.view(-1))
        print(' - - - ')
        if(batch_idx == 5):
            break
            
            
            
        total_loss.append(loss.item())

        
        # print(normalized)
    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / len(test_loader) * 100)
        )
# %% 
    
    