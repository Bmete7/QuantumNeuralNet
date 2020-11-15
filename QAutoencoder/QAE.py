# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:26:18 2020

@author: Burak Mete, M.Sc Informatik, Technische Universitat Muenchen

"""


'''
TODO: 

    All done

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


from skimage import io, transform

from torch.utils.data import Dataset, DataLoader

# %% PART 1a ) This is the IRIS Dataset implementation. This dataset is used for the development phase
# Since it has a very low dimensionality, easy to train and easy to visualize

# =============================================================================
# dataset = pd.read_csv('../dataset/iris.csv')
# 
# # transform species to numerics
# dataset.loc[dataset.species=='Iris-setosa', 'species'] = 0
# dataset.loc[dataset.species=='Iris-versicolor', 'species'] = 1
# dataset.loc[dataset.species=='Iris-virginica', 'species'] = 2
# 
# 
# train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:4]].values,
#                                                     dataset.species.values, test_size=0.8)
# 
# # wrap up with Variable in pytorch
# traindata = (torch.Tensor(train_X).float())
# testdata = (torch.Tensor(test_X).float())
# 
# class IRISDataset(Dataset):
#     """Face Landmarks dataset."""
# 
#     def __init__(self, traindata,transform=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         # self.traindata = traindata
#         self.traindata = torch.cat( ( torch.cat(  (traindata.clone().detach(),traindata), dim = 1) ,  torch.cat(  (traindata.clone().detach(),traindata),dim = 1 )) , dim = 1)
#         # self.traindata  = traindata
#         self.transform = transform
# 
#     def __len__(self):
#         # return len(self.traindata)
#         return 1
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
# 
#         
#         data = self.traindata[idx]
#          
#         
#         
#         sample = {'data': data}
# 
#         if self.transform:
#             sample = self.transform(sample)
# 
#         return sample
# myData = IRISDataset(traindata)
# 
# 
# dataloader = DataLoader(myData)
# =============================================================================

# %% Dataset + preprocessing 


n_samples = 30
img_shape = 8
batch_size = 1

# X_train = datasets.MNIST(root='./data', train=True, download=True,
# transform=transforms.Compose([transforms.Resize((4,4)),transforms.ToTensor() , 
# transforms.Normalize((0.5,) , (1,))
# ]))

X_train = datasets.MNIST(root='./data', train=True, download=True,
transform=transforms.Compose([transforms.Resize((img_shape,img_shape)),transforms.ToTensor() 
]))
# We actually do not need the targets, since we are training an autoenc.
X_train.data = X_train.data[:n_samples]
X_train.targets = X_train.targets[:n_samples]

train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)




# To get number of qubits dynamically
data,target = iter(train_loader).__next__()
normalized = data.view(1,-1).numpy().reshape(-1)

# Latent space size will be in N - lat_spa_size
latent_space_size = 1 # TODO determined according to reference size
auxillary_qubit_size = 1 # for the SWAP Test

training_qubits_size = int(np.ceil(np.log2(normalized.shape[0])))
n_qubits = training_qubits_size + latent_space_size  + auxillary_qubit_size

#normalized = torch.Tensor(normalized).view(1,-1)

dev = qml.device("default.qubit", wires=n_qubits,shots = 1000)
normalized = torch.Tensor(normalized).view(1,-1)
# Default qubit can be changed to gaussian, since it is very slow

# %% Whole network is defined within this class
class Net(nn.Module):
    def __init__(self):
    
        super(Net, self).__init__()
        
        # inputs shpould be a keyword argument, for AmplitudeEmbedding !!
        # then it becomes non-differentiable and the network suits with autograd
        self.training_mode = True
        self.return_latent = False
        # This class constitutes the whole network, which includes a 
        # data embeddings, parametric quantum circuit (encoder), SWAP TEST(Calculating the fidelity)
        # Also the conj. transpose of the encoder(decoder)
        
        
        @qml.qnode(dev)
        def q_circuit(weights_r ,weights_cr,weights_st ,inputs = False):
            self.embedding(inputs)
            
            # qml.templates.StronglyEntanglingLayers(weights_st, range(latent_space_size+auxillary_qubit_size,n_qubits))
            
            
            # These lists holds the conj tranposes of the programmable gates
            # since we would need them in the testing
            self.first_rots = []
            self.final_rots = []
            self.cnot = []
            self.wires_list = []
            
            
            # Single rotation gates for each qubit- Number of gates = N
            # Number of parameters = N * 3 
            for i in range(latent_space_size + auxillary_qubit_size , n_qubits):
                ind = i - (latent_space_size + auxillary_qubit_size)
                self.first_rots.append(np.matrix((qml.Rot(*weights_r[0, ind], wires = i)).matrix).H)
                
            # Controlled rotation gates for each qubit pair- Number of gates = N(N-1)/2
            # Number of parameters = 3* N(N-1)/2
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
            # Single rotation gates for each qubit- Number of gates = N
            # Number of parameters = N * 3                         
            for i in range(latent_space_size + auxillary_qubit_size , n_qubits):
                ind = i - (latent_space_size + auxillary_qubit_size)
                self.final_rots.append(np.matrix(qml.Rot(*weights_r[1, ind], wires = i).matrix).H)
                
            if(self.training_mode==True):
                self.SWAP_Test()
                return qml.probs(0)
            
            else:
                
                
                # In the testing, SWAP the Reference Bit and the trash states
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
        
        qml.templates.AmplitudeEmbedding(inputs, wires = range(latent_space_size+auxillary_qubit_size,n_qubits), normalize = True,pad=(0.j))
        #qml.QubitStateVector(inputs, wires = range(n_qubits))
    @qml.template 
    
    @qml.template
    def SWAP_Test(self):
        # SWAP Test measures the similarity between 2 qubits 
        # see https://arxiv.org/pdf/quant-ph/0102001.pdf
        
        qml.Hadamard(wires = 0)
        for i in range(auxillary_qubit_size, latent_space_size + auxillary_qubit_size):
            qml.CSWAP(wires = [0, i, i + latent_space_size])
        qml.Hadamard(wires = 0)
        
    def forward(self, x, training_mode = True, return_latent = False):
        self.training_mode = training_mode
        self.return_latent = return_latent
        x =  self.qlayer(x)
        
        #printing once before training
        if(self.DRAW_CIRCUIT_FLAG):
            self.DRAW_CIRCUIT_FLAG = False
            print(self.qlayer.qnode.draw())
            # Within Torch Object, you reach the circuit with TorchObj.qnode
        
        if(training_mode == False):
            print(self.qlayer.qnode.draw())

        
        return x


# %% Our implementation of the Loss

def Fidelity_loss(measurements):
    
    fidelity = (2 * measurements[0] - 1.00)
    return torch.log(1- fidelity)


# %%

model = Net()


# %% Model HyperParameters, batch size, number of epochs, optimizer,loss funct.

learning_rate = 0.1
epochs = 15
loss_list = []



# opt = torch.optim.SGD(model.parameters() , lr = learning_rate )
opt = torch.optim.Adam(model.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)



# loss_func = torch.nn.CrossEntropyLoss() # TODO replaced with fidelity
loss_func = Fidelity_loss

test_loss = nn.MSELoss()



# When we have n number of pixels, where n is not a square number (i.e 6x6 img)
# The data should be padded with 0's, since we are embedding the data to 
# ⌈log2(n)⌉ amount of qubits
pad_amount = int(2 ** (np.ceil(np.log2(training_qubits_size ** 2))) - normalized.view(-1,1).shape[0])
padding_op = False
if(pad_amount > 0 ):
    padding_op = True

    pad_tensor  = torch.zeros(pad_amount)




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
        
        normalized = np.abs(nn.functional.normalize((data ).view(1,-1)).numpy()).reshape(-1)
        normalized = torch.Tensor(normalized).view(1,-1)
        if(padding_op):
            new_arg = torch.cat((normalized[0], pad_tensor), dim=0)    
            new_arg = torch.Tensor(new_arg).view(1,-1)
        
        if(padding_op):
            out = model(new_arg,True)
        else:
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
    


# %% Test Dataloader + Preprocess
n_samples = 10


X_test = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.Compose([transforms.Resize((img_shape,img_shape)),transforms.ToTensor()]))

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
    
    data = data.view(1,1,img_shape,-1)
    
    axes[0].imshow(data[0].numpy().squeeze(), cmap='gray')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    out = out.view(1,1,img_shape,-1)

    axes[1].imshow(out[0].numpy().squeeze(), cmap='gray')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    plt.show()

# %% Plots of the probability distribution

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
    for batch_idx, datas in enumerate(test_loader):
        
        # for iris dataset
        # data = data['data']
        data,target = datas
        
        normalized = np.abs(nn.functional.normalize((data ).view(1,-1)).numpy()).reshape(-1)
        normalized = torch.Tensor(normalized).view(1,-1)
        
        if(padding_op):
            new_arg = torch.cat((normalized[0], pad_tensor), dim=0)    
            new_arg = torch.Tensor(new_arg).view(1,-1)
            output = model(new_arg, training_mode = False)        
        else:
            output = model(normalized, training_mode = False, return_latent = True)        
            
            loss = test_loss((normalized**2).view(-1), output.view(-1))
            
        visualize(output, normalized ** 2 )
        visualize_state_vec(output , 'output' + str(batch_idx))
        visualize_state_vec(normalized**2, 'data' + str(batch_idx))
        
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
    
    