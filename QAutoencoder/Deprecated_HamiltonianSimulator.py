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


import qiskit
import torchvision


from torch.utils.data import Dataset, DataLoader
import sys

sys.path.append('../dataLoad')

sys.path.append('../PennyLane')
sys.path.append('../representation')
sys.path.append('../QAutoencoder')
sys.path.append('../Hamiltonian')
sys.path.append('../tests')
# from IrisLoad import importIRIS, IRISDataset
# from MNISTLoad import importMNIST
from qCircuit import EmbedNet

# from visualizations import visualize, visualize_state_vec
from copy import deepcopy
# import seaborn
# import tensorflow as tf
# from scipy.linalg import expm, sinm, cosm
from measurement_tools import *
from TimeEvolution import *
from HermitianDecomposition import *
from complex_circuit import *
from pennylane import numpy as np
from unittests import *
#%% numpy implementations for the gates

I  = np.array(((0j + 1, 0), (0, 1)))
pauli_x  = np.array(((0j, 1), (1, 0)))
pauli_y = np.array(((0, -1j), (1j, 0)))
pauli_z = np.array(((1+0j, 0), (0, -1)))
paulis = [I,pauli_x,pauli_y, pauli_z]
# %% 
model_saved =False

# %% Fidelity loss calculation 

def Fidelity_loss(mes):
    tot  =0
    for i in mes:
        tot += i[0]
    fidelity = (2 * (tot) / len(mes[0])  - 1.00)
    return torch.log(1- fidelity)
    

# %% Preparing 2-qubit Data
import numpy as np
n_embed_samples = 10 # how many training samples there are
#Structure of the dataset [in1, in2 . .  , evol1_t1 , evol1_t2 , evol1_t3, evol2_t1 . . . evoln_t3] 
pauli_list = [I,pauli_x, pauli_y, pauli_z]
observables = [np.kron(i,j) for i in pauli_list for j in pauli_list]
# Create random hamiltonians from pauli coefficients
hamiltonians = []
coefs = []
input_states = np.zeros((n_embed_samples*4,4), dtype ='complex128')
# time_steps = 3 # how many time evolution steps are going to be simulated
time_steps = 3 # how many time evolution steps are going to be simulated
for i in range(n_embed_samples):
    coef = np.random.rand(16) * np.random.rand() + np.random.rand()   
    hamiltonian = np.einsum('i,ijk->jk', coef,observables)
    coefs.append(coef)
    hamiltonians.append(hamiltonian)
    eig,eiv = np.linalg.eig(hamiltonian)
    state = eiv[np.random.randint(4)]
    input_states[i] = state
    for t in range(1, time_steps+ 1):
        input_states[i+ (n_embed_samples * (t))] = time_evolution_simulate(hamiltonian, t, state )
        
n_inputs =len(input_states)
# Unit test, to see if dataset creates states that are actually correspond to a valid time evolution
TimeEvolutionTest(hamiltonians, 1, input_states,n_embed_samples,0)


# %% 4 qubit data
from pennylane import numpy as np

n_embed_samples = 20  
pauli_list = [I,pauli_x, pauli_y, pauli_z]
observables = [np.kron(np.kron(np.kron(i,j), k), m) for k in pauli_list for m in pauli_list for i in pauli_list for j in pauli_list]
# Create random hamiltonians from pauli coefficients
hamiltonians = []
coefs = []
input_states = np.zeros((n_embed_samples*4,16), dtype ='complex128')
time_steps = 3 # how many time evolution steps are going to be simulated
for i in range(n_embed_samples):
    coef = np.random.rand(256) * np.random.rand() + np.random.rand()
    coefs.append(coef)
    hamiltonian = np.einsum('i,ijk->jk', coef,observables)
    hamiltonians.append(hamiltonian)
    eig,eiv = np.linalg.eig(hamiltonian)
    state = eiv[np.random.randint(16)]
    input_states[i] = state
    for t in range(1, time_steps+ 1):
        input_states[i+ (n_embed_samples * (t))] = time_evolution_simulate(hamiltonian, t, state )
n_inputs =len(input_states)        
input_states.requires_grad = False
# Unit test, to see if dataset creates states that are actually correspond to a valid time evolution
TimeEvolutionTest(hamiltonians, 1, input_states,n_embed_samples,0)

# %% 
embed_model_qae = []
dev_embed_qae = []
model_params = []
# %%
from qCircuit import EmbedNet


dev_embed = qml.device("default.qubit", wires=6,shots = 1000)
embed_model = EmbedNet(dev_embed,input_states, 1, 6, 4, 1)
    
epochs = 100
loss_list = []

loss_func = Fidelity_loss
learning_rate = 0.01
opt = torch.optim.Adam(embed_model.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
average_fidelities = []
batch_size = 4
for epoch in range(epochs):
    all_outs = 0
    running_loss = 0
    total_loss = []
    start_time = timeit.time.time()
    #for i in batch_id:
    batch_id = np.arange(batch_size) #
    np.random.shuffle(batch_id)
    for i in batch_id:
        opt.zero_grad()
        out = embed_model(torch.zeros([2]), True)
        all_outs += out[0][0]
        
        loss = loss_func(out)
        loss.backward()
        # print(out)
        running_loss += loss
        
        opt.step()
        total_loss.append(loss.item())
    end_time = timeit.time.time()
    
    print('Time elapsed for the epoch' +  str(epoch)  + ' : {:.2f}'.format(end_time-start_time))
    loss_list.append(running_loss/batch_size)
    average_fidelities.append(all_outs / batch_size )
    print('Training [{:.0f}%]\tLoss: {:.4f}\t Average Fidelity {:.4f}'.format(100. * (epoch + 1) / epochs,  running_loss / batch_size , all_outs/ batch_size ))
    if (running_loss /n_inputs < 0.000015):
        print('loss achieved')
        break   

    # embed_model_qae.append(embed_model)
    # dev_embed_qae.append(dev_embed)
    # model_params.append(opt.param_groups[0]['params'])

# %% Important parameters are saved for the trained network
    np.save('input_states.npy' , input_states)
    np.save('coefs.npy' , coefs)
    np.save('hamiltonians.npy' , hamiltonians)
    np.save('dev_embed_qae.npy' , dev_embed_qae)
    np.save('model_params.npy' , model_params)
    


# %% fidelities plot, How many qubits yield which fidelities
batch_id = np.arange(n_embed_samples)
np.random.shuffle(batch_id)
losses = np.zeros((n_embed_samples,1))
for i in batch_id:
    normalized = torch.Tensor(embed_features[i])
    out = embed_model(normalized,True)
    losses[i] = loss_func(out).detach().numpy()
import matplotlib
import matplotlib.pyplot as plt
w = 4
h = 3
d = 70
plt.figure(figsize=(w, h), dpi=d)
plt.hist(average_fidelities, bins=np.arange(0.5,1, 0.0525))

plt.plot(average_fidelities)
plt.xlabel('Epochs')
plt.ylabel('Fidelity')
# %% SAVE AND LOAD THE MODULE 
PATH = './autoencoder.npy'

if(model_saved == True):
    embed_model.load_state_dict(torch.load(PATH))
    print('model loaded')
else:
    PATH = './autoencoder_more_qubits.npy'
    torch.save(embed_model.state_dict(), PATH)
print('model parameters: ',  embed_model.state_dict)


 

# %% Parameters of the embed_model extracted


    
first_rot_params_0 = opt.param_groups[0]['params'][0][0][0]
first_rot_params_1 = opt.param_groups[0]['params'][0][0][1]

crot_params_01 = opt.param_groups[0]['params'][1][0][0]
crot_params_10 = opt.param_groups[0]['params'][1][1][0]

final_rot_params_0 = opt.param_groups[0]['params'][0][1][0]
final_rot_params_1 = opt.param_groups[0]['params'][0][1][1]

first_rot_params_0 = first_rot_params_0.detach().numpy()
first_rot_params_1 = first_rot_params_1.detach().numpy()

final_rot_params_0 = final_rot_params_0.detach().numpy()
final_rot_params_1 = final_rot_params_1.detach().numpy()

crot_params_10 = crot_params_10.detach().numpy()
crot_params_01 = crot_params_01.detach().numpy()

 
# %% 
def Fidelity_Loss_Prob(mes):    
    
    fidelity = 2*mes-1.00
    return torch.log(1- fidelity)

# %% Hamiltonian Approximator
n_qubit = 4
# Fidelity_loss = Fidelity_Loss_Prob
@qml.qnode(ham_dev, interface="torch")
def quantum_net(q_weights_flat,ind):
    """
    The variational quantum circuit.
    """
    inp = input_states[ind]
    
    inp = np.kron(inp, input_states[ind])    
    
    qml.templates.embeddings.AmplitudeEmbedding(inp,wires = range(0, n_qubit * 2 ), normalize = True,pad=(0.j))
    
    
    for i in range(0,4):
        
        qml.Rot(*model_params[ind][0][0][i] , wires = i)
        qml.Rot(*model_params[ind][0][0][i] , wires = i+4)
    
        for i in range(4):
            index = 0
            for j in range(4):
                if(i==j):
                    continue
                qml.CRot(*model_params[ind][1][i][index], wires = [i,j])
                qml.CRot(*model_params[ind][1][i][index], wires = [i+4,j+4])
                index+=1
    
    for i in range(0,4):    
        qml.Rot(*model_params[ind][0][1][i] , wires = i)
        qml.Rot(*model_params[ind][0][1][i] , wires = i+4)
            
    
    for i in range(0,4):
        qml.Rot(*model_params[ind ][0][0][i] , wires = i + 4)
        
    for i in range(4):
        index = 0
        for j in range(4):
            if(i==j):
                continue
            qml.CRot(*model_params[ind][1][i][index], wires = [i+4,j+4])
            index+=1
        
    for i in range(0,4):
        qml.Rot(*model_params[ind][0][1][i] , wires = i+ 4)
        
    
    for i in range(0,3):
        qml.Rot(*q_weights_flat[0 + 3*i :3 + 3*i], wires = i + 1)
        qml.Rot(*q_weights_flat[0 + 3*i :3 + 3*i], wires = i + 1 + 4)
    ctr = 0 
    for i in range(3):
        for j in range(3):
            if(i==j):
                continue
            qml.CRot(*q_weights_flat[9 + ctr*3 : 12 + ctr *3], wires = [i+1,j+1])
            qml.CRot(*q_weights_flat[9 + ctr*3 : 12 + ctr *3], wires = [i+1 + 4,j+1 + 4])
            ctr += 1
    for i in range(0,3):
        qml.Rot(*q_weights_flat[27 + 3*i :30 + 3*i], wires = i + 1)
        qml.Rot(*q_weights_flat[27 + 3*i :30 + 3*i], wires = i + 1 + 4)
    
    for i in range(0,3):
    
        qml.Rot(*q_weights_flat[0 + 3*i :3 + 3*i], wires = i + 1 + 4)
    ctr = 0 
    for i in range(3):
        for j in range(3):
            if(i==j):
                continue
            qml.CRot(*q_weights_flat[9 + ctr*3 : 12 + ctr *3], wires = [i+1 + 4,j+1 + 4])
            ctr += 1
    for i in range(0,3):        
        qml.Rot(*q_weights_flat[27 + 3*i :30 + 3*i], wires = i + 1 + 4)
    
    return  [qml.probs([1,2,3]), qml.probs([5,6,7])]



class DressedQuantumNet(nn.Module):
    """
    Torch module implementing the *dressed* quantum net.
    """

    def __init__(self):
        """
        Definition of the *dressed* layout.
        """
        super().__init__()
        self.pre_net = nn.Linear(256, 128)
        self.relu = torch.nn.LeakyReLU(0.1)
        self.pre_net2 = nn.Linear(128, 64) 
        self.relu2 = torch.nn.LeakyReLU(0.1)
        self.pre_net3 = nn.Linear(64, 36)
    def forward(self, input_features,ind):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """
        
        pre_out = self.pre_net(input_features)        
        
        pre_out = self.pre_net2(pre_out)
        
        pre_out = self.pre_net3(pre_out)
        print(pre_out)
        q_out_elem = quantum_net(pre_out, ind).float().unsqueeze(0)        

        
        return q_out_elem




model_hybrid = DressedQuantumNet()
opt = optim.Adam(model_hybrid.parameters() , lr = 0.1)

# %% 
def custom_loss(outs,targets):
    loss1 = torch.sum((outs[0] - targets[0] )**2)
    loss2 = torch.sum((outs[1] - targets[1] )**2)
    return loss1 + loss2
customLoss = custom_loss

# %% 
n_qubits = 4
ground_truths = np.zeros((n_embed_samples,2, 2 ** (n_qubits -1 )))
ground_truths.requires_grad = False
for i in range(n_embed_samples):
    ground_truths[i][0] = (latent_circuit(4,i +n_embed_samples ,input_states,model_params))
    ground_truths[i][1] = (latent_circuit(4,i +n_embed_samples*2 ,input_states,model_params))
ground_truths.requires_grad    

# %%
losses = []
for ep in range(10):
    batch_random = np.arange((n_embed_samples))
    np.random.shuffle(batch_random)
    running_loss  = 0
    start = timeit.time.time()
    for ind in range(0,1):
    # for ind in batch_random:
        opt.zero_grad()
        out = model_hybrid(torch.Tensor(coefs[ind]), ind )[0]
        l = customLoss(out,ground_truths[ind])
        # l = loss_func_ham(out[0][0]) #+ loss_func_ham(out[1][0]) # loss_func_ham(out[2][0])
        
        
        running_loss += l        
        l.backward()
        losses.append(l)
        print(out, ground_truths[ind])
        opt.step()
        if(l <= -1.8):
            break
    print("Epoch: [{}], Loss: {:.3f}".format(ep,running_loss/(2)))
    end = timeit.time.time()
    print('Time elapsed for the epoch [{}] : {:.2f}'.format(ep,end-start))

