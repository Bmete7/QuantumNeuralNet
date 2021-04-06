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

sys.path.append('C:/Users/burak/OneDrive/Desktop/Quantum Machine Learning/CodeBank/QuantumNeuralNet/dataLoad')
sys.path.append('C:/Users/burak/OneDrive/Desktop/Quantum Machine Learning/CodeBank/QuantumNeuralNet/pennyLane')
sys.path.append('C:/Users/burak/OneDrive/Desktop/Quantum Machine Learning/CodeBank/QuantumNeuralNet/representation')
sys.path.append('../PennyLane')
sys.path.append('../representation')
from IrisLoad import importIRIS, IRISDataset
from MNISTLoad import importMNIST
from qCircuit import EmbedNet

from visualizations import visualize, visualize_state_vec
from copy import deepcopy
import seaborn
import tensorflow as tf

#%% numpy implementations for the gates


I  = np.array(((0j + 1, 0), (0, 1)))
pauli_x  = np.array(((0j, 1), (1, 0)))
pauli_y = np.array(((0, -1j), (1j, 0)))
pauli_z = np.array(((1+0j, 0), (0, -1)))
paulis = [I,pauli_x,pauli_y, pauli_z]
# %% 

model_saved =False


# %%
def Fidelity_loss(mes):
    tot  =0
    for i in mes:
        
        tot += i[0]
    fidelity = (2 * (tot) / len(mes[0])  - 1.00)
    return torch.log(1- fidelity)
    


# %% 
 
dev_embed = qml.device("default.qubit", wires=4+2+2,shots = 1000)
embed_model = EmbedNet(dev_embed, 1, 4, 2, 1)

learning_rate = 0.01
epochs = 40
loss_list = []

loss_func = Fidelity_loss
opt = torch.optim.Adam(embed_model.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# %% 
n_embed_samples = 30000
embed_features = np.random.rand(n_embed_samples,2)* np.pi
if(model_saved):
    embed_features =np.load('embed_features.npy')
else: 
    np.save('embed_features_more_qubits.npy' ,embed_features)



# %% 
amps_d = qml.device("default.qubit", wires=2,shots = 1000)
@qml.qnode(amps_d)
def circ_amp(inputs):
    qml.templates.embeddings.AngleEmbedding(inputs,wires = range(0,2), rotation = 'X')
    return qml.probs([0,1])

def time_evolution_simulate(i, t):
    return sp.linalg.expm(hamiltonians[i] * -1j * t)
def find_rx_params(cur_q_val):
    amp_res = (cur_q_val[0] + cur_q_val[1])
    amp_res /= cur_q_val[3] + cur_q_val[2]
    
    amp_res2 = (cur_q_val[0] + cur_q_val[2])
    amp_res2 /= cur_q_val[1] + cur_q_val[3]
    
    amp_res = np.abs(amp_res)    
    amp_res2 = np.abs(amp_res2)
    a2 = 1 / (1 + (amp_res) ** 2)
    b2 = 1 / (1 + (amp_res2) ** 2)
    a1 = 1- a2
    b1 = 1-b2
    return np.arccos(np.sqrt(a1)) * 2 , np.arccos(np.sqrt(b1)) * 2
if (model_saved == False):
    pauli_list = [I,pauli_x, pauli_y, pauli_z]
    observables = [np.kron(i,j) for i in pauli_list for j in pauli_list]
    
    hamiltonians = []
    for i in range(7500):
        coefs = 2.5 * np.random.randn(16) + 0.5
        ham = coefs[0] * observables[0]
        for i in range(1,16):
            ham += coefs[i] * observables[i] 
        hamiltonians.append(ham)
    
    q_values =  [] 
    for i in range(7500):
        circ_amp(embed_features[i])
        q_values.append(amps_d._state)
    import scipy as sp
    x_embed_params = []    
    x2_embed_params = [] 
    
    
        
    for i in range(7500):
        cur_q_val = q_values[i].reshape(4,)
        
        U_t1 = time_evolution_simulate(i, 1)
        U_t2 = time_evolution_simulate(i, 2)
        U_t3 = time_evolution_simulate(i, 3)
        
        cur_q_val_t1 = U_t1 @ cur_q_val
        cur_q_val_t2 = U_t2 @ cur_q_val
        cur_q_val_t3 = U_t3 @ cur_q_val
        
        rx_1_t1, rx_2_t1 =  find_rx_params(cur_q_val_t1)
        rx_1_t2, rx_2_t2 = find_rx_params(cur_q_val_t2)
        rx_1_t3, rx_2_t3 = find_rx_params(cur_q_val_t3)
        
        x_embed_params.append(rx_1_t1)
        x2_embed_params.append(rx_2_t1)
        x_embed_params.append(rx_1_t2)
        x2_embed_params.append(rx_2_t2)
        x_embed_params.append(rx_1_t3)
        x2_embed_params.append(rx_2_t3)
    
    
    
    
    x_params = torch.from_numpy(np.array(x_embed_params))
    x2_params = torch.from_numpy(np.array(x2_embed_params))
    
    embed_features_hams = torch.stack((x_params,x2_params)).T
    
    embed_features[7500:] = embed_features_hams
    embed_features = torch.Tensor(embed_features)


# %%  Training for the Autoencoder
if(model_saved == False):
    # batch_id = np.arange(n_embed_samples)
    batch_id = np.arange(7500) #
    np.random.shuffle(batch_id)
    
    for epoch in range(epochs):
        running_loss = 0
        total_loss = []
        start_time = timeit.time.time()
        #for i in batch_id:
        for i in batch_id:
            opt.zero_grad()
    
            normalized = embed_features[i]
            
            out = embed_model(normalized,True)
            out1 = embed_model(embed_features[i+7500],True)#
            out2 = embed_model(embed_features[i+7501],True) #
            out3 = embed_model(embed_features[i+7502],True)
            
            loss = loss_func(out)
            loss += loss_func(out1) # 
            loss += loss_func(out2) #
            loss += loss_func(out3) #
            loss.backward()
            running_loss += loss/4
            if(i%100 == 0):
                print(out)
            opt.step()
            total_loss.append(loss.item())
        end_time = timeit.time.time()
        print('Time elapsed for the epoch' +  str(epoch)  + ' : {:.2f}'.format(end_time-start_time))
        loss_list.append(sum(total_loss)/len(total_loss))
        print('Training [{:.0f}%]\tLoss: {:.4f}'.format(100. * (epoch + 1) / epochs,  running_loss / 7500 ))    
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
plt.hist(losses, bins=np.arange(0,0.5, 0.00525))



# %% SAVE AND LOAD THE MODULE 
PATH = './autoencoder.npy'

if(model_saved == True):
    embed_model.load_state_dict(torch.load(PATH))
    print('model loaded')
else:
    PATH = './autoencoder_more_qubits.npy'
    torch.save(embed_model.state_dict(), PATH)
print('model parameters: ',  embed_model.state_dict)


# %% 
amp_dev = qml.device("default.qubit", wires=4,shots = 1000)
@qml.qnode(amp_dev)
def get_amps(count, inputs = False):
    qml.templates.AngleEmbedding(inputs, wires = range(0,count), rotation = 'X' )
    return qml.probs(range(0,count))

# %% 
embed_features= embed_features.numpy()
if(model_saved == False)
    succesfull_train_sample_index = []
    
    # Checks if the encode/decode phase is well 
    for i in range(2500):
        if((embed_model(torch.Tensor(embed_features[i]))[0][0].item()) >= 0.975 and (embed_model(embed_features[i*3 +2500])[0][0].item()) >= 0.975 and (embed_model(embed_features[i*3+2500 + 1])[0][0].item()) >= 0.985  ):
            print(i)
            succesfull_train_sample_index.append(i)
            
            
    initial_succesfull_train_sample_index = deepcopy(succesfull_train_sample_index)
    
    # hamiltonians that are acting on the cherry-picked data
    acting_hamiltonians = np.array(hamiltonians)[initial_succesfull_train_sample_index]
    
    np.save('hamiltonians.npy_more_qubits' ,hamiltonians)
    np.save('acting_hamiltonians_more_qubits.npy' ,acting_hamiltonians)
    
    
    # index numbers for the cherry-picked qubits
    initial_succesfull_train_sample_index_t1 = [i*3 + 2500 for i in initial_succesfull_train_sample_index]
    initial_succesfull_train_sample_index_t2 = [i*3 + 2500 +1 for i in initial_succesfull_train_sample_index]
    initial_succesfull_train_sample_index_t3 = [i*3 + 2500 +2 for i in initial_succesfull_train_sample_index]
    
    # # Save succesful data indices
    np.save('initial_succesfull_train_sample_index_more_qubits.npy' ,initial_succesfull_train_sample_index)
    np.save('initial_succesfull_train_sample_index_t1_more_qubits.npy' ,initial_succesfull_train_sample_index_t1)
    np.save('initial_succesfull_train_sample_index_t2_more_qubits.npy' ,initial_succesfull_train_sample_index_t2)
    np.save('initial_succesfull_train_sample_index_t3_more_qubits.npy' ,initial_succesfull_train_sample_index_t3)



# %% 

if(model_saved == True):
    hamiltonians = np.load('hamiltonians.npy')
    acting_hamiltonians = np.load('acting_hamiltonians.npy') # hamiltonians refer to initial_succesfull_train_sample_index
    
    initial_succesfull_train_sample_index= np.load('initial_succesfull_train_sample_index.npy') # selected indices
    initial_succesfull_train_sample_index_t1 = np.load('initial_succesfull_train_sample_index_t1.npy') # time evolution of the selected indices with t=1
    initial_succesfull_train_sample_index_t2= np.load('initial_succesfull_train_sample_index_t2.npy') 
    initial_succesfull_train_sample_index_t3  =np.load('initial_succesfull_train_sample_index_t3.npy')
    
    succesfull_train_samples_input= np.load('succesfull_train_samples_input.npy') # the features refer to initial_succesfull_train_sample_index
    succesfull_train_samples_t1= np.load('succesfull_train_samples_t1.npy')
    succesfull_train_samples_t2= np.load('succesfull_train_samples_t2.npy')
    succesfull_train_samples_t3= np.load('succesfull_train_samples_t3.npy')
    hamiltonian_coefs = np.load('hamiltonian_coefs.npy')
# np.save('initial_succesfull_train_sample_index.npy' , initial_succesfull_train_sample_index)
# %% Loading all the necessary data
from HermitianDecomposition import *


if(model_saved == False):
    hamiltonian_coefs = []
    for i in range(len(acting_hamiltonians)):
        hamiltonian_coefs.append(decompose_hamiltonian(acting_hamiltonians[i], paulis))
    hamiltonian_coefs = np.array(hamiltonian_coefs)
    np.save('hamiltonian_coefs_more_qubits.npy' ,hamiltonian_coefs)

    succesfull_train_samples_all = deepcopy(embed_features[initial_succesfull_train_sample_index])

    succesfull_train_samples_input = deepcopy(embed_features[initial_succesfull_train_sample_index])
    succesfull_train_samples_t1 = deepcopy(embed_features[initial_succesfull_train_sample_index_t1])
    succesfull_train_samples_t2 = deepcopy(embed_features[initial_succesfull_train_sample_index_t2])
    succesfull_train_samples_t3 = deepcopy(embed_features[initial_succesfull_train_sample_index_t3])
    # Save the cherry-picked data points
    
    np.save('succesfull_train_samples_input_more_qubits.npy' ,succesfull_train_samples_input)
    np.save('succesfull_train_samples_t1_more_qubits.npy' ,succesfull_train_samples_t1)
    np.save('succesfull_train_samples_t2_more_qubits.npy' ,succesfull_train_samples_t2)
    np.save('succesfull_train_samples_t3_more_qubits.npy' ,succesfull_train_samples_t3)




# %% Parameters of the embed_model


    
first_rot_params_0 = opt.param_groups[0]['params'][0][0][0]
first_rot_params_1 = opt.param_groups[0]['params'][0][0][1]

crot_params_01 = opt.param_groups[0]['params'][1][0][0]
crot_params_10 = opt.param_groups[0]['params'][1][1][0]

final_rot_params_0 = opt.param_groups[0]['params'][0][1][0]
final_rot_params_1 = opt.param_groups[0]['params'][0][1][1]

# 

first_rot_params_0 = first_rot_params_0.detach().numpy()
first_rot_params_1 = first_rot_params_1.detach().numpy()

final_rot_params_0 = final_rot_params_0.detach().numpy()
final_rot_params_1 = final_rot_params_1.detach().numpy()

crot_params_10 = crot_params_10.detach().numpy()
crot_params_01 = crot_params_01.detach().numpy()


# %% Recreating the Encoder/Decoder, to get the latent spaces

# succesfull_train_samples_input
# succesfull_train_samples_t1
# succesfull_train_samples_t2
# succesfull_train_samples_t3

# These will be used with AmplitudeEmbedding later on
if(model_saved == False):
    succesfull_train_samples_input_latent = []
    succesfull_train_samples_t1_latent = []
    succesfull_train_samples_t2_latent = []
    succesfull_train_samples_t3_latent = []
    
    succesfull_train_samples_input_latent_exp = []
    succesfull_train_samples_t1_latent_exp = []
    succesfull_train_samples_t2_latent_exp = []
    succesfull_train_samples_t3_latent_exp = []
    
    succesfull_train_samples_input_latent_rot = []
    succesfull_train_samples_t1_latent_rot = []
    succesfull_train_samples_t2_latent_rot = [] 
    succesfull_train_samples_t3_latent_rot = []



latents_dev = qml.device("default.qubit", wires=3,shots = 1000)
@qml.qnode(latents_dev)
def latent_circuit(inputs):
    qml.templates.embeddings.AngleEmbedding(inputs,wires = range(1,3), rotation = 'X')
    qml.Rot(*first_rot_params_0, wires = 1)
    qml.Rot(*first_rot_params_1, wires = 2)
    qml.CRot(*crot_params_01 , wires = [1,2])
    qml.CRot(*crot_params_10 , wires = [2,1])
    qml.Rot(*final_rot_params_0, wires = 1)
    qml.Rot(*final_rot_params_1, wires = 2)
    qml.SWAP(wires=[0,1])
    
    return qml.probs([2])

latents_exp_dev = qml.device("default.qubit", wires=3,shots = 1000)
@qml.qnode(latents_exp_dev)
def latent_circuit_exp(inputs):
    qml.templates.embeddings.AngleEmbedding(inputs,wires = range(1,3), rotation = 'X')
    qml.Rot(*first_rot_params_0, wires = 1)
    qml.Rot(*first_rot_params_1, wires = 2)
    qml.CRot(*crot_params_01 , wires = [1,2])
    qml.CRot(*crot_params_10 , wires = [2,1])
    qml.Rot(*final_rot_params_0, wires = 1)
    qml.Rot(*final_rot_params_1, wires = 2)
    qml.SWAP(wires=[0,1])
    
    return qml.expval(qml.PauliX(2))
if(model_saved == False):
    for i in range(len(succesfull_train_samples_input)):
        
        # Global phase shift is not important
        res = latent_circuit(succesfull_train_samples_input[i])
        succesfull_train_samples_input_latent.append(latents_dev._state[0][0])
        succesfull_train_samples_input_latent_exp.append(latent_circuit_exp(succesfull_train_samples_input[i]))
        succesfull_train_samples_input_latent_rot.append(np.arccos(res[0] **2))
        
        res = latent_circuit(succesfull_train_samples_t1[i])
        succesfull_train_samples_t1_latent.append(latents_dev._state[0][0])
        succesfull_train_samples_t1_latent_exp.append(latent_circuit_exp(succesfull_train_samples_t1[i]))
        succesfull_train_samples_t1_latent_rot.append(np.arccos(res[0] **2))
        
        res = latent_circuit(succesfull_train_samples_t2[i])
        succesfull_train_samples_t2_latent.append(latents_dev._state[0][0])
        succesfull_train_samples_t2_latent_exp.append(latent_circuit_exp(succesfull_train_samples_t2[i]))
        succesfull_train_samples_t2_latent_rot.append(np.arccos(res[0] **2))
        
        res = latent_circuit(succesfull_train_samples_t3[i])
        succesfull_train_samples_t3_latent.append(latents_dev._state[0][0])
        succesfull_train_samples_t3_latent_exp.append(latent_circuit_exp(succesfull_train_samples_t3[i]))
        succesfull_train_samples_t3_latent_rot.append(np.arccos(res[0] **2))
# %%  saving the input/outputs both exp values and state vectors
if(model_saved == False):
    succesfull_train_samples_input_latent  = np.array(succesfull_train_samples_input_latent)
    succesfull_train_samples_t1_latent  = np.array(succesfull_train_samples_t1_latent)
    succesfull_train_samples_t2_latent  = np.array(succesfull_train_samples_t2_latent)
    succesfull_train_samples_t3_latent  = np.array(succesfull_train_samples_t3_latent)
    
    succesfull_train_samples_input_latent_exp  = np.array(succesfull_train_samples_input_latent_exp)
    succesfull_train_samples_t1_latent_exp  = np.array(succesfull_train_samples_t1_latent_exp)
    succesfull_train_samples_t2_latent_exp  = np.array(succesfull_train_samples_t2_latent_exp)
    succesfull_train_samples_t3_latent_exp  = np.array(succesfull_train_samples_t3_latent_exp)
    
    succesfull_train_samples_input_latent_rot = np.array(succesfull_train_samples_input_latent_rot)
    succesfull_train_samples_t1_latent_rot = np.array(succesfull_train_samples_t1_latent_rot)
    succesfull_train_samples_t2_latent_rot = np.array(succesfull_train_samples_t2_latent_rot)
    succesfull_train_samples_t3_latent_rot = np.array(succesfull_train_samples_t3_latent_rot)
    
    np.save('succesfull_train_samples_input_latent_more_qubits.npy' ,succesfull_train_samples_input_latent)
    np.save('succesfull_train_samples_t1_latent_more_qubits.npy' ,succesfull_train_samples_t1_latent)
    np.save('succesfull_train_samples_t2_latent_more_qubits.npy' ,succesfull_train_samples_t2_latent)
    np.save('succesfull_train_samples_t3_latent_more_qubits.npy' ,succesfull_train_samples_t3_latent)
    
    np.save('succesfull_train_samples_input_latent_exp_more_qubits.npy' ,succesfull_train_samples_input_latent_exp)
    np.save('succesfull_train_samples_t1_latent_exp_more_qubits.npy' ,succesfull_train_samples_t1_latent_exp)
    np.save('succesfull_train_samples_t2_latent_exp_more_qubits.npy' ,succesfull_train_samples_t2_latent_exp)
    np.save('succesfull_train_samples_t3_latent_exp_more_qubits.npy' ,succesfull_train_samples_t3_latent_exp)
    
    np.save('succesfull_train_samples_input_latent_rot_more_qubits.npy' ,succesfull_train_samples_input_latent_rot)
    np.save('succesfull_train_samples_t1_latent_rot_more_qubits.npy' ,succesfull_train_samples_t1_latent_rot)
    np.save('succesfull_train_samples_t2_latent_rot_more_qubits.npy' ,succesfull_train_samples_t2_latent_rot)
    np.save('succesfull_train_samples_t3_latent_rot_more_qubits.npy' ,succesfull_train_samples_t3_latent_rot)

# %% Hamiltonian Approximator

ham_dev = qml.device("default.qubit", wires=4)
@qml.qnode(ham_dev, interface="torch")
def quantum_net(q_input_features, q_weights_flat):
    """
    The variational quantum circuit.
    """

    # Reshape weights
    # Embedding of the latent space into qubits
    qml.RX(q_input_features, wires=0)
    qml.RX(q_input_features, wires=2)

    # Apply the same unitary t+1 times for each wire t
    qml.Rot(*q_weights_flat, wires = 0)
    
    qml.Rot(*q_weights_flat, wires = 2)
    qml.Rot(*q_weights_flat, wires = 2)
    
    qml.Rot(*final_rot_params_0 , wires = 1).inv()
    qml.Rot(*final_rot_params_1 , wires = 0).inv()
    qml.CRot(*crot_params_10, wires = [0,1]).inv()
    qml.CRot(*crot_params_01, wires = [1,0]).inv()
    qml.Rot(*first_rot_params_0 , wires = 1).inv()
    qml.Rot(*first_rot_params_1 , wires = 0).inv()

    qml.Rot(*final_rot_params_0 , wires = 3).inv()
    qml.Rot(*final_rot_params_1 , wires = 2).inv()
    qml.CRot(*crot_params_10, wires = [2,3]).inv()
    qml.CRot(*crot_params_01, wires = [3,2]).inv()
    qml.Rot(*first_rot_params_0 , wires = 3).inv()
    qml.Rot(*first_rot_params_1 , wires = 2).inv()

    # Expectation values in the Z basis
    probs =  [qml.probs([0,1]) , qml.probs([2,3])]
    return tuple(probs)
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(2)]
    return tuple(exp_vals)
    
# phi = torch.tensor([0.011, 0.012,0.011, 0.012,0.011, 0.012,0.011, 0.012,0.011, 0.012,0.011, 0.012,0.011, 0.012,0.011, 0.012], requires_grad=True)

def cost(labels,probs):
    exps = probs[0]
    loss = (labels[0] - exps[0]) ** 2
    loss += (labels[1] - exps[1]) ** 2    
    return loss

def prob_cost(labels,probs):
  
    exps = probs[0]
    loss = (torch.sum(labels[0] - exps[0]) ** 2)
    loss += torch.sum((labels[1] - exps[1]) ** 2    )
    
    return loss

class DressedQuantumNet(nn.Module):
    """
    Torch module implementing the *dressed* quantum net.
    """

    def __init__(self):
        """
        Definition of the *dressed* layout.
        """
        super().__init__()
        self.pre_net = nn.Linear(16, 3)                
    def forward(self, input_features,input_2):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """
        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        
        pre_out = self.pre_net(input_features)        
        q_in = torch.tanh(pre_out) * np.pi / 2.0
        
        # Apply the quantum circuit to each element of the batch and append to q_out        
        q_out_elem = quantum_net(input_2 , q_in).float().unsqueeze(0)        

        # return the two-dimensional prediction from the postprocessing layer
        return q_out_elem


labels = [succesfull_train_samples_t1_latent_exp , succesfull_train_samples_t2_latent_exp]
labels = torch.Tensor(np.array(labels).T)

model_hybrid = DressedQuantumNet()
opt = optim.Adam(model_hybrid.parameters() , lr = 0.0008)
loss_function = cost
loss_function_prob = prob_cost

train_samples_t1_probs = []
train_samples_t2_probs = []
for i in range(succesfull_train_samples_t1.shape[0]):
    train_samples_t1_probs.append(circ_amp(succesfull_train_samples_t1[i]))
    train_samples_t2_probs.append(circ_amp(succesfull_train_samples_t2[i]))

train_samples_t1_probs = np.array(train_samples_t1_probs)
train_samples_t2_probs = np.array(train_samples_t2_probs)
prob_labels = [ train_samples_t1_probs , train_samples_t2_probs]
prob_labels = torch.Tensor(np.array(prob_labels)).view(-1,2,4)


for ep in range(20):
    batch_random = np.arange(len(succesfull_train_samples_input_latent_rot))
    np.random.shuffle(batch_random)
    running_loss  = 0
    for ind in batch_random:
        opt.zero_grad()

        l = loss_function_prob(prob_labels[ind], model_hybrid(torch.Tensor(hamiltonian_coefs[ind]), succesfull_train_samples_input_latent_rot[ind]) )
        running_loss += l
        l.backward()
        opt.step()
    print("Epoch: [{}], Loss: {:.3f}".format(ep,running_loss/len(succesfull_train_samples_input_latent_rot)))
    
    
    
for ind in range(138,146):
    print(labels[ind])
    print(model_hybrid(torch.Tensor(hamiltonian_coefs[ind]), succesfull_train_samples_input_latent_rot[ind]))
    print(loss_function_prob(prob_labels[ind], model_hybrid(torch.Tensor(hamiltonian_coefs[ind]), succesfull_train_samples_input_latent_rot[ind]) ))
    print(' - ')