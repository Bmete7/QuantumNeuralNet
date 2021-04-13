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
sys.path.append('../QAutoencoder')
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
    





# %%Numpy opt

params = np.random.random([18])
params = torch.from_numpy(params)
grad_func = qml.grad(circuit)
# input_states.requires_grad = False
opt = torch.optim.Adam(embed_model.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

loss = 0
batch_id = np.arange(n_inputs) #
np.random.shuffle(batch_id)

def cost(x):
    
    fidelity = circuit(x, input_states[i], True)
    
    if(i == batch_id[0] or i == batch_id[-1] or i == 6):
        print(fidelity, i)
    loss = torch.log(1- fidelity)
    return torch.log(1- fidelity)

def closure():
    opt.zero_grad()
    loss = cost(params)
    loss.backward()
    running_loss += loss
    return loss


for epoch in range(100):
    start_time = timeit.time.time()
    
    running_loss = 0
    for i in batch_id:
        
        opt.step(closure)
        
        
    end_time = timeit.time.time()
    print(' * ')
    print(running_loss / n_inputs)
    print('Time elapsed for the epoch' +  str(epoch)  + ' : {:.2f}'.format(end_time-start_time))
# %% 
import numpy as np
n_embed_samples = 20

pauli_list = [I,pauli_x, pauli_y, pauli_z]
observables = [np.kron(i,j) for i in pauli_list for j in pauli_list]
# Create random hamiltonians from pauli coefficients
hamiltonians = []
input_states = np.zeros((n_embed_samples*3,4), dtype ='complex128')
# time_steps = 3 # how many time evolution steps are going to be simulated
time_steps = 2 # how many time evolution steps are going to be simulated
for i in range(n_embed_samples):
    # coefs = np.random.rand(16) * np.random.rand() + np.random.rand()
    coefs = np.zeros(16 , dtype = 'float64')
    coefs[np.random.randint(16)] = np.random.randint(4)
    coefs[np.random.randint(16)] = np.random.randint(4)
    
    hamiltonian = np.einsum('i,ijk->jk', coefs,observables)
    hamiltonians.append(hamiltonian)
    eig,eiv = np.linalg.eig(hamiltonian)
    state = eiv[np.random.randint(4)]
    input_states[i] = state
    for t in range(1, time_steps+ 1):
        input_states[i+ (n_embed_samples * (t))] = time_evolution_simulate(hamiltonian, t, state )
# expm(hamiltonians[233] * -2j) @ input_states[233] == input_states[1233] To show that it holds
n_inputs =len(input_states)

# %% 4 qubit data
        
n_embed_samples = 2        
pauli_list = [I,pauli_x, pauli_y, pauli_z]
observables = [np.kron(np.kron(np.kron(i,j), k), m) for k in pauli_list for m in pauli_list for i in pauli_list for j in pauli_list]
# Create random hamiltonians from pauli coefficients
hamiltonians = []
input_states = np.zeros((n_embed_samples*4,16), dtype ='complex128')
time_steps = 3 # how many time evolution steps are going to be simulated
for i in range(n_embed_samples):
    coefs = np.random.rand(256) * np.random.rand() + np.random.rand()
    hamiltonian = np.einsum('i,ijk->jk', coefs,observables)
    hamiltonians.append(hamiltonian)
    eig,eiv = np.linalg.eig(hamiltonian)
    state = eiv[np.random.randint(16)]
    input_states[i] = state
    for t in range(1, time_steps+ 1):
        input_states[i+ (n_embed_samples * (t))] = time_evolution_simulate(hamiltonian, t, state )
# expm(hamiltonians[233] * -2j) @ input_states[233] == input_states[1233] To show that it holds
n_inputs =len(input_states)        
# %% 




# %% 
 
dev_embed = qml.device("default.qubit", wires=6,shots = 1000)
embed_model = EmbedNet(dev_embed,input_states, 1, 6, 4, 1)


epochs = 12
loss_list = []

loss_func = Fidelity_loss
learning_rate = 0.1
opt = torch.optim.Adam(embed_model.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)



# %%  Training for the Autoencoder
if(model_saved == False):
    # batch_id = np.arange(n_embed_samples)
    
    
    for epoch in range(30):
        running_loss = 0
        total_loss = []
        start_time = timeit.time.time()
        #for i in batch_id:
        batch_id = np.arange(n_inputs) #
        np.random.shuffle(batch_id)
        for i in batch_id:
            opt.zero_grad()
            out = embed_model(torch.zeros([2]), True)
            
            
            loss = loss_func(out)
            loss.backward()
            running_loss += loss

            opt.step()
            total_loss.append(loss.item())
        end_time = timeit.time.time()
        if (running_loss /n_inputs < 0.00015):
            break   
        print('Time elapsed for the epoch' +  str(epoch)  + ' : {:.2f}'.format(end_time-start_time))
        loss_list.append(sum(total_loss)/len(total_loss))
        print('Training [{:.0f}%]\tLoss: {:.4f}'.format(100. * (epoch + 1) / epochs,  running_loss /n_inputs  ))
    
# %% 
def hamiltonian_simulate(H, t):
    """ Returns the unitary driven by some hamiltonian
    and time t
        
    Non-Keyword arguments:
        H- Hermitian Matrix
        t- Time parameter
        
    Returns:
        Unitary Matrix
    """
    return sp.linalg.expm(H* -1j * t)
def time_evolution_simulate(H, t, psi):
    """ Returns the time evolution driven by some hamiltonian
    and time t on a specific state
        
    Non-Keyword arguments:
        H- Hermitian matrix
        i- Index of the hamiltonian in the array
        t- Time parameter
        psi- input state
        
    Returns:
        Another quantum state
    """
    return hamiltonian_simulate(H, t) @ psi
    
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
        AngleEmbeddingCircuit(embed_features[i])
        q_values.append(angle_measurement_dev._state)
    import scipy as sp
    x_embed_params = []    
    x2_embed_params = [] 
    
    q_values_t1 =  [] 
    q_values_t2 =  [] 
    q_values_t3 =  [] 
        
    for i in range(7500):
        cur_q_val = q_values[i].reshape(4,)
        
        U_t1 = time_evolution_simulate(i, 1)
        U_t2 = time_evolution_simulate(i, 2)
        U_t3 = time_evolution_simulate(i, 3)
        
        cur_q_val_t1 = U_t1 @ cur_q_val
        cur_q_val_t2 = U_t2 @ cur_q_val
        cur_q_val_t3 = U_t3 @ cur_q_val
        
        q_values_t1.append(cur_q_val_t1)
        q_values_t2.append(cur_q_val_t2)
        q_values_t3.append(cur_q_val_t3)
        
        
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
    batch_id = np.arange(1) #
    np.random.shuffle(batch_id)
    
    for epoch in range(epochs):
        running_loss = 0
        total_loss = []
        start_time = timeit.time.time()
        #for i in batch_id:
        for i in batch_id:
            opt.zero_grad()
    
            normalized = torch.Tensor(input_states[0])
            
            out = embed_model(normalized,True)
            
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
embed_features= embed_features.numpy()
if(model_saved == False)
    succesfull_train_sample_index = []
    
    # Checks if the encode/decode phase is well 
    for i in range(7500):
        if((embed_model(torch.Tensor(embed_features[i]))[0][0].item()) >= 0.975 and (embed_model(embed_features[i*3 +2500])[0][0].item()) >= 0.975 and (embed_model(embed_features[i*3+2500 + 1])[0][0].item()) >= 0.985  ):
            print(i)
            succesfull_train_sample_index.append(i)
            
            
    initial_succesfull_train_sample_index = deepcopy(succesfull_train_sample_index)
    
    # hamiltonians that are acting on the cherry-picked data
    acting_hamiltonians = np.array(hamiltonians)[initial_succesfull_train_sample_index]
    
    
    
    
    # index numbers for the cherry-picked qubits
    initial_succesfull_train_sample_index_t1 = [i*3 + 2500 for i in initial_succesfull_train_sample_index]
    initial_succesfull_train_sample_index_t2 = [i*3 + 2500 +1 for i in initial_succesfull_train_sample_index]
    initial_succesfull_train_sample_index_t3 = [i*3 + 2500 +2 for i in initial_succesfull_train_sample_index]
    
    



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



if(model_saved == False):
    hamiltonian_coefs = []
    for i in range(len(acting_hamiltonians)):
        hamiltonian_coefs.append(decompose_hamiltonian(acting_hamiltonians[i], paulis))
    hamiltonian_coefs = np.array(hamiltonian_coefs)
    

    succesfull_train_samples_all = deepcopy(embed_features[initial_succesfull_train_sample_index])
    
    selected_qubits_input = np.array(q_values)[initial_succesfull_train_sample_index]
    selected_qubits_t1 = np.array(q_values_t1)[initial_succesfull_train_sample_index]
    selected_qubits_t2 = np.array(q_values_t2)[initial_succesfull_train_sample_index]
    selected_qubits_t3 = np.array(q_values_t3)[initial_succesfull_train_sample_index]
    
    
    succesfull_train_samples_input = deepcopy(embed_features[initial_succesfull_train_sample_index])
    succesfull_train_samples_t1 = deepcopy(embed_features[initial_succesfull_train_sample_index_t1])
    succesfull_train_samples_t2 = deepcopy(embed_features[initial_succesfull_train_sample_index_t2])
    succesfull_train_samples_t3 = deepcopy(embed_features[initial_succesfull_train_sample_index_t3])
    # Save the cherry-picked data points
    



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


# succesfull_train_samples_input = succesfull_train_samples_input.numpy()
# succesfull_train_samples_t1 = succesfull_train_samples_t1.numpy()
# succesfull_train_samples_t2 = succesfull_train_samples_t2.numpy()
# succesfull_train_samples_t3 =succesfull_train_samples_t3.numpy()
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
    
    for i in range(len(succesfull_train_samples_input)):
        
        # Global phase shift is not important
        res = latent_circuit(succesfull_train_samples_input[i])
        succesfull_train_samples_input_latent.append(latents_dev._state[0][0])
        succesfull_train_samples_input_latent_exp.append(latent_circuit_exp(succesfull_train_samples_input[i]))
        succesfull_train_samples_input_latent_rot.append(np.arccos(np.sqrt(res)[0]) * 2)
        
        res = latent_circuit(succesfull_train_samples_t1[i])
        succesfull_train_samples_t1_latent.append(latents_dev._state[0][0])
        succesfull_train_samples_t1_latent_exp.append(latent_circuit_exp(succesfull_train_samples_t1[i]))
        succesfull_train_samples_t1_latent_rot.append(np.arccos(np.sqrt(res)[0]) * 2)
        
        res = latent_circuit(succesfull_train_samples_t2[i])
        succesfull_train_samples_t2_latent.append(latents_dev._state[0][0])
        succesfull_train_samples_t2_latent_exp.append(latent_circuit_exp(succesfull_train_samples_t2[i]))
        succesfull_train_samples_t2_latent_rot.append(np.arccos(np.sqrt(res)[0]) * 2)
        
        res = latent_circuit(succesfull_train_samples_t3[i])
        succesfull_train_samples_t3_latent.append(latents_dev._state[0][0])
        succesfull_train_samples_t3_latent_exp.append(latent_circuit_exp(succesfull_train_samples_t3[i]))
        succesfull_train_samples_t3_latent_rot.append(np.arccos(np.sqrt(res)[0]) * 2)
        
        
# %% Angle check if values are correct
        
angle_dev = qml.device("default.qubit", wires=1,shots = 1000)
@qml.qnode(angle_dev)
def angle_circuit(inputs):
    qml.templates.embeddings.AngleEmbedding(inputs,wires = range(0,1), rotation = 'X')
    
    return qml.probs([0])
latent_circuit(succesfull_train_samples_t2[235])
angle_circuit([succesfull_train_samples_t2_latent_rot[235]])

#Yields the same result
expm(acting_hamiltonians[235] * -1j) @ selected_qubits_input[235].reshape(-1,1)
selected_qubits_t1[235]
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
    
    # Coefficients of the hamiltonians, real numbers referring to the pauli product state coeffs.
    # a1 III + a2 IIX + . . . aN ZZZ
    
    np.save('hamiltonian_coefs_more_qubits.npy' ,hamiltonian_coefs)
    
    # Selected input features that can be encoded and decoded also with the given time evolution.
    np.save('succesfull_train_samples_input_more_qubits.npy' ,succesfull_train_samples_input)
    
    np.save('succesfull_train_samples_t1_more_qubits.npy' ,succesfull_train_samples_t1)
    np.save('succesfull_train_samples_t2_more_qubits.npy' ,succesfull_train_samples_t2)
    np.save('succesfull_train_samples_t3_more_qubits.npy' ,succesfull_train_samples_t3)
    
    #Hamiltonians, and the Hamiltonians only acting on the selected features
    np.save('hamiltonians.npy_more_qubits' ,hamiltonians)
    np.save('acting_hamiltonians_more_qubits.npy' ,acting_hamiltonians)
    
    #indices of the selected features of input and time evolved qubits
    np.save('initial_succesfull_train_sample_index_more_qubits.npy' ,initial_succesfull_train_sample_index)
    np.save('initial_succesfull_train_sample_index_t1_more_qubits.npy' ,initial_succesfull_train_sample_index_t1)
    np.save('initial_succesfull_train_sample_index_t2_more_qubits.npy' ,initial_succesfull_train_sample_index_t2)
    np.save('initial_succesfull_train_sample_index_t3_more_qubits.npy' ,initial_succesfull_train_sample_index_t3)
    
    # State vectors of the latent spaces of input and time evolved qubits
    np.save('succesfull_train_samples_input_latent_more_qubits.npy' ,succesfull_train_samples_input_latent)
    np.save('succesfull_train_samples_t1_latent_more_qubits.npy' ,succesfull_train_samples_t1_latent)
    np.save('succesfull_train_samples_t2_latent_more_qubits.npy' ,succesfull_train_samples_t2_latent)
    np.save('succesfull_train_samples_t3_latent_more_qubits.npy' ,succesfull_train_samples_t3_latent)
    
    # Exp. value of the measurement w.r.t PauliZ of the latent space rep. of inputs and the time evolved qubits
    np.save('succesfull_train_samples_input_latent_exp_more_qubits.npy' ,succesfull_train_samples_input_latent_exp)
    np.save('succesfull_train_samples_t1_latent_exp_more_qubits.npy' ,succesfull_train_samples_t1_latent_exp)
    np.save('succesfull_train_samples_t2_latent_exp_more_qubits.npy' ,succesfull_train_samples_t2_latent_exp)
    np.save('succesfull_train_samples_t3_latent_exp_more_qubits.npy' ,succesfull_train_samples_t3_latent_exp)
    
    
    # rotation angles for the embeddings of the latent space rep. of inputs and the time evolved qubits
    np.save('succesfull_train_samples_input_latent_rot_more_qubits.npy' ,succesfull_train_samples_input_latent_rot)
    np.save('succesfull_train_samples_t1_latent_rot_more_qubits.npy' ,succesfull_train_samples_t1_latent_rot)
    np.save('succesfull_train_samples_t2_latent_rot_more_qubits.npy' ,succesfull_train_samples_t2_latent_rot)
    np.save('succesfull_train_samples_t3_latent_rot_more_qubits.npy' ,succesfull_train_samples_t3_latent_rot)
    
    
    # State vectors of the inputs and the time evolved qubits after the embedding
    np.save('selected_qubits_input.npy', selected_qubits_input)
    np.save('selected_qubits_t1.npy', selected_qubits_t1)
    np.save('selected_qubits_t2.npy', selected_qubits_t2)
    np.save('selected_qubits_t3.npy', selected_qubits_t3)
 
    
    

# %% Hamiltonian Approximator

ham_dev = qml.device("default.qubit", wires=6)

@qml.qnode(ham_dev, interface="torch")
def quantum_net(q_input_features, q_weights_flat,t = 1, inputs = False):
    """
    The variational quantum circuit.
    """
    
    qml.templates.embeddings.AmplitudeEmbedding(inputs,wires = range(2,4), normalize = True,pad=(0.j))
    

    # Embedding of the latent space into qubits
    qml.RX(q_input_features, wires=1)
    # qml.RX(q_input_features, wires=3)

    # Apply the same unitary t+1 times for each wire t
    for i in range(t):
        qml.Rot(*q_weights_flat, wires = 1)
    
    # qml.Rot(*q_weights_flat, wires = 3)
    # qml.Rot(*q_weights_flat, wires = 3)
    
    
    qml.Rot(*final_rot_params_0 , wires = 0).inv()
    qml.Rot(*final_rot_params_1 , wires = 1).inv()
    qml.CRot(*crot_params_10, wires = [1,0]).inv()
    qml.CRot(*crot_params_01, wires = [0,1]).inv()
    qml.Rot(*first_rot_params_0 , wires = 0).inv()
    qml.Rot(*first_rot_params_1 , wires = 1).inv()

    # SWAP Test
    qml.Hadamard(wires = 4)
    qml.Hadamard(wires = 5)
    
    qml.CSWAP(wires = [4, 0 , 2])
    qml.CSWAP(wires = [5,1,3])
    
    qml.Hadamard(wires = 4)
    qml.Hadamard(wires = 5)
    
    # Expectation values in the Z basis
    # probs =  [qml.probs([0,1]) , qml.probs([2,3])]
    # return tuple(probs)
    # exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(2)]
    # exp_vals = [qml.expval(qml.PauliZ(4)) , qml.expval(qml.PauliZ(5))]
    # return tuple(exp_vals)
    return [qml.probs(i) for i in range(4,6)]


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
    def forward(self, input_features,input_2,t,prob_labels):
        """
        Defining how tensors are supposed to move through the *dressed* quantum
        net.
        """
        # obtain the input features for the quantum circuit
        # by reducing the feature dimension from 512 to 4
        
        pre_out = self.pre_net(input_features)        
        q_in = torch.tanh(pre_out) * np.pi / 2.0
        
        # Apply the quantum circuit to each element of the batch and append to q_out        
        q_out_elem = quantum_net(input_2 , q_in,t,prob_labels).float().unsqueeze(0)        

        # return the two-dimensional prediction from the postprocessing layer
        return q_out_elem




model_hybrid = DressedQuantumNet()
opt = optim.Adam(model_hybrid.parameters() , lr = 0.0006)


loss_function = prob_cost
loss_function = cost

opt.param_groups[0]['params'][0].size()

prob_labels = [ selected_qubits_t1[:] , selected_qubits_t2 ]


labels = [succesfull_train_samples_t1_latent_exp , succesfull_train_samples_t2_latent_exp]
labels = torch.Tensor(np.array(labels).T)

for ep in range(20):
    batch_random = np.arange(len(succesfull_train_samples_input_latent_rot))
    np.random.shuffle(batch_random)
    running_loss  = 0
    start = timeit.time.time()
    for ind in batch_random:
        opt.zero_grad()
        out1 = model_hybrid(torch.Tensor(hamiltonian_coefs[ind]), succesfull_train_samples_input_latent_rot[ind], 1 , prob_labels[0][ind] ) 
        out2 = model_hybrid(torch.Tensor(hamiltonian_coefs[ind]), succesfull_train_samples_input_latent_rot[ind], 2 , prob_labels[1][ind] )
        
        l = loss_func(out1[0])
        
        l += loss_func(out2[0])
        
        running_loss += l
        
        l.backward()
        opt.step()
    print("Epoch: [{}], Loss: {:.3f}".format(ep,running_loss/len(succesfull_train_samples_input_latent_rot)))
    end = timeit.time.time()
    print('Time elapsed for the epoch [{}] : {:.2f}'.format(ep,end-start))
    
    
for ind in batch_random:
    
    print( model_hybrid(torch.Tensor(hamiltonian_coefs[ind]), succesfull_train_samples_input_latent_rot[ind], 1 , prob_labels[0][ind] ) )
    