# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 16:17:28 2021

@author: burak
"""

# Different hamiltonians, and their ground states

import pennylane as qml
import torch
import numpy as np
import timeit
from DataCreation import dataPreparation
from QAutoencoder.QAutoencoder import QuantumAutoencoder



# %% Data preparation
norm = lambda x:np.linalg.norm(x)
normalize = lambda x: x/norm(x)

torchnorm = lambda x:torch.norm(x)
torchnormalize = lambda x: x/torchnorm(x)
    
# %% Loading the previous model

tensorlist = dataPreparation(saved = False, save_tensors = False)
tensorlist = torch.as_tensor(tensorlist)

    
# %% Import saved qubits

PATH = './autoencoder_more_qubits.npy'
PATH2 = './train_qubits.npy'
n_qubit_size = 8
latent_size = 1
n_auxillary_qubits = latent_size
dev = qml.device('default.qubit', wires = n_qubit_size + latent_size + n_auxillary_qubits)
n_data = len(tensorlist)
qAutoencoder = QuantumAutoencoder(n_qubit_size, dev, latent_size, n_auxillary_qubits, n_data)

# qAutoencoder.load_state_dict(torch.load(PATH))

# tensorlist = np.load(PATH2,allow_pickle = True)
# %% 


# for n_qubits in range(1,10):
#     vec_dev = qml.device('default.qubit', wires = n_qubits)
    
#     @qml.qnode(vec_dev)
#     def qCircuit(inputs):
#         qml.QubitStateVector(inputs, wires = range(0,n_qubits))
        
#         return qml.probs(range(0,n_qubits))
#         return [qml.expval(qml.PauliZ(q)) for q in range(0,1)]
    
#     print(qCircuit(tensorlist[n_qubits][0]))

# %% 

def fidelityLossMulti(mes):
    return sum(torch.log(1 - (mes[i])) for i in range(len(mes)))
loss_func = fidelityLossMulti
learning_rate = 0.003
opt = torch.optim.Adam(qAutoencoder.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
opt = torch.optim.RMSprop(qAutoencoder.parameters(), lr = learning_rate)

batch_size = 4
average_fidelities = []
avg_loss = 0
epochs = 1

# qAutoencoder(H)

#x = torch.stack([torch.Tensor(qml.init.strong_ent_layers_normal(n_qubit_size, qAutoencoder.n_total_qubits - (2* n_auxillary_qubits))) for i in range(n_data) ])



# %% Training
for epoch in range(epochs):
    running_loss = 0
    start_time = timeit.time.time()
    batch_id = np.arange(n_data)
    np.random.shuffle(batch_id)
    for i in batch_id:
        opt.zero_grad()
        # out = qAutoencoder(tensorlist[n_qubit_size][i])
        out = qAutoencoder(tensorlist[i])
        loss = loss_func(out)
        loss.backward()
        if(i % 100 == 0):
            print(out, i)
        running_loss += loss
        opt.step()
    epoch_loss = running_loss / n_data
    avg_loss = (avg_loss * epoch + epoch_loss) / (epoch + 1)
    print(epoch_loss)
    end_time = timeit.time.time()
    
    print('Time elapsed for the epoch' +  str(epoch)  + ' : {:.2f}'.format(end_time-start_time))
    
    
# %% Testing


n_qubit_size = 8
for i in range(10):
    # state = (torch.rand(2**n_qubit_size, dtype = torch.cfloat)) - 1.013j
    # state = (torch.zeros(2**n_qubit_size))
    # state[0] = 1
    # ctr = 0
    # while(torchnorm(state) != 1.0 and ctr <= 5):
    #     state = torchnormalize(state)
    #     ctr += 1
    
    state = tensorlist[i]
    
    out = (qAutoencoder(state, training_mode = False))
    out = (qAutoencoder(state, training_mode = True ))
    print(out)
    # input_state  = np.kron(np.kron([1,0], [1,0]),tensorlist[n_qubit_size][i].numpy())
    input_state  = np.kron(np.kron([1,0], [1,0]),state.numpy())
    
    result = np.array(dev.state.detach())
    
    # how similar is the output to the input
    similarity = sum(np.abs((result-input_state) ** 2)) / 256
    print('similarity is {:.9f}'.format(similarity))

# %% Calculation of durations
'''
n_qubit_sizes = list(range(3,12))
iteration_duration = []

for n in n_qubit_sizes:
    
    n_qubit_size = n
    latent_size = 2
    n_auxillary_qubits = latent_size
    
    
    dev = qml.device('default.qubit', wires = n_qubit_size + latent_size + n_auxillary_qubits)
    loss_func = fidelityLossMulti
    learning_rate = 0.02
    
    qAutoencoder = QuantumAutoencoder(n_qubit_size, dev, latent_size, n_auxillary_qubits)
    
    opt = torch.optim.Adam(qAutoencoder.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    average_fidelities = []
    batch_size = 4
    epochs = 30
    
    n_data = 10
    
    
    start_time = timeit.time.time()
    
    opt.zero_grad()
    out = qAutoencoder(tensorlist[n_qubit_size][0])
    
    loss = loss_func(out)
    print(out)
    loss.backward()
    opt.step()
    
    end_time = timeit.time.time()
    iteration_duration.append(end_time - start_time )
    print('Time elapsed for {} qubits:  {:.2f} '.format(n_qubit_size, end_time-start_time ))
'''
# %%  Saving the model


PATH
torch.save(qAutoencoder.state_dict(), PATH)
PATH2 = './inputs_states.npy'
torch.save(tensorlist, PATH2)
X = torch.load(PATH2)
# %%  NEXT THINGS TO DO

# PREPARE THE DEV.STATE WITH THE INPUT THAT IS PREPARED WITH QARBITRARYSTATE - DONE
# Prepare examplary hamiltonians, 
# 