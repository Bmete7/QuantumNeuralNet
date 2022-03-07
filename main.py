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
import utils


# %% Loading the previous model


LOAD_DATA = False
LOAD_MODEL = False
    
# %% Import saved qubits
# change those without the 1's 
PATH = './autoencoder_more_qubits1.npy'
PATH2 = './inputs_states1.npy'

n_data = 20
n_qubit_size = 8
if(LOAD_DATA == True):
    tensorlist = torch.load(PATH2)
else:
    tensorlist = dataPreparation(saved = False, save_tensors = False, method = 'state_prepare', number_of_samples = n_data, number_of_qubits= n_qubit_size)
    # tensorlist = torch.as_tensor(tensorlist)

latent_size = 1
n_auxillary_qubits = latent_size

dev = qml.device('default.qubit', wires = n_qubit_size + latent_size + n_auxillary_qubits)

qAutoencoder = QuantumAutoencoder(n_qubit_size, dev, latent_size, n_auxillary_qubits, n_data)

if(LOAD_MODEL == True):
    qAutoencoder.load_state_dict(torch.load(PATH))
# %% For utility functions / sampling, getting the state vec etc.


util_dev = qml.device('default.qubit', wires = n_qubit_size)
utils_object = utils.Utils(util_dev, n_qubit_size)
utils_object.n_qubit_size
utils_object.getState(tensorlist[1])

# %% Prepare an arbitrary Hamiltonian
from HamiltonianUtils import *

coeffs = [ np.random.rand() * 2 - 1 for i in range(n_qubit_size)]
pauliSet = [qml.PauliX(0).matrix, qml.PauliZ(0).matrix, qml.PauliY(0).matrix]
randomPauliGroup = [np.random.randint(3) for i in range(n_qubit_size)]


H = findHermitian(coeffs, randomPauliGroup, n_qubit_size)
torchH = torch.zeros_like(torch.Tensor(H), dtype=torch.cdouble)

torchH[:,:] = torch.Tensor(H.real[:,:])
torchH[:,:] += torch.Tensor(H.imag[:,:]) * 1j

Hamiltonian = qml.Hermitian(H, np.arange(n_qubit_size))

hamiltonian_dev = qml.device('default.qubit', wires = n_qubit_size, shots= 1)
@qml.qnode(hamiltonian_dev)
def measureEnergy(H, psi):
    qml.QubitStateVector(psi, wires = range(0,n_qubit_size))
    return qml.sample(Hamiltonian)

NUMBER_OF_EVOLUTIONS = 3 

evolved_states = []
dataset_size = len(tensorlist)
for i in range(dataset_size):
    psi = tensorlist[i]
    for t in range(1,NUMBER_OF_EVOLUTIONS + 1):
        start_time = timeit.time.time()
        evolved_states.append(timeEvolution(torchH, psi, t))
        end_time = timeit.time.time()
        print('Time elapsed for preparing the time evolved state : {:.2f}, for {}th state'.format(end_time - start_time, i * NUMBER_OF_EVOLUTIONS + 1 + t))

input_states = []
ctr = 0
for i in range(dataset_size):
    input_states.append(tensorlist[i])
    for t in range(NUMBER_OF_EVOLUTIONS):        
        input_states.append(evolved_states[ctr])
        ctr += 1
        
# %% Checking the dataset
print('Time evolutions checks with the dataset' , qml.math.allclose(timeEvolution(torchH, input_states[NUMBER_OF_EVOLUTIONS + 1] , 1 ), input_states[NUMBER_OF_EVOLUTIONS + 2]))
print('Time evolutions checks with the dataset' , qml.math.allclose(timeEvolution(torchH, input_states[NUMBER_OF_EVOLUTIONS*2 + 3] , 1 ), input_states[NUMBER_OF_EVOLUTIONS*2 + 4]))

# %% 

def fidelityLossMulti(mes):
    return sum(torch.log(1 - (mes[i])) for i in range(len(mes)))

loss_func = fidelityLossMulti
learning_rate = 0.01
opt = torch.optim.Adam(qAutoencoder.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
opt = torch.optim.RMSprop(qAutoencoder.parameters(), lr = learning_rate)

batch_size = 4
average_losses = []
avg_loss = 0
epochs = 50


#x = torch.stack([torch.Tensor(qml.init.strong_ent_layers_normal(n_qubit_size, qAutoencoder.n_total_qubits - (2* n_auxillary_qubits))) for i in range(n_data) ])
loss_checkpoints = []


# %% Training

for epoch in range(epochs):
    # N = len(input_states)
    N = len(tensorlist)
    running_loss = 0
    start_time = timeit.time.time()
    
    batch_id = np.arange(N)
    np.random.shuffle(batch_id)
    cur_losses_for_instances = []
    opt.zero_grad()
    for idx, i in enumerate(batch_id):
        # out = qAutoencoder(input_states[i] , training_mode = True)
        out = qAutoencoder(tensorlist[i] , training_mode = True)
        
        loss = loss_func(out)
        loss.backward()
        if(idx % 1 == 0):
            cur_losses_for_instances.append(loss)
            print(out)
        running_loss += loss
        if(idx % 5 == 0):
            opt.step()
            opt.zero_grad()
    epoch_loss = running_loss / N
    avg_loss = (avg_loss * epoch + epoch_loss) / (epoch + 1)
    average_losses.append(epoch_loss)
    
    end_time = timeit.time.time()
    print('Time elapsed for the epoch' +  str(epoch)  + ' : {:.2f}, with {} loss'.format(end_time - start_time, epoch_loss))


# %% Testing, w/ fidelity

for i in range(1):
    batch_id = np.arange(n_data)
    np.random.shuffle(batch_id)
    
    # state = input_states[i]
    state = tensorlist[i]
    out = qAutoencoder(state, training_mode = False)
    print(out)
    # print('Fidelity is {:.9f}'.format(out.detach().numpy().squeeze()))

els = np.zeros(int(len(dev.state)/4) , dtype = np.complex128)
coo = [0]
for j in range(n_qubit_size, total_qubit_size):
    coo.append(2**j)

for e in range(4):
    els[e] = 0
    for i in range(2):        
        for j in range(2):
            summ = coo[i * 1] + coo[j * 2]
            els[e] += (dev.state[summ + e] ** 2)
            # for k in range(2):
            #     for l in range(2):
            #         summ = coo[i * 1] + coo[j * 2] + coo[k  * 3] + coo[l * 4]
            #         els[e] += (dev2.state[summ + e] ** 2)
els = np.sqrt(els)
tensorlist[0]
    
    # dev2 = qml.device('default.qubit', wires = 4)
    # qq = QuantumAutoencoder(2, dev2, 1, 1, n_data)
    # qq(dd[0], training_mode = False)
    
    
    
# %% Training with SPSA

from noisyopt import minimizeSPSA

from pennylane import numpy as np
n_qubit_size = 8
N_QUBIT_SIZE = n_qubit_size
n_data = 2
dev_sampler_spsa = qml.device("qiskit.aer", wires = n_qubit_size + latent_size + n_auxillary_qubits, shots=1000)
tensorlist = dataPreparation(saved = False, save_tensors = False, method = 'state_prepare', number_of_samples = n_data, number_of_qubits= n_qubit_size)

total_qubit_size = n_qubit_size + latent_size + n_auxillary_qubits

data =[ tensorlist[i].numpy() for i in range(n_data)]
data_counter = 0

@qml.qnode(dev_sampler_spsa)
def qCircuit(params, inputs = False):
    qml.QubitStateVector(inputs, wires = range(n_auxillary_qubits + latent_size, total_qubit_size))
    for l in range(num_layers):                        
        for idx, i in enumerate(range(n_auxillary_qubits + latent_size, total_qubit_size)):
            qml.Rot(*params[l, idx, 0] , wires = i)
        for idx, i in enumerate(range(n_auxillary_qubits + latent_size, total_qubit_size)):
            
            for jdx, j in enumerate(range(n_auxillary_qubits + latent_size, total_qubit_size)):
                ctr=1
                if(i==j):
                    pass
                else:
                    qml.CRot( *params[l, idx, ctr], wires= [i, j])
                    ctr += 1
        for idx, i in enumerate(range(n_auxillary_qubits + latent_size, total_qubit_size)):
            qml.Rot(*params[l, idx, N_QUBIT_SIZE] , wires = i)
        
        
        for i in range(n_auxillary_qubits):
            qml.Hadamard(wires = i)
        for i in range(n_auxillary_qubits):
            qml.CSWAP(wires = [i, i + n_auxillary_qubits , n_auxillary_qubits + i + latent_size])
        for i in range(n_auxillary_qubits):
            qml.Hadamard(wires = i)
 
        return [qml.expval(qml.PauliZ(q)) for q in range(0, n_auxillary_qubits)]

num_layers = 3
flat_shape = num_layers * (n_qubit_size) * (n_qubit_size + 1) * 3
param_shape = (num_layers, n_qubit_size, n_qubit_size + 1, 3)
init_params = np.random.normal(scale=0.1, size=param_shape, requires_grad=True)
init_params_spsa = init_params.reshape(flat_shape)

def fidelityLossMultiple(init_params):
    global data
    global data_counter    
    mes = qCircuit(init_params, data[data_counter])
    return sum(np.log(1 - (mes[i])) for i in range(len(mes)))


niter_spsa = 100

# Evaluate the initial cost
cost_list = [fidelityLossMultiple(init_params)]
device_execs_spsa = [0]


start_time = timeit.time.time()
def callback_fn(xk):
    global data
    global data_counter
    
    
    cost_val = fidelityLossMultiple(xk)
    cost_list.append(cost_val)

    # We've evaluated the cost function, let's make up for that
    num_executions = int(dev_sampler_spsa.num_executions / 2)
    device_execs_spsa.append(num_executions)
    
    iteration_num = len(cost_list)
    if iteration_num % 1 == 0:
        
        data_counter += 1
        data_counter = data_counter % len(data)
        
        end_time = timeit.time.time()
        print(
            f"Iteration = {iteration_num}, "
            f"Number of device executions = {num_executions}, "
            f"Cost = {cost_val}"
        )
        print('time elapsed: {:.2f}'.format(end_time - start_time))
        
        
res = minimizeSPSA(
    fidelityLossMultiple,
    x0=init_params,
    niter=niter_spsa,
    paired=False,
    c=0.10,
    a=0.1,
    callback=callback_fn,
)



# %% 

valid_states = []
for k in cur_losses_for_instances:
    valid_states.append(k.detach().numpy())
valid_states = np.array(valid_states)
valid_states_map = valid_states < -3

timeEvolution(torchH, input_states[1], 1)
input_states[2]

# Prepare seperate inputs for valid states
valid_input_states = []

for i in range(0, len(input_states), NUMBER_OF_EVOLUTIONS + 1):
    for k in range(0,NUMBER_OF_EVOLUTIONS + 1):
        counter = False
        print(i, k)
        if(valid_states_map[i + k] == True):
            first_input_batch = []
            counter = True
            j = k + 1
            while(j < NUMBER_OF_EVOLUTIONS + 1 and valid_states_map[j] == True):
                print(i , j)
                if(counter == True):
                    # first_input_batch.append(input_states[i + k])
                    first_input_batch.append(i+k)
                    counter = False
                    # first_input_batch.append(input_states[i + j])
                first_input_batch.append(i+j)
                j += 1
            
            if(len(valid_input_states) > 0 and set(first_input_batch) <= set(valid_input_states[-1])):
                continue
            valid_input_states.append(first_input_batch)


# %% 

dev1 = qml.device("default.qubit", wires=n_qubit_size)
input_index = 0
set_index = 0
@qml.qnode(dev1, interface='torch')
def circuit(params):
    global input_index
    global set_index 
    
    if(set_index == len(valid_input_states)):
        set_index = 0
        
    print(set_index , input_index)
    inputs = input_states[valid_input_states[set_index][input_index]]
    targets = input_states[valid_input_states[set_index][input_index + 1]]
    
    density = quantumOuter(targets)
    
    
    
    
    qml.QubitStateVector(inputs, wires = range(0, n_qubit_size))
   
    for i in range(n_qubit_size): # 4 qubits
        qml.Rot(*params[i][0], wires=i)
    for idx, i in enumerate(range(n_qubit_size)):
        ctr=0
        for jdx, j in enumerate(range(n_qubit_size)):
            if(i==j):
                pass
            else:
                qml.CRot( *params[i][ctr + 1], wires= [i,j])
                ctr += 1
    for i in range(n_qubit_size): # 4 qubits
        qml.Rot(*params[i][n_qubit_size], wires=i)
        
    input_index += 1
    if(input_index == len(valid_input_states[set_index])  - 1):
        set_index += 1
        input_index = 0
    return qml.expval(qml.Hermitian(density, wires=list(range(0,n_qubit_size)) ))



init_params = np.random.rand(n_qubit_size, n_qubit_size + 1,  3)
init_params = torch.from_numpy(init_params)
init_params.requires_grad = True


# init_params = torch.Tensor(init_params, requires_grad = True)
# target_state = np.ones(2**4)/np.sqrt(2**4)
# density = np.outer(target_state, target_state)
# density = np.outer(target_state, target_state)

quantumOuter = lambda inputs: torch.outer(inputs.conj().T, inputs)




def isHermitian(H):
    return qml.math.allclose(H ,H.T.conj())
isHermitian(density)

def cost(var):
    return 1-circuit(var)


res = (cost(init_params))
res.backward()


opt = torch.optim.Adam([init_params], lr = 0.1)

steps = 200

def closure():
    opt.zero_grad()
    loss = cost(init_params)
    print(loss)
    loss.backward()
    return loss

for i in range(steps):
    opt.step(closure)



# %% Testing, w/ vector similarity

# for i in range(10):    
#     batch_id = np.arange(n_data)
#     np.random.shuffle(batch_id)
#     state = tensorlist[batch_id[i]]
    
#     out = qAutoencoder(state, training_mode = False )
    
#     # input_state  = np.kron(np.kron([1,0], [1,0]),tensorlist[n_qubit_size][i].numpy())
#     input_state  = np.kron(np.kron([1,0], [1,0]), state.numpy())
#     result = np.array(dev.state.detach())
#     # how similar is the output to the input? 
#     similarity = sum(np.abs((result-input_state) ** 2)) / (2**n_qubit_size)
#     print('similarity is {:.9f}'.format(similarity))

# %%  NEXT THINGS TO DO

# PREPARE THE DEV.STATE WITH THE INPUT THAT IS PREPARED WITH QARBITRARYSTATE - DONE
# Prepare examplary hamiltonians, 
# 

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


# PATH

# PATH2 = './inputs_states.npy'

X = torch.load(PATH2)

torch.save(torchH, './torchH.npy')
torch.save(coeffs, './coeffs.npy')
torch.save(randomPauliGroup, './randomPauliGroup.npy')
torch.save(qAutoencoder.state_dict(), './qAutoencoder.npy')
torch.save(input_states, './input_states.npy')
H

# %% 


# for n_qubits in range(1,10):
#     vec_dev = qml.device('default.qubit', wires = n_qubits)
    
#     @qml.qnode(vec_dev)
#     def qCircuit(inputs):
#         qml.QubitStateVector(inputs, wires = range(0,n_qubits))
        
#         return qml.probs(range(0,n_qubits))
#         return [qml.expval(qml.PauliZ(q)) for q in range(0,1)]
    
#     print(qCircuit(tensorlist[n_qubits][0]))
