# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 18:07:10 2022

@author: burak
"""

import pennylane as qml
import numpy as np
from tqdm.notebook import tqdm

# de , deneme = dataPreparation(saved = False, save_tensors = False, method = 'state_prepare', number_of_samples = 5)
# inputs = tensorlist[2]

# density = quantumOuter(inputs)


dev1 = qml.device("default.qubit", wires=11)
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
    
    
    
    
    qml.QubitStateVector(inputs, wires = range(0, 11))
   
    for i in range(11): # 4 qubits
        qml.Rot(*params[i][0], wires=i)
    for idx, i in enumerate(range(11)):
        ctr=0
        for jdx, j in enumerate(range(11)):
            if(i==j):
                pass
            else:
                qml.CRot( *params[i][ctr + 1], wires= [i,j])
                ctr += 1
    for i in range(11): # 4 qubits
        qml.Rot(*params[i][11], wires=i)
        
    input_index += 1
    if(input_index == len(valid_input_states[set_index])  - 1):
        set_index += 1
        input_index = 0
    return qml.expval(qml.Hermitian(density, wires=list(range(0,11)) ))



init_params = np.random.rand(11, 12, 3)
init_params = torch.from_numpy(init_params)
init_params.requires_grad = True


init_params = torch.Tensor(init_params, requires_grad = True)
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


opt = qml.GradientDescentOptimizer(stepsize=0.3) 
# set the number of steps
steps = 100
# set the initial parameter values
params = init_params




for i in tqdm(range(steps)):
    # update the circuit parameters
    params = opt.step(cost, params)
    if (i + 1) % 1 == 0:
        print("Cost after step {:5d}: {: .7f}".format(i + 1, cost(params)))
        
# %% 
dev2 = qml.device("default.qubit", wires=11)
@qml.qnode(dev2, interface='torch')
def circuitQAE(params):
    global index
    
    
    
    inputs = input_states[index]
    qml.QubitStateVector(inputs, wires = range(0, 11))
    # targets = input_states[valid_input_states[set_index][input_index + 1]]
    # density = quantumOuter(targets)
    
   
    for i in range(11): # 4 qubits
        qml.Rot(*params[i][0], wires=i)
    for idx, i in enumerate(range(11)):
        ctr=0
        for jdx, j in enumerate(range(11)):
            if(i==j):
                pass
            else:
                qml.CRot( *params[i][ctr + 1], wires= [i,j])
                ctr += 1
    for i in range(11): # 4 qubits
        qml.Rot(*params[i][11], wires=i)
    
    return qml.expval(qml.PauliZ(0))
#%%
index= 0
init_params = qml.init.strong_ent_layers_normal(1, 11)
# qml.math.allclose(torch.Tensor(init_params), init_params)
init_params = torch.Tensor(init_params)
init_params.requires_grad = True

init_params = np.random.rand(11, 12, 3)
init_params = torch.from_numpy(init_params)
init_params.requires_grad = True
# %% 
opt = torch.optim.Adam([init_params], lr = 0.007)

steps = 3
def cost(var):
    return 1-circuitQAE(var)
losses = torch.zeros(30)
def closure():
    global index
    opt.zero_grad()
    loss = cost(init_params)
    losses[index ] = loss
    loss.backward()
    return loss
start_time = timeit.time.time()
for i in range(steps):
    
    opt.step(closure)
    index += 1
    # if(idx == len(input_states)):
    if(index >= 30):
        
        
        index = 0
        end_time = timeit.time.time()
        print('Time elapsed for the epoch' +  str(steps)  + ' : {:.2f}'.format(end_time - start_time))
        print(np.sum(losses.detach().numpy()))
        start_time = timeit.time.time()
        
# %% 

init_params = U

dev1 = qml.device("default.qubit", wires=11)

@qml.qnode(dev1)
def ArbitraryU(params, interface='torch'):
    
    qml.QubitUnitary(params , wires = range(0,11))
    return qml.expval(qml.PauliX(2))


dev1 = qml.device("default.qubit", wires=n_qubit_size)




def cost(var):
    return 1-ArbitraryU(var)


opt = torch.optim.Adam([U], lr = 0.1)
steps = 5
def closure():
    opt.zero_grad()
    print('a')
    loss = cost(init_params)
    print(loss)
    loss.backward()
    return loss

for i in range(steps):
    opt.step(closure)
    
    
# %% 

dev1 = qml.device("default.qubit", wires=4)

@qml.qnode(dev1)
def qae(inputs, interface='torch'):
    qml.QubitStateVector(inputs, wires = range(2,4))
    n_layers
    

                    for l in range(n_layers):
                        for idx, i in enumerate(range(.n_auxillary_qubits + .n_latent_qubits, .n_total_qubits)):
                            qml.Rot(*weights_r[l, 0, idx] , wires = i)
                        for idx, i in enumerate(range(.n_auxillary_qubits + .n_latent_qubits, .n_total_qubits)):
                            ctr=0
                            for jdx, j in enumerate(range(.n_auxillary_qubits + .n_latent_qubits, .n_total_qubits)):
                                if(i==j):
                                    pass
                                else:
                                    qml.CRot( *weights_cr[l, idx, ctr], wires= [i, j])
                                    ctr += 1
                        for idx, i in enumerate(range(.n_auxillary_qubits + .n_latent_qubits, .n_total_qubits)):
                            qml.Rot(*weights_r[l, 1, idx] , wires = i)