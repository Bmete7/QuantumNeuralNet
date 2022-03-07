# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 18:30:55 2021

@author: burak
"""

import torch
import numpy as np
import timeit
import random
import pennylane as qml

# %% Functions for preparing the data
# Built specifically to L2 normalize the randomly generated complex vectors


norm = lambda x:np.linalg.norm(x)
normalize = lambda x: x/norm(x)

torchnorm = lambda x:torch.norm(x)
torchnormalize = lambda x: x/torchnorm(x)

# %% Data creation
def dataPreparation(saved = True, save_tensors = False, method = 'kron', number_of_samples = 2000, number_of_qubits = 11):
    # prepare normalized qubits for any number of qubits
    # save flag saves the vectors as .npy
    
    # method: str {'kron' ,'state_prepare'}
    if(saved == True):
        try:
            PATH2 = './inputs_states.npy'
            tensorlist = np.load(PATH2, allow_pickle = True)
        except:
            pass

    else:
        n_qubits = number_of_qubits
        tensorlist = []
        tensorlist_np = []
        if(method == 'kron'):

            add_real = np.arange(0,20)
            add_imag = np.arange(0,20)
    
            norm_coef = (np.abs(20+20j))
    
            states = []
            for i in range(20):
                for j in range(20):
                    first_prob = ((add_real[i] + add_imag[j] * 1j) / norm_coef)
                    second_prob = np.sqrt(1- np.abs(first_prob)**2)
                    states.append([first_prob, second_prob])
    
            
            for i in range(number_of_samples):
                state = 1
                for j in range(n_qubits):
                    state = np.kron(state,states[np.random.randint(len(states))])
                tensorlist.append(torch.Tensor(state))
            
            return tensorlist
        
        elif(method== 'state_prepare'):
            data_preparation_device = qml.device('default.qubit', wires = n_qubits)
            start_time = timeit.time.time()
            @qml.qnode(data_preparation_device)
            def qData(weights, cnots):
                for i in range(n_qubits):
                    
                    qml.Rot( *weights[i], wires = i)
                    
                # for tup in cnots:
                #     qml.CNOT(wires = tup)
                # for i in range(n_qubits):
                #     qml.RZ( weights[i][0][1] , wires = i)
                #     qml.RY( weights[i][1][1] , wires = i)
                #     qml.RZ( weights[i][2][1] , wires = i)
                return qml.state()
            
            
            for k in range(number_of_samples):
                cnots = []
                ctr = 0
                for i in range(n_qubits):
                    for j in range(n_qubits):
                        rand = np.random.randint(n_qubits**2)
                        if (i==j or rand >= n_qubits/2):
                            continue
                        if(j == i - 1 or j == i + 1):
                            cnots.append((i,j))
                            ctr += 1
                        if(ctr >= 1):
                            break
                    if(ctr >= 1):
                        break
                weights  = torch.rand(n_qubits,3) * (torch.pi) * 2 # - (torch.pi)
                random.shuffle(cnots)
                res = qData(weights, cnots)
                tensorlist_np.append(res)
                state = torch.zeros_like(torch.Tensor(res), dtype=torch.cdouble)
                state[:] = torch.Tensor(res.real[:])
                state[:] += torch.Tensor(res.imag[:]) * 1j
                
                
                # In case results are not numerically stable due to floating point errors
                ctr = 0
                while(torchnorm(state) != 1.0 and ctr <= 5):
                    state = torchnormalize(state)
                    ctr += 1
                if(torchnorm(state) == 1.0):
                    tensorlist.append((state))
            end_time = timeit.time.time()
            print('Time elapsed for dataset creation: ' + ' : {:.2f}'.format(end_time-start_time))
            return tensorlist
        
        if(save_tensors == True):
            PATH2 = './inputs_states.npy'
            torch.save(tensorlist, PATH2)    
    
    return tensorlist
    
    
