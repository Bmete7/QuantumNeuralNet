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
def dataPreparation(saved = True, save_tensors = False):
    # prepare normalized qubits for any number of qubits
    # save flag saves the vectors as .npy
    
    if(saved == True):
        try:
            PATH2 = './inputs_states.npy'
            tensorlist = np.load(PATH2, allow_pickle = True)
        except:
            pass

    else:
        
        
        n_qubits = 8
        tensorlist = []
        
        
        
        
        
        
        
        add_real = np.arange(0,20)
        add_imag = np.arange(0,20)

        norm_coef = (np.abs(20+20j))

        states = []
        for i in range(20):
            for j in range(20):
                first_prob = ((add_real[i] + add_imag[j] * 1j) / norm_coef)
                second_prob = np.sqrt(1- np.abs(first_prob)**2)
                states.append([first_prob, second_prob])

        
        for i in range(2000):
            state = 1
            for j in range(n_qubits):
                state = np.kron(state,states[np.random.randint(len(states))])
            tensorlist.append(torch.Tensor(state))
        
        return tensorlist
    
        datadev = qml.device('default.qubit', wires = n_qubits)
        @qml.qnode(datadev)
        def qData(weights, cnots):
            for i in range(n_qubits):
                qml.RZ( weights[i][0][0] , wires = i)
                qml.RY( weights[i][1][0] , wires = i)
                qml.RZ( weights[i][2][0] , wires = i)
                
            # for tup in cnots:
            #     qml.CNOT(wires= tup)
            for i in range(n_qubits):
                qml.RZ( weights[i][0][1] , wires = i)
                qml.RY( weights[i][1][1] , wires = i)
                qml.RZ( weights[i][2][1] , wires = i)
            return qml.expval(qml.PauliX(0))
        
        for i in range(100):
            
            cnots = []
            for i in range(n_qubits):
                for j in range(n_qubits):
                    rand = np.random.randint(n_qubits**2)
                    if (i==j or rand >= n_qubits/2):
                        continue
                    cnots.append((i,j))
            
            weights  = torch.rand(n_qubits,3,2) * (torch.pi) * 2 - (torch.pi)
            random.shuffle(cnots)
            qData(weights, cnots)
            ctr = 0
            
            state= torch.Tensor(datadev.state)
            
            while(torchnorm(state) != 1.0 and ctr <= 5):
                state = torchnormalize(state)
                ctr += 1
            if(torchnorm(state) == 1.0):
                tensorlist.append((state))
        if(save_tensors == True):
            PATH2 = './inputs_states.npy'
            torch.save(tensorlist, PATH2)    
    
    
    return tensorlist
    
