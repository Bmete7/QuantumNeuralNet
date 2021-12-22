# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 15:58:13 2021

@author: burak
"""

import pennylane as qml
import torch
import numpy as np
import timeit
from DataCreation import dataPreparation
from QAutoencoder.QAutoencoder import QuantumAutoencoder
from scipy.linalg import expm, logm

n_qubit_size = 8

coeffs = [ np.random.rand() * 2 - 1 for i in range(n_qubit_size)]
pauliSet = [qml.PauliX(0).matrix, qml.PauliZ(0).matrix, qml.PauliY(0).matrix]

randomPauliGroup = [np.random.randint(3) for i in range(n_qubit_size)]
def findHermitian(coeffs, randomPauliGroup):
    I = np.eye(2)
    pauliSet = [qml.PauliX(0).matrix, qml.PauliZ(0).matrix, qml.PauliY(0).matrix]
    hamiltonian = 0j
    for i in range(n_qubit_size):
        pauli_word = np.array([1])
        cur_hermitian = pauliSet[randomPauliGroup[i]] * coeffs[i]
        for j in range(n_qubit_size):
            if(i==j):
                pauli_word = np.kron(pauli_word, cur_hermitian)
            else:
                pauli_word = np.kron(pauli_word, I)
                
        hamiltonian += pauli_word
    return hamiltonian
H = findHermitian(coeffs, randomPauliGroup)
torchH = torch.zeros_like(torch.Tensor(H), dtype=torch.cdouble)

torchH[:,:] = torch.Tensor(H.real[:,:])
torchH[:,:] += torch.Tensor(H.imag[:,:]) * 1j

Hamiltonian = qml.Hermitian(H, np.arange(8))

hamiltonian_dev = qml.device('default.qubit', wires = n_qubit_size, shots= 1)
@qml.qnode(hamiltonian_dev)
def measureEnergy(H, psi):
    qml.QubitStateVector(psi, wires = range(0,n_qubit_size))
    print(H)
    return qml.sample(Hamiltonian)
measureEnergy(Hamiltonian, tensorlist[0])

(Hamiltonian.eigvals)

def timeEvolution(local_hamiltonian, psi, timestamp):
    # U = expm(-1j * H * t )
    U = torch.matrix_exp(local_hamiltonian * -1j * timestamp)
    
    return U @ psi

k = torch.Tensor(1)
#1 They all give the same result (since it is 1-local)
'''expm(-1j * H * 1 )
k.values()
torch.matrix_exp(torchH * -1j * 1)

a=1
H.real
for i in range(n_qubit_size):
    a = np.kron(a , expm(-1j * 1 *pauliSet[randomPauliGroup[i]] * coeffs[i] ))'''
#1 end

# H, torchH, tensorlist[]

evolved_states = []

for i in range(10):    
    batch_id = np.arange(n_data)
    np.random.shuffle(batch_id)
    psi = tensorlist[batch_id[i]]
    
    for t in range(1,6):
        evolved_states.append(timeEvolution(torchH, psi, t))

fidelities = []
fidelities_original = []
for i in range(50):
    
    
    state = evolved_states[batch_id[i]]
    state = evolved_states[i]
    out = qAutoencoder(state, training_mode = True )
    fidelities.append(out.detach().numpy())
    if(i%5 == 0):
        out2 = qAutoencoder(tensorlist[batch_id[int(i/5)]], training_mode = True )
        fidelities_original.append(out2.detach().numpy())
    print('Fidelity is {:.9f}'.format(out.detach().numpy().squeeze()))