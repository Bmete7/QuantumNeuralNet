# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 15:58:13 2021

@author: burak
"""

import pennylane as qml
import torch
import numpy as np


def findHermitian(coeffs, randomPauliGroup, n_qubit_size):
    '''
    Given Pauli strings, it prepares an arbitrary Hamiltonian

    Parameters
    ----------
    coeffs : list
        List of pauli coefficients
    randomPauliGroup : list
        List of Pauli strings
    n_qubit_size : TYPE
        Number of qubits

    Returns
    -------
    hamiltonian : np.ndarray
        Hermitian matrix

    '''
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



def timeEvolution(local_hamiltonian, psi, timestamp):
    # U = expm(-1j * H * t )
    U = torch.matrix_exp(local_hamiltonian * -1j * timestamp)
    
    return U @ psi

'''expm(-1j * H * 1 )
k.values()
torch.matrix_exp(torchH * -1j * 1)

a=1
H.real
for i in range(n_qubit_size):
    a = np.kron(a , expm(-1j * 1 *pauliSet[randomPauliGroup[i]] * coeffs[i] ))'''
