# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 19:30:43 2022

@author: burak
"""

import pennylane as qml
import numpy as np
import torch

class Utils():
    '''
    Utility Functions for QAutoencoder and Hamiltonian Simulator
    
    Attributes
    ----------
    dev:
        quantum device object
    
    
    Methods
    ----------
    
    returnState(params):
        returns the state vector for a given sequence
        
    '''
    def __init__(self, dev, n_qubit_size = 8):
        self.dev = dev
        self.n_qubit_size = n_qubit_size
        
        @qml.qnode(self.dev)
        def returnState(psi):
            qml.QubitStateVector(psi, wires = range(0, self.n_qubit_size))
            return qml.state()
        
        self.ReturnState = returnState
    def getState(self, psi):
        return self.ReturnState(psi)
    
norm = lambda x:np.linalg.norm(x)
normalize = lambda x: x/norm(x)

torchnorm = lambda x:torch.norm(x)
torchnormalize = lambda x: x/torchnorm(x)
quantumOuter = lambda inputs: torch.outer(inputs.conj().T, inputs)
wv = torch.ones(4, dtype = torch.cdouble) / torch.sqrt(torch.Tensor([4]))

qml.Hermitian(quantumOuter(wv), wires = [0,1]).eigendecomposition['eigvec'].T[-1]

def createHamiltonian(edges, n,  pauli = 'Z'):
    pauli_Z = np.array([[1,0] , [0j, -1]])
    pauli_Y = np.array([[0,-1j] , [1j, 0]])
    pauli_X = np.array([[0,1] , [0j + 1, 0]])
    H_locals = []
    H_final = 0
    started = False
    def createLocalHamiltonian(i,j, pauli = pauli):
        emp= np.array([1])
        for k in range(n):
            if(k==i or k == j):
                emp = np.kron(emp,pauli_Z)
            else:
                emp = np.kron(emp,np.eye(2))
        return emp
    num_of_hamiltonians = len(edges)
    for edge in edges:
        H_local = createLocalHamiltonian(edge[0] , edge[1] ) / np.sqrt(num_of_hamiltonians)
        if(started == False):
            started = True
            H_final = H_local * 1
        else:
            H_final = H_final + H_local

    return H_final   

H = createHamiltonian([[1,0] , [0,3] , [1,2] , [2,3] ] , 4 )
np.linalg.eig(H)
