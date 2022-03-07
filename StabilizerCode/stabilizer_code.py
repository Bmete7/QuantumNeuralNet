# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 19:34:19 2022

@author: burak
"""

import pennylane as qml
import numpy as np

pauli_Z = np.array([[1,0] , [0j, -1]])
pauli_Y = np.array([[0,-1j] , [1j, 0]])
pauli_X = np.array([[0,1] , [0j + 1, 0]])

devv = qml.device("default.qubit", wires=4)

@qml.qnode(devv)
def qCirc():
    qml.Hadamard(wires = 0)
    qml.CNOT(wires = [0,1])
    qml.CNOT(wires = [0,2])
    qml.CNOT(wires = [1,3])
    return qml.state()

z1 = np.kron(np.kron(np.kron(pauli_Z , pauli_Z) , np.eye(2)) , np.eye(2))
z2 = np.kron(np.kron(np.kron(pauli_Z , np.eye(2)) , pauli_Z) , np.eye(2))
z3 = np.kron(np.kron(np.kron(np.eye(2) , pauli_Z) , np.eye(2)) , pauli_Z)
z4 = np.kron(np.kron(np.kron(np.eye(2) , np.eye(2)) , pauli_Z) , pauli_Z)

Plaquette = np.kron(np.kron(np.kron(pauli_X , pauli_X) ,  pauli_X) , pauli_X)
I = np.eye(16)
U = ((I+Plaquette)/ np.sqrt(2))
((U.conj().T)== U).all() # is hermitian
((U.conj().T) @ U == np.eye(16)).all() # is unitary

H =(z1 + z2 + z3 + z4 + Plaquette) * -1
el, ev = np.linalg.eig(H)
ground_state = ev[:, np.argmin(el)]

H @ qCirc()
ZZ = np.kron(pauli_Z,pauli_Z)
XX = np.kron(pauli_X,pauli_X)


np.linalg.eig((ZZ + XX ) * -1)

aaa = np.zeros(16, dtype = 'complex128')
aaa[0] = 1+0j
aaa[-1] = 1+0j
aaa /= np.sqrt(2)
HH @ aaa



s1 + np.eye(4)
qml.CNOT(wires = [0,1]).matrix @ np.kron(qml.Hadamard(wires = 0).matrix , np.eye(2)) 


s1 = np.kron(pauli_X, pauli_X)
s2 = np.kron(pauli_Z, pauli_Z)
np.linalg.eig((s1+s2))
from scipy.linalg import expm
expm(-1j * (s1+s2) ) 
