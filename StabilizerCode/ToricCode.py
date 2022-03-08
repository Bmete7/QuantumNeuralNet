# -*- coding: utf-8 -*-
"""
@author: burak
"""

'''
This codes simulates the Toric Code Hamiltonian, and the circuit Proposed in 
the paper: https://arxiv.org/abs/2104.01180
for the 12 qubit setting (e.g 4 plaquette operators acting on the corresponding lattice sites)

'''




# %% 
import pennylane as qml
import numpy as np

N_QUBITS = 12

pauli_Z = np.array([[1,0] , [0j, -1]])
pauli_Y = np.array([[0,-1j] , [1j, 0]])
pauli_X = np.array([[0,1] , [0j + 1, 0]])
I = np.eye(2)
dev = qml.device("default.qubit", wires= N_QUBITS)

# %% Utility Functions
def isHermitian(U):
    '''
    Check if the matrix is Hermitian
    '''
    return ((U.conj().T)== U).all()    
def isUnitary(U):
    '''
    Check if the matrix is unitary
    '''
    N = U.shape[0]
    return ((U.conj().T) @ U == np.eye(N)).all() # is unitary 

def CheckResult(ground_state, H):
    
    '''
    Parameters
    ----------
    ground_state : 
        Ground State of the Toric Code Hamiltonian (TCH)
    H : 
        TCH, used for evaluated circuit measurement

    Returns
    -------
    Bool
        Is the result equal to the TCH ground state 

    '''
    out = H @ ToricCodeCircuit()
    difference = (ground_state / out)
    coefs = []
    for dif in difference:
        if (np.absolute(dif) > 1e-10):
            coefs.append(dif)
            
    indices = np.argwhere(~np.isnan(difference))
    print(coefs)
    return all(difference[indices])
'''
Numbering of the qubits from the paper:
    
    0    1
2     3     4  
    5    6
7     8     9
    10   11 

'''

def isUpperQubit(i):
    '''
    Hadamard gates apply to the upper qubits from the paper.
    Check if those qubits belong to the upper qubits:

    Parameters
    ----------
    i : qubit index

    bool: true if i is an upper qubit

    '''
    return (isRightUpperQubit(i) or isLeftUpperQubit(i))

def isLeftUpperQubit(i):
    if(N_QUBITS < 5):
        return (i % 5 == 0)    
    else:
        return (i % 5 == 0 )

def isRightUpperQubit(i):
    if(N_QUBITS < 5):
        return (i % 5 == 0)
    else:
        return (i % 5 == 1 )    
def isLeftMostQubit(i):
    if(N_QUBITS < 5):
        return (i % 5 == 1)
    else:
        return (i % 5 == 2 )    

# for i in range(12):
#     print(isLeftUpperQubit(i), isRightUpperQubit(i), isUpperQubit(i))

def applyingCNOT_order():
    '''
    

    Returns
    -------
    List:
        List of tuples for CNOT applications for the ground state preparation
        algorithm
        (see page.3 of the paper, Fig-1.b)

    '''
    order_list = []
    if(N_QUBITS > 5):
        for i in range(N_QUBITS):
            if(isRightUpperQubit(i)):
                if( i + 3 < N_QUBITS):
                    order_list.append([i, i + 2])
                
        for i in range(N_QUBITS):
            if(isLeftUpperQubit(i)):
                if( i + 3 < N_QUBITS):
                    order_list.append([i, i + 2])
        
        for i in range(N_QUBITS):
            if(isLeftUpperQubit(i)):
                if( i + 3  < N_QUBITS):
                    order_list.append([i, i + 3])
                
        for i in range(N_QUBITS):
            if(isLeftMostQubit(i)):
                order_list.append([i, i + 3])
    else:
        order_list.append([0,1])
        order_list.append([0,2])
        order_list.append([1,3])
    return order_list
        
# %% Simulate the proposed circuit

@qml.qnode(dev)
def ToricCodeCircuit():
    cnot_order_list = applyingCNOT_order()
    
    for i in range(N_QUBITS):
        if(isUpperQubit(i)):
            qml.Hadamard(wires = i)
    
    for cnot in cnot_order_list:
        qml.CNOT(wires = cnot)
    return qml.state()

# %% Prepare the Toric Code Hamiltonian

def KroneckerProduct(listOfQubits, pauli):
    out = np.array([1])
    for i in range(N_QUBITS):
        if(i in listOfQubits):
            out = np.kron(out, pauli)
        else:
            out = np.kron(out, I)
    return out

ListOfPlaquettes = [
    [0,2,3,5], 
    [1,3,4,6],
    [5,7,8,10],
    [6,8,9,11]
    ]

ListOfStars = [
    [0, 2], 
    [0, 1, 3],
    [1, 4],
    [2, 5, 7],
    [3, 5, 6, 8],
    [4, 6, 9],
    [7, 10],
    [8, 10, 11],
    [9, 11]
    ]

StarOperators = [ KroneckerProduct(star, pauli_Z) for star in ListOfStars]
PlaquetteOperators = [ KroneckerProduct(plaquette, pauli_X) for plaquette in ListOfPlaquettes ]

A = sum(StarOperators)
B = sum(PlaquetteOperators)

ToricCodeHamiltonian = -1 * (A + B)
ew, ev = np.linalg.eig(ToricCodeHamiltonian)
ground_state = ev[:, np.argmin(ew)]
print( CheckResult(ground_state, ToricCodeHamiltonian))
simulation = ToricCodeCircuit()
# U = (( I + B)/ np.sqrt(2))


    


