# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 00:03:14 2022

@author: burak
"""

import numpy as np

# Pauli Matrices
z = np.array([[1,0] , [0j, -1]])
y = np.array([[0,-1j] , [1j, 0]])
x = np.array([[0,1] , [0j + 1, 0]])
I = np.eye(2)

# Annihilation, Creation Operators
Splus = x + 1j*y
Sminus = x - 1j*y

createBitString = lambda x,y=9: str(bin(x)[2:].zfill(y))
int2bit = lambda x,y=9: str(bin(x)[2:].zfill(y))
bit2int = lambda b: int("".join(str(bs) for bs in b), base = 2) 
quantumOuter = lambda i1,i2: np.outer(i1, i2.conj().T) # braket

def commuteCheck(A,B):
    # Check if two matrices A and B commute
    return np.allclose(A@B, B@A, 10e-3)
    
def KroneckerProduct(listOfQubits, pauli, N):
    out = np.array([1])
    for i in range(N):
        if(i in listOfQubits):
            out = np.kron(out, pauli)
        else:
            out = np.kron(out, I)
    return out


def KroneckerProductString(listOfQubits, paulis, N):
    '''
    Given a Pauli String, return the Kronecker product for N qubits
    i.e (1,2) (Z,Z) (5) = I Z Z I I 

    Parameters
    ----------
    listOfQubits : list
        Qubit lines where the operators apply
    paulis : Square matrix
        The operator
    N : int
        Number of qubit lines

    Returns
    -------
    out : matrix
        The kronecker product

    '''
    out = np.array([1])
    for i in range(N):
        if(i in listOfQubits):
            idx = listOfQubits.index(i)
            out = np.kron(out, paulis[idx])
        else:
            out = np.kron(out, I)
    return out


def findGroundStates(H, size = 9):
    '''

    Parameters
    ----------
    H : Hamiltonian

    Returns
    -------
    ground_state : Ground state vector
    lowest_energy_states : lowest eigenvalue index
    lowest_energy : lowest eigenvalue

    '''
    ew , ev = np.linalg.eig(H)
    
    lowest_energy_state = np.argmin(ew)
    ground_state = ev[:, lowest_energy_state]
    solutions = np.where(np.abs(ground_state) > 10e-2)
    
    return [createBitString(solution, size) for solution in solutions[0]], ground_state, solutions, lowest_energy_state



def checkConstraints(solution, n = 3):
    '''
    Given a one-hot-encoding, find the edges in a solut

    Parameters
    ----------
    solution : str
        One-hot encoded solution
    n : int
        number of vertices in the graph

    Returns
    -------
    bool, True if the solution satisfies the constraints

    '''
    
    for q in range(n):
        for i in range(n):
            for j in range(n):
                if(i != j):
                    u = q * n + i
                    v = q * n + j
                    if( solution[u] == '1' and solution[v] == '1'):
                        return False
    for j in range(n):
        for q1 in range(n):
            for q2 in range(n):
                if(q1 == q2):
                    continue
                u = q1 * n + j
                v = q2 * n + j
                if( solution[u] == '1' and solution[v] == '1'):
                    return False
                
    return sum([int(el) for el in solution]) == n



def testCheckConstraints(n = 3):
    '''
    Unit test for checkConstraint method

    Parameters
    ----------
    n : int
        number of vertices in the graph

    Returns
    -------
    bool

    '''
    ground_truth = [84,98,140,161,266,273]
    sol = []
    for i in range(511):
        if(checkConstraints(int2bit(i), 3)):
            sol.append(i)
    
    return sol == ground_truth
            