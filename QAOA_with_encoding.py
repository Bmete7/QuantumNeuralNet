# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:44:00 2022

@author: burak
"""

# import pennylane as qml


import numpy as np
import networkx as nx
from scipy.linalg import expm
import time
import qiskit
from QAutoencoder.Utils import *
from qiskit.algorithms.optimizers import SPSA
from copy import deepcopy





# %% Problem Parameters

n = 3  # number of nodes in the graph
N = n**2 # number of qubits to represent the one-hot-encoding of TSP

G = nx.Graph()

for i in range(n):
    G.add_node(i)

G.add_edge(0,1, weight  = 50)
G.add_edge(0,2, weight  = 2)
G.add_edge(1,2, weight  = 2)

nx.draw(G, with_labels=True, alpha=0.8)
adj = nx.adjacency_matrix(G).todense()
edges = list(G.edges)

# %% 
# 2 Backends, one(qasm) for simulating the circuit
# the other (unitary) is for debugging purposes

backend = qiskit.BasicAer.get_backend('qasm_simulator')
unitary_backend = qiskit.BasicAer.get_backend('unitary_simulator')


def QAOA(edges, adj, params = np.random.rand(6)*np.pi / 10,  initial_state = None, decode = True, layers = 1):
        
    '''
    Building the QAOA Ansatz, that includes U_Hc and encoding scheme,
    
    first e^(-i Hc beta) applies, which only shifts the phases, since the eigenvectors 
    are all in the computational basis state. Then the encoder maps those into the latent space.
    
    After swapping the trash states with ancilla qubits, and decoding again, we acquire only the states
    that obeys the constraints
    

    Returns
    -------
    circ : qiskit circuit

    '''
    
    global n, N
    
    '''
    
    i.e when the input is 98, output is 001000000, the 1 being the 6th qubit!
    the right most one is the 0th qubit, left most is the 8th qubit
    graph3_dict = {'001010100': '000000000', 
                   '001100010': '000000001',
                   '010001100': '000000010',
                   '010100001': '000000011',
                   '100001010': '000000100',
                   '100010001': '000000101'} 
    '''
    
    ancillas = 6
    c = qiskit.ClassicalRegister(3) # stores the measurements
    if(decode == False):
        c = qiskit.ClassicalRegister(3) # stores the measurements
    q = qiskit.QuantumRegister(N+ ancillas)
    circ = qiskit.QuantumCircuit(q,c)
    
    beta = params[:layers]
    gamma = params[layers:]
    
    def initializePlusState():
        # initial_state = np.ones((2**N)) / (np.sqrt(2) **N )
        initial_state = np.zeros((2**N))
        initial_state[266] = 1
        circ.initialize(initial_state , [q[i] for i in range(N)])
        
    # def initializer(initial_state):
    #     # b = np.ones((2**N)) / (np.sqrt(2) **N ) # equal superposition
    #     if(initial_state is None):
    #         # initial_state = np.zeros((2**N), dtype = np.complex128)
    #         # initial_state[266] = 1 # or the statevector for one of the feasible solutions
    #         initial_state = np.ones((2**N)) / (np.sqrt(2) **N ) # equal superposition
    #         circ.initialize(initial_state , [q[i] for i in range(N)])
    #     else:
    #         if(stateVectorCheck(initial_state) == False):
    #             raise('state vector is not valid')
            
    #         circ.initialize(initial_state , [q[i] for i in range(N)])
    
    N_index = N - 1 #since qiskit has a different ordering (LSB for the first qubit), we subtract the index from this number
    
    def costHamiltonian(l):
        for edge in edges:
            u, v = edge
            for step in range(n - 1):
                
                q_1 = (u * n) + step
                q_2 = (v * n) + step + 1 # (+1 because we want that node visited in the next step)
                
                q_1_rev = (u * n) + step + 1 # (reverse path from above)
                q_2_rev = (v * n) + step  
                
                q_1 = (N - 1) - q_1
                q_2 = (N - 1) - q_2
                
                q_1_rev = (N - 1) - q_1_rev
                q_2_rev = (N - 1) - q_2_rev
                
                circ.rzz(2 * beta[l] * (adj[u,v]) , q_1 , q_2)
                circ.rzz(2 * beta[l] * (adj[u,v]) , q_1_rev, q_2_rev) # exponential of the cost Hamiltonian
                
                
    
    def mixerHamiltonian(l):
        for i in range(N):
            circ.rx( gamma[l] , i) # exponential of the mixer Hamiltonian
    
    # Encoder Circuit
    def encoderCircuit(n):
        if(n == 2):
            circ.cnot(3,0)
            circ.x(3)
            circ.cnot(3,2)
            circ.cnot(3,1)
            circ.x(3)
        elif (n == 3):
            circ.x( N_index - 0)
            circ.x( N_index - 1)
            circ.toffoli( N_index - 0, N_index - 1, N_index - 2)
            circ.x( N_index - 0)
            circ.x( N_index - 1)
            circ.toffoli( N_index - 0, N_index -4, N_index - 2)
            circ.toffoli( N_index - 1, N_index - 3, N_index - 2)
            circ.toffoli( N_index - 3, N_index -7, N_index - 2)
            
            circ.toffoli( N_index - 0, N_index - 2, N_index -4)
            circ.toffoli( N_index - 0, N_index - 2, N_index -8)
            
            circ.toffoli( N_index - 1, N_index - 2, N_index - 3)
            circ.toffoli( N_index - 1, N_index - 2, N_index -8)
            
            # Applying C_inv_toffoli of apply_CinvToffoli(bitstring, 2, 0, 1, 3)
            
            circ.cnot(N_index - 2, N_index - 0)
            circ.cnot(N_index - 2, N_index - 1)
            circ.mcx([N_index - 2, N_index - 0, N_index - 1], N_index - 3)
            circ.cnot(N_index - 2, N_index - 1)
            circ.cnot(N_index - 2, N_index - 0)
            
            circ.cnot(N_index - 2, N_index - 0)
            circ.cnot(N_index - 2, N_index - 1)
            circ.mcx([N_index - 2, N_index - 0, N_index - 1], N_index - 7)
            circ.cnot(N_index - 2, N_index - 1)
            circ.cnot(N_index - 2, N_index - 0)
            
            circ.x(N_index - 2)
            
            circ.toffoli( N_index - 0, N_index - 2, N_index -5 )
            circ.toffoli( N_index - 0, N_index - 2, N_index -7 )
            
            circ.toffoli( N_index - 1, N_index - 2, N_index -5 )
            circ.toffoli( N_index - 1, N_index - 2, N_index -6 )
            
            circ.cnot(N_index - 2, N_index - 0)
            circ.cnot(N_index - 2, N_index - 1)
            circ.mcx([N_index - 2, N_index - 0, N_index - 1], N_index - 4)
            circ.cnot(N_index - 2, N_index - 1)
            circ.cnot(N_index - 2, N_index - 0)
            
            
            circ.cnot(N_index - 2, N_index - 0)
            circ.cnot(N_index - 2, N_index - 1)
            circ.mcx([N_index - 2, N_index - 0, N_index - 1], N_index - 6)
            circ.cnot(N_index - 2, N_index - 1)
            circ.cnot(N_index - 2, N_index - 0)
            
            circ.x(N_index - 2)
            
            
            # Swapping the Least Sig. Qubit with MSQ
            for i in range(n):
                circ.swap((N_index - 2)+ i, i)
    
        elif(n == 4):
            circ.x(N_index - 0)
            circ.x(N_index - 1)
            circ.x(N_index - 2)
            circ.mcx([N_index - 0, N_index - 1, N_index - 2], N_index - 3)
            circ.x(N_index - 2)
            circ.x(N_index - 1)
            circ.x(N_index - 0) # reset 3rd qubit, it will hold 1>2?
            
            circ.toffoli(N_index - 4, N_index - 9, N_index - 3)
            circ.toffoli(N_index - 4, N_index - 10, N_index - 3)
            circ.toffoli(N_index - 4, N_index - 11, N_index - 3)
            circ.toffoli(N_index - 5, N_index - 10, N_index - 3)
            circ.toffoli(N_index - 5, N_index - 11, N_index - 3)
            circ.toffoli(N_index - 6, N_index - 11, N_index - 3) # 3rd qubit= 1>2 ? 
            
            circ.x(N_index - 8)
            circ.x(N_index - 9)
            circ.x(N_index - 10)
            circ.mcx([N_index - 8, N_index - 9, N_index - 10], N_index - 11)
            circ.x(N_index - 8)
            circ.x(N_index - 9)
            circ.x(N_index - 10) # reset 11th qubit, it will hold 1>3 ?
            
            circ.toffoli(N_index - 4, N_index - 13, N_index - 11)
            circ.toffoli(N_index - 4, N_index - 14, N_index - 11)
            circ.toffoli(N_index - 4, N_index - 15, N_index - 11)
            circ.toffoli(N_index - 5, N_index - 14, N_index - 11)
            circ.toffoli(N_index - 5, N_index - 15, N_index - 11)
            circ.toffoli(N_index - 6, N_index - 15, N_index - 11) # 11th qubit= 1>3 ? (later to be swapped with 4th)
            
            circ.x(N_index - 4)
            circ.x(N_index - 6)
            circ.x(N_index - 7)
            circ.mcx([N_index - 4, N_index - 6, N_index - 7], N_index - 5)
            circ.x(N_index - 4)
            circ.x(N_index - 6)
            circ.x(N_index - 7) # reset 5th qubit, it will hold 2>3 ?
            
            circ.toffoli(N_index - 8, N_index - 13, N_index - 5)
            circ.toffoli(N_index - 8, N_index - 14, N_index - 5)
            circ.toffoli(N_index - 8, N_index - 15, N_index - 5)
            circ.toffoli(N_index - 9, N_index - 14, N_index - 5)
            circ.toffoli(N_index - 9, N_index - 15, N_index - 5)
            circ.toffoli(N_index - 10, N_index - 15, N_index - 5) # 5th qubit= 2>3 ? 
            
            # encoding is complete, now we have to reset the rest of the qubits
            
            # 111 
            
            circ.mcx([N_index - 0, N_index - 3, N_index - 11, N_index - 5], N_index - 15)
            circ.mcx([N_index - 0, N_index - 3, N_index - 11, N_index - 5], N_index - 10)
            # circ.mcx([N_index - 0, N_index - 3, N_index - 11, N_index - 5], N_index - 15)
            
            circ.mcx([N_index - 1, N_index - 3, N_index - 11, N_index - 5], N_index - 15)
            circ.mcx([N_index - 1, N_index - 3, N_index - 11, N_index - 5], N_index - 10)
            circ.mcx([N_index - 1, N_index - 3, N_index - 11, N_index - 5], N_index - 4)
            
            circ.mcx([N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 15)
            circ.mcx([N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 9)
            circ.mcx([N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 4)
            
            #last step of 111
            circ.x(N_index - 0)
            circ.x(N_index - 1)
            circ.x(N_index - 2)
            circ.mcx([N_index - 0, N_index - 1, N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 14)
            circ.mcx([N_index - 0, N_index - 1, N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 9)
            circ.mcx([N_index - 0, N_index - 1, N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 4)
            circ.x(N_index - 2)
            circ.x(N_index - 1)
            circ.x(N_index - 0)
            
            
            # reverse 3 , 011
            circ.x(N_index - 3)
            
            circ.mcx([N_index - 0, N_index - 3, N_index - 11, N_index - 5], N_index - 15)
            circ.mcx([N_index - 0, N_index - 3, N_index - 11, N_index - 5], N_index - 9)
            circ.mcx([N_index - 0, N_index - 3, N_index - 11, N_index - 5], N_index - 6)
            
            circ.mcx([N_index - 1, N_index - 3, N_index - 11, N_index - 5], N_index - 15)
            circ.mcx([N_index - 1, N_index - 3, N_index - 11, N_index - 5], N_index - 8)
            circ.mcx([N_index - 1, N_index - 3, N_index - 11, N_index - 5], N_index - 6)
            
            circ.mcx([N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 15)
            circ.mcx([N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 8)
            # circ.mcx([N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 4)
            
            
            #last step of 011
            circ.x(N_index - 0)
            circ.x(N_index - 1)
            circ.x(N_index - 2)
            circ.mcx([N_index - 0, N_index - 1, N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 14)
            circ.mcx([N_index - 0, N_index - 1, N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 8)
            # circ.mcx([N_index - 0, N_index - 1, N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 4)
            circ.x(N_index - 2)
            circ.x(N_index - 1)
            circ.x(N_index - 0)
            
            
            #reverse 11, 001
            circ.x(N_index - 11)

            circ.mcx([N_index - 0, N_index - 3, N_index - 11, N_index - 5], N_index - 14)
            circ.mcx([N_index - 0, N_index - 3, N_index - 11, N_index - 5], N_index - 9)
            circ.mcx([N_index - 0, N_index - 3, N_index - 11, N_index - 5], N_index - 7)
            
            circ.mcx([N_index - 1, N_index - 3, N_index - 11, N_index - 5], N_index - 14)
            circ.mcx([N_index - 1, N_index - 3, N_index - 11, N_index - 5], N_index - 8)
            circ.mcx([N_index - 1, N_index - 3, N_index - 11, N_index - 5], N_index - 7)
            
            circ.mcx([N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 13)
            circ.mcx([N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 8)
            circ.mcx([N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 7)
            
            #last step of 001
            circ.x(N_index - 0)
            circ.x(N_index - 1)
            circ.x(N_index - 2)
            circ.mcx([N_index - 0, N_index - 1, N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 13)
            circ.mcx([N_index - 0, N_index - 1, N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 8)
            circ.mcx([N_index - 0, N_index - 1, N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 6)
            circ.x(N_index - 2)
            circ.x(N_index - 1)
            circ.x(N_index - 0)
            
            
            #reverse 5 000
            circ.x(N_index - 5)
            
            circ.mcx([N_index - 0, N_index - 3, N_index - 11, N_index - 5], N_index - 13)
            circ.mcx([N_index - 0, N_index - 3, N_index - 11, N_index - 5], N_index - 10)
            circ.mcx([N_index - 0, N_index - 3, N_index - 11, N_index - 5], N_index - 7)
            
            circ.mcx([N_index - 1, N_index - 3, N_index - 11, N_index - 5], N_index - 12)
            circ.mcx([N_index - 1, N_index - 3, N_index - 11, N_index - 5], N_index - 10)
            circ.mcx([N_index - 1, N_index - 3, N_index - 11, N_index - 5], N_index - 7)
            
            circ.mcx([N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 12)
            circ.mcx([N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 9)
            circ.mcx([N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 7)
            
            #last step of 000
            circ.x(N_index - 0)
            circ.x(N_index - 1)
            circ.x(N_index - 2)
            circ.mcx([N_index - 0, N_index - 1, N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 12)
            circ.mcx([N_index - 0, N_index - 1, N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 9)
            circ.mcx([N_index - 0, N_index - 1, N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 6)
            circ.x(N_index - 2)
            circ.x(N_index - 1)
            
            
            
            # unreverse 3, 100
            circ.x(N_index - 3)
            
            circ.mcx([N_index - 0, N_index - 3, N_index - 11, N_index - 5], N_index - 13)
            # circ.mcx([N_index - 0, N_index - 3, N_index - 11, N_index - 5], N_index - 10)
            circ.mcx([N_index - 0, N_index - 3, N_index - 11, N_index - 5], N_index - 6)
            
            circ.mcx([N_index - 1, N_index - 3, N_index - 11, N_index - 5], N_index - 12)
            # circ.mcx([N_index - 1, N_index - 3, N_index - 11, N_index - 5], N_index - 10)
            circ.mcx([N_index - 1, N_index - 3, N_index - 11, N_index - 5], N_index - 6)
            
            circ.mcx([N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 12)
            # circ.mcx([N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 9)
            # circ.mcx([N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 7)
            
            #last step of 100
            circ.x(N_index - 0)
            circ.x(N_index - 1)
            circ.x(N_index - 2)
            circ.mcx([N_index - 0, N_index - 1, N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 12)
            circ.mcx([N_index - 0, N_index - 1, N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 10)
            # circ.mcx([N_index - 0, N_index - 1, N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 6)
            circ.x(N_index - 2)
            circ.x(N_index - 1)
            
            
            # unreverse 11, 110
            circ.x(N_index - 11)
            
            circ.mcx([N_index - 0, N_index - 3, N_index - 11, N_index - 5], N_index - 14)
            # circ.mcx([N_index - 0, N_index - 3, N_index - 11, N_index - 5], N_index - 10)
            # circ.mcx([N_index - 0, N_index - 3, N_index - 11, N_index - 5], N_index - 7)
            
            circ.mcx([N_index - 1, N_index - 3, N_index - 11, N_index - 5], N_index - 14)
            # circ.mcx([N_index - 1, N_index - 3, N_index - 11, N_index - 5], N_index - 10)
            circ.mcx([N_index - 1, N_index - 3, N_index - 11, N_index - 5], N_index - 4)
            
            circ.mcx([N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 13)
            # circ.mcx([N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 9)
            circ.mcx([N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 4)


            #last step of 110
            circ.x(N_index - 0)
            circ.x(N_index - 1)
            circ.x(N_index - 2)
            circ.mcx([N_index - 0, N_index - 1, N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 13)
            circ.mcx([N_index - 0, N_index - 1, N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 10)
            circ.mcx([N_index - 0, N_index - 1, N_index - 2, N_index - 3, N_index - 11, N_index - 5], N_index - 4)
            circ.x(N_index - 2)
            circ.x(N_index - 1)


            # unreverse 5
            circ.x(N_index - 5)
            # 101 and 010 are not valid since 1>2 & 2>3 & 3>1 is not a valid logical assignment
            
            circ.swap(N_index - 11,N_index - 4)
            
            
            
            
    def discardQubits(n = 3):
        if(n == 2):
            circ.reset(2)
            circ.reset(1)
            circ.reset(0)
        if(n==3):
            for i in range(ancillas):
                circ.swap(N + i, N_index - i)
        
    def decoderCircuit(n = 3):
        if(n == 2):
            circ.x(3).inverse
            circ.cnot(3,2).inverse
            circ.cnot(3,1).inverse
            
            circ.x(3).inverse
            circ.cnot(3,0).inverse
            
        if(n == 3):
            
            for i in range(n):
                circ.swap((N_index - 2)+ i, i).inverse()
            
            circ.x(N_index - 2).inverse
            
            circ.cnot(N_index - 2, N_index - 0).inverse
            circ.cnot(N_index - 2, N_index - 1).inverse
            circ.mcx([N_index - 2, N_index - 0, N_index - 1], N_index - 6).inverse
            circ.cnot(N_index - 2, N_index - 1).inverse
            circ.cnot(N_index - 2, N_index - 0).inverse
            
            
            circ.cnot(N_index - 2, N_index - 0).inverse
            circ.cnot(N_index - 2, N_index - 1).inverse
            circ.mcx([N_index - 2, N_index - 0, N_index - 1], N_index - 4).inverse
            circ.cnot(N_index - 2, N_index - 1).inverse
            circ.cnot(N_index - 2, N_index - 0).inverse
            
            circ.toffoli(N_index - 1, N_index - 2, N_index - 6 ).inverse
            circ.toffoli(N_index - 1, N_index - 2, N_index - 5 ).inverse
            
            circ.toffoli(N_index - 0, N_index - 2, N_index - 7 ).inverse
            circ.toffoli(N_index - 0, N_index - 2, N_index - 5 ).inverse
            
            circ.x(N_index - 2).inverse
            
            circ.cnot(N_index - 2, N_index - 0).inverse
            circ.cnot(N_index - 2, N_index - 1).inverse
            circ.mcx([N_index - 2, N_index - 0, N_index - 1], N_index - 7).inverse
            circ.cnot(N_index - 2, N_index - 1).inverse
            circ.cnot(N_index - 2, N_index - 0).inverse
            
            circ.cnot(N_index - 2, N_index - 0).inverse
            circ.cnot(N_index - 2, N_index - 1).inverse
            circ.mcx([N_index - 2, N_index - 0, N_index - 1], N_index - 3).inverse
            circ.cnot(N_index - 2, N_index - 1).inverse
            circ.cnot(N_index - 2, N_index - 0).inverse
            

            circ.toffoli(N_index - 1, N_index - 2, N_index - 8).inverse
            circ.toffoli(N_index - 1, N_index - 2, N_index - 3).inverse
            circ.toffoli(N_index - 0, N_index - 2, N_index - 8).inverse
            circ.toffoli(N_index - 0, N_index - 2, N_index - 4).inverse
            
            circ.toffoli( N_index - 3, N_index - 7, N_index - 2).inverse
            circ.toffoli( N_index - 1, N_index - 3, N_index - 2).inverse
            circ.toffoli( N_index - 0, N_index - 4, N_index - 2).inverse
            
            circ.x(N_index - 1).inverse
            circ.x(N_index - 0).inverse
            
            
            circ.toffoli( N_index - 0, N_index - 1, N_index - 2 ).inverse
            
            circ.x( N_index - 1 ).inverse
            circ.x( N_index - 0 ).inverse
       
            
    initializePlusState()
    
    for l in range(layers):
        costHamiltonian(l)
        mixerHamiltonian(l)
    encoderCircuit(n)
    if(decode):
        discardQubits(n)
        # decoderCircuit(n)
        
    if(decode == False):
        circ.measure(q[:3],c)
    else:
        circ.measure(q[:3],c[:3])
    
    return circ

# %% Loss function and other utils

def majorityVote(result):
    # Get the majority vote as the output of the circuit
    max_count = 0
    out = None
    for _ , (res, count )in enumerate(result.get_counts().items()):
        if(res == '110000110' or res == '111110000'):
            continue
        if(count > max_count):
            out = res
            max_count = count
    if(out == None):
        raise('error in measurement')
    return out


#one-hot-vectors and their embeddings
graph_dict = {'001010100': '000000000', 
               '001100010': '000000001',
               '010001100': '000000010',
               '010100001': '000000011',
               '100001010': '000000100',
               '100010001': '000000101'} 
    
# false positives: '110000110' -> gets encoded as '110' and 
#                   111110000 -> gets encoded as '111'


iteration_count = 0
out = 0

def lossFunction(counts):
    global cost_dict
    global iteration_count
    
    iteration_count += 1
    
    loss = 0
    count_total = 0
    for idx, (keys, vals) in enumerate(counts.items()):
        
        count_total += vals
        loss += (cost_dict[keys] * vals)
    loss /= count_total
    
    if(iteration_count % 10 == 0):
        print(loss)
    
    return loss 

def createPathCosts(graph3_dict, adj, n = 3):
    cost = np.array([100,100,100,100,100,100,100,100])
    cost_dict = {'111': 100,
     '101': 52,
     '001': 4,
     '011': 52,
     '100': 4,
     '000': 52,
     '010': 52,
     '110': 100}
    for idx, (keys, vals) in enumerate(graph3_dict.items()):
        path = findPath(keys)
        loss = adj[path[0], path[1]] + adj[path[1], path[2]]
        cost[idx] = loss
    
    
    return cost, cost_dict

cost, cost_dict = createPathCosts(graph_dict, adj, n = 3)


# %% 
loss_history = []
running_loss_history = []
running_loss = 0
run_index= 0
def getExpVal(edges,adj):
    backend = qiskit.Aer.get_backend('qasm_simulator')
    backend.shots = 1024
    def execute_circ(params):
        global out
        global loss_history

        circuit = QAOA(edges, adj, params, decode = True)
        counts = backend.run(circuit, seed_simulator = 10, shots = 1024).result()
        out = deepcopy(counts)
        loss_val = lossFunction(counts.get_counts())
        loss_history.append(loss_val)

        return loss_val
        
        
    return execute_circ

init_params = np.random.rand(6)*np.pi / 10
exp = getExpVal(edges, adj)
print(exp(init_params))
# %% Optimization
start = time.time()
optimizer = SPSA(maxiter=1000)
optimized_parameters, final_cost, number_of_circuit_calls = optimizer.optimize(6, exp, initial_point = init_params )
end = time.time()

print('Optimization terminated within {:.3f} seconds'.format(end-start))
print('The final cost: {}'.format(exp(optimized_parameters)))
print(out)
# %% 

circuit = QAOA(edges, adj, init_params, decode = True)
backend = qiskit.Aer.get_backend('qasm_simulator')
counts = backend.run(circuit, seed_simulator = 10, shots = 1024).result()
counts.get_counts()


exp = getExpVal(edges, adj)
exp(optimized_parameters)

running_loss= 0
for idx, l in enumerate(loss_history):
    running_loss += l
    
    running_loss_history.append(running_loss / (idx + 1))

from matplotlib import pyplot as plt

plt.plot(running_loss_history)
# %% Unit Tests

# sys.argv = ['Tests\\unitTest.py','n=3']
# execfile('Tests\\unitTest.py')

# %% 

# for i in range(10):
#     random_idx = np.random.randint(512)
#     initial_state_out = np.zeros((2**N), dtype = np.complex128)
#     initial_state_out[random_idx] = 1 # or the statevector for one of the feasible solutions
    
#     start = time.time()
    
#     circ = QAOA(edges, adj,initial_state = initial_state_out, decode = True)
#     job = backend.run(qiskit.transpile(circ, backend))
#     result = job.result()
    
#     end = time.time()

#     print('{}th Circuit run within {:.3f} seconds'.format(i ,end-start))
#     # print('Expected Output: {}'.format(vals))
    
#     out = majorityVote(result)
#     print('Output: {}'.format(out))    
#     print('****')


# betas = np.random.rand(3)
# for idx, (keys, vals) in enumerate(graph_dict.items()):
    
#     state =  bit2int(keys)
    
#     initial_state_out = np.zeros((2**N), dtype = np.complex128)
#     initial_state_out[state] = 1 # or the statevector for one of the feasible solutions
    
#     start = time.time()
    
#     # circ = QAOA(edges, adj,initial_state = initial_state_out, decode = True)
#     circ = QAOA(edges, adj, betas)
#     job = backend.run(qiskit.transpile(circ, backend))
#     result = job.result()
    
#     end = time.time()

#     print('{}th Circuit run within {:.3f} seconds'.format(idx ,end-start))
#     print('Expected Output: {}'.format(vals))
    
#     out = majorityVote(result)
#     print('Output: {}'.format(out))    
#     print('****')

