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

# %% Problem Parameters

n = 3  # number of nodes in the graph
N = n**2 # number of qubits to represent the one-hot-encoding of TSP

G = nx.Graph()

for i in range(n):
    G.add_node(i)

G.add_edge(0,1, weight  = 4)
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

def QAOA(edges, adj):
        
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
    
    c = qiskit.ClassicalRegister(N) # stores the measurements
    q = qiskit.QuantumRegister(N)
    circ = qiskit.QuantumCircuit(q,c)
    
    
    
    def initializer():
        # b = np.ones((2**N)) / (np.sqrt(2) **N ) # equal superposition
        b = np.zeros((2**N))
        b[266] = 1 # or the statevector for one of the feasible solutions
        circ.initialize(b , [q[i] for i in range(N)])
    
    
    
    
    beta = np.pi / 18 # initial parameter for HC
    
    N_index = N - 1 #since qiskit has a different ordering (LSB for the first qubit), we subtract the index from this number
    
    def costHamiltonian():
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
                
                circ.rzz(2 * beta * (adj[u,v]) , q_1 , q_2)
                circ.rzz(2 * beta * (adj[u,v]) , q_1_rev, q_2_rev)
                # exponential of the cost Hamiltonian
    
    # Encoder Circuit
    def encoderCircuit(n):
        if(n == 2):
            circ.cnot(3,0)
            circ.x(3)
            circ.cnot(3,2)
            circ.cnot(3,1)
            circ.x(3)
        else:
            
            
            circ.x( N_index -0 )
            circ.x( N_index -1 )
            circ.toffoli( N_index -0, N_index -1, N_index -2 )
            circ.x( N_index -0 )
            circ.x( N_index -1 )
            circ.toffoli( N_index -0, N_index -4, N_index -2 )
            circ.toffoli( N_index -1, N_index -3, N_index -2 )
            circ.toffoli( N_index -3, N_index -7, N_index -2 )
            
            circ.toffoli( N_index -0, N_index -2, N_index -4 )
            circ.toffoli( N_index -0, N_index -2, N_index -8 )
            
            circ.toffoli( N_index -1, N_index -2, N_index -3 )
            circ.toffoli( N_index -1, N_index -2, N_index -8 )
            
            # Applying C_inv_toffoli of apply_CinvToffoli(bitstring, 2, 0, 1, 3)
            
            circ.cnot(N_index - 2,N_index - 0)
            circ.cnot(N_index - 2,N_index - 1)
            circ.mcx([N_index - 2,N_index - 0,N_index - 1], N_index - 3)
            circ.cnot(N_index - 2,N_index - 1)
            circ.cnot(N_index - 2,N_index - 0)
            
            circ.cnot(N_index - 2,N_index - 0)
            circ.cnot(N_index - 2,N_index - 1)
            circ.mcx([N_index - 2,N_index - 0,N_index - 1], N_index - 7)
            circ.cnot(N_index - 2,N_index - 1)
            circ.cnot(N_index - 2,N_index - 0)
            
            circ.x(N_index - 2)
            
            circ.toffoli( N_index -0, N_index -2, N_index -5 )
            circ.toffoli( N_index -0, N_index -2, N_index -7 )
            
            circ.toffoli( N_index -1, N_index -2, N_index -5 )
            circ.toffoli( N_index -1, N_index -2, N_index -6 )
            
            circ.cnot(N_index - 2,N_index - 0)
            circ.cnot(N_index - 2,N_index - 1)
            circ.mcx([N_index - 2,N_index - 0,N_index - 1], N_index - 4)
            circ.cnot(N_index - 2,N_index - 1)
            circ.cnot(N_index - 2,N_index - 0)
            
            
            circ.cnot(N_index - 2,N_index - 0)
            circ.cnot(N_index - 2,N_index - 1)
            circ.mcx([N_index - 2,N_index - 0,N_index - 1], N_index - 6)
            circ.cnot(N_index - 2,N_index - 1)
            circ.cnot(N_index - 2,N_index - 0)
            
            circ.x(N_index - 2)

    
    def discardQubits(n = 3):
        #TODO: to be changed with ancilla qubits 
        if(n == 2):
            circ.reset(2)
            circ.reset(1)
            circ.reset(0)
        if(n==3):
            # Takes too much time
            circ.reset(N - 8)
            circ.reset(N - 7)
            circ.reset(N - 6)
            circ.reset(N - 5)
            circ.reset(N - 4)
            circ.reset(N - 3)
        
    def decoderCircuit(n = 3):
        if(n == 2):
            circ.x(3).inverse
            circ.cnot(3,2).inverse
            circ.cnot(3,1).inverse
            
            circ.x(3).inverse
            circ.cnot(3,0).inverse
            
        if(n == 3):
            circ.x(N_index - 2).inverse
            
            circ.cnot(N_index - 2,N_index - 0).inverse
            circ.cnot(N_index - 2,N_index - 1).inverse
            circ.mcx([N_index - 2,N_index - 0,N_index - 1], N_index - 6).inverse
            circ.cnot(N_index - 2,N_index - 1).inverse
            circ.cnot(N_index - 2,N_index - 0).inverse
            
            
            circ.cnot(N_index - 2,N_index - 0).inverse
            circ.cnot(N_index - 2,N_index - 1).inverse
            circ.mcx([N_index - 2,N_index - 0,N_index - 1], N_index - 4).inverse
            circ.cnot(N_index - 2,N_index - 1).inverse
            circ.cnot(N_index - 2,N_index - 0).inverse
            
            circ.toffoli( N_index -1, N_index -2, N_index -6 ).inverse
            circ.toffoli( N_index -1, N_index -2, N_index -5 ).inverse
            
            circ.toffoli( N_index -0, N_index -2, N_index -7 ).inverse
            circ.toffoli( N_index -0, N_index -2, N_index -5 ).inverse
            
            circ.x(N_index - 2).inverse
            
            circ.cnot(N_index - 2,N_index - 0).inverse
            circ.cnot(N_index - 2,N_index - 1).inverse
            circ.mcx([N_index - 2,N_index - 0,N_index - 1], N_index - 7).inverse
            circ.cnot(N_index - 2,N_index - 1).inverse
            circ.cnot(N_index - 2,N_index - 0).inverse
            
            circ.cnot(N_index - 2,N_index - 0).inverse
            circ.cnot(N_index - 2,N_index - 1).inverse
            circ.mcx([N_index - 2,N_index - 0,N_index - 1], N_index - 3).inverse
            circ.cnot(N_index - 2,N_index - 1).inverse
            circ.cnot(N_index - 2,N_index - 0).inverse
            

            circ.toffoli( N_index -1, N_index -2, N_index -8 ).inverse
            circ.toffoli( N_index -1, N_index -2, N_index -3 ).inverse
            circ.toffoli( N_index -0, N_index -2, N_index -8 ).inverse
            circ.toffoli( N_index -0, N_index -2, N_index -4 ).inverse
            
            circ.toffoli( N_index -3, N_index -7, N_index -2 ).inverse
            circ.toffoli( N_index -1, N_index -3, N_index -2 ).inverse
            circ.toffoli( N_index -0, N_index -4, N_index -2 ).inverse
            
            circ.x( N_index -1 ).inverse
            circ.x( N_index -0 ).inverse
            
            
            circ.toffoli( N_index -0, N_index -1, N_index -2 ).inverse
            
            circ.x( N_index -1 ).inverse
            circ.x( N_index -0 ).inverse
       
            
    initializer()
    costHamiltonian()
    encoderCircuit(n)
    # discardQubits(n)
    decoderCircuit(n)
    
    circ.measure(q,c)
    
    return circ


start = time.time()
circ = QAOA(edges, adj)


job = backend.run(qiskit.transpile(circ, backend))
result = job.result()
end = time.time()
print('Circuit run within {:.3f} seconds' , end-start)
print(result.get_counts())

# %% Unit Tests




def TestCase(edges, adj):
    '''
    Checks if the exponential of the problem hamiltonian, and the 
    unitary gate resulting from Ising ZZ Coupling gate are the same 
    
    
    p.s qubits have MSB ( most significant bit) order for the qubits
    
    Returns
    -------
    Bool

    '''
    
    beta = np.pi / 18 # initial parameter for HC
    
    def U_Circ(beta):
        unitary_backend = qiskit.BasicAer.get_backend('unitary_simulator')
    
        c = qiskit.ClassicalRegister(N) # stores the measurements
        q = qiskit.QuantumRegister(N)
        circ = qiskit.QuantumCircuit(q,c)
        # circ.initialize(b , [q[0], q[1], q[2] ,q[3]])
        
        
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
                
                
                
                circ.rzz(2 * beta * (adj[u,v]) , q_1 , q_2)
                circ.rzz(2 * beta * (adj[u,v]) , q_1_rev, q_2_rev)
                
                
                
        unitary_job = unitary_backend.run(qiskit.transpile(circ, unitary_backend))
        unitary_result = unitary_job.result()
        
        
        U = unitary_result.get_unitary(circ, decimals=3) # unitary of the qiskit circuit
        return U
    U = U_Circ
    
    H = 0
    for edge in edges:
        u, v = edge
        for step in range(n - 1):
            q_1 = (u * n) + step
            q_2 = (v * n) + step + 1 # (+1 because we want that node visited in the next step)
            
            q_1_rev = (u * n) + step + 1 # (reverse path from above)
            q_2_rev = (v * n) + step  
            
            z1 = KroneckerProductString([q_1, q_2], [z,z], N) * adj[u,v]
            z2 = KroneckerProductString([q_1_rev, q_2_rev], [z,z], N) * adj[u,v]
            # z1 =  KroneckerProductString([q_1], [z], N) * adj[u,v]
            
            H += ((z1 + z2))
            
    return np.allclose(expm(H * -1j * beta) , U(beta), 10e-2), U(beta), expm(H * -1j * beta)


def TestCase2():
    '''
    Checks if the encoding scheme works
    meaning that the resulting circuit only outputs statevectors
    which include a valid solution to the constrained optimization problem
    i.e , for TSP with 2 nodes, it outputs either |0110> or |1001>

    Returns
    -------
    bool

    '''
    
    
    N = 4
    c = qiskit.ClassicalRegister(N) # stores the measurements
    q = qiskit.QuantumRegister(N)
    circ = qiskit.QuantumCircuit(q,c)
    psi = np.ones(2**N) /  2**((N/2))
    
    circ.initialize(psi, [q[0], q[1], q[2] ,q[3]])
    
    #encoding
    circ.cnot(3,0)
    circ.x(3)
    circ.cnot(3,2)
    circ.cnot(3,1)
    circ.x(3)

    # discarding
    circ.reset(2)
    circ.reset(1)
    circ.reset(0)
    
    # decoding
    circ.x(3).inverse
    circ.cnot(3,2).inverse
    circ.cnot(3,1).inverse
    circ.x(3).inverse
    circ.cnot(3,0).inverse
        
    circ.measure(q,c)

    job = backend.run(qiskit.transpile(circ, backend))
    result = job.result()
    solution = result.get_counts()
    
    solution = list(solution.keys())
    def checkEncoding(solution):
        n = 2
        for sol in solution:
            for u in range(n):
                for v in range(u+1, n):
                    for step in range(n):
                        if(sol[u*n + step] == sol[v*n + step]):
                            return False
                        for step2 in range(step + 1, n):
                            if(sol[u*n + step] == sol[u*n + step2] or sol[v*n + step] == sol[v*n + step2]):
                                return False
        return True
    return checkEncoding(solution)



print('Result of the Test 1: {}'.format(TestCase(edges,adj)[0]))
print('Result of the Test 2: {}'.format( TestCase2()) )



