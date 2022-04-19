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

def QAOA(edges, adj, beta = np.array([np.pi/10, np.pi/10, np.pi/10]) , decode = True, layers = 3):
        
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
    c = qiskit.ClassicalRegister(N) # stores the measurements
    if(decode == False):
        c = qiskit.ClassicalRegister(3) # stores the measurements
    q = qiskit.QuantumRegister(N+ ancillas)
    circ = qiskit.QuantumCircuit(q,c)
    
    
    
    def initializer(initial_state):
        # b = np.ones((2**N)) / (np.sqrt(2) **N ) # equal superposition
        if(initial_state is None):
            # initial_state = np.zeros((2**N), dtype = np.complex128)
            # initial_state[266] = 1 # or the statevector for one of the feasible solutions
            initial_state = np.ones((2**N)) / (np.sqrt(2) **N ) # equal superposition
            circ.initialize(initial_state , [q[i] for i in range(N)])
        else:
            if(stateVectorCheck(initial_state) == False):
                raise('state vector is not valid')
            
            circ.initialize(initial_state , [q[i] for i in range(N)])
    
    
    
    
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
                circ.rzz(2 * beta[l] * (adj[u,v]) , q_1_rev, q_2_rev)
                # exponential of the cost Hamiltonian
    def mixerHamiltonian():
        for i in range(N):
            circ.rx( np.pi/30 , i)
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
            
            
            # Swapping the Least Sig. Qubit with MSQ
            for i in range(n):
                circ.swap((N_index - 2)+ i, i)
    
    def discardQubits(n = 3):
        #TODO: to be changed with ancilla qubits 
        if(n == 2):
            circ.reset(2)
            circ.reset(1)
            circ.reset(0)
        if(n==3):
            # Takes too much time
            
            
            for i in range(ancillas):
                circ.swap(N + i, N_index - i)
            # circ.reset(N - 8)
            # circ.reset(N - 7)
            # circ.reset(N - 6)
            # circ.reset(N - 5)
            # circ.reset(N - 4)
            # circ.reset(N - 3)
        
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
       
            
    initializer(initial_state)
    for l in range(layers):
        
        costHamiltonian(l)
        encoderCircuit(n)
        if(decode):
            discardQubits(n)
            decoderCircuit(n)
        
    if(decode == False):
        circ.measure(q[:3],c)
    else:
        circ.measure(q[:N],c[:N])
    
    return circ

# %% 

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

betas = np.random.rand(3)
for idx, (keys, vals) in enumerate(graph_dict.items()):
    
    state =  bit2int(keys)
    
    initial_state_out = np.zeros((2**N), dtype = np.complex128)
    initial_state_out[state] = 1 # or the statevector for one of the feasible solutions
    
    start = time.time()
    
    # circ = QAOA(edges, adj,initial_state = initial_state_out, decode = True)
    circ = QAOA(edges, adj, betas)
    job = backend.run(qiskit.transpile(circ, backend))
    result = job.result()
    
    end = time.time()

    print('{}th Circuit run within {:.3f} seconds'.format(idx ,end-start))
    print('Expected Output: {}'.format(vals))
    
    out = majorityVote(result)
    print('Output: {}'.format(out))    
    print('****')
    
    
# false positives: '110000110' -> gets encoded as '110' and 
#                   111110000 -> gets encoded as '111'
# %% 

for i in range(10):
    random_idx = np.random.randint(512)
    initial_state_out = np.zeros((2**N), dtype = np.complex128)
    initial_state_out[random_idx] = 1 # or the statevector for one of the feasible solutions
    
    start = time.time()
    
    circ = QAOA(edges, adj,initial_state = initial_state_out, decode = True)
    job = backend.run(qiskit.transpile(circ, backend))
    result = job.result()
    
    end = time.time()

    print('{}th Circuit run within {:.3f} seconds'.format(i ,end-start))
    # print('Expected Output: {}'.format(vals))
    
    out = majorityVote(result)
    print('Output: {}'.format(out))    
    print('****')

# %% Loss function
from qiskit.algorithms.optimizers import SPSA

def lossFunction(out):
    constraint_check = checkConstraints(out)
    penalty_term = 100000
    if(constraint_check == False):
        return penalty_term
    path = findPath(out)
    loss = adj[path[0], path[1]] * adj[path[1], path[2]]
    return loss
        
            

def getExpVal(edges,adj):
  backend = qiskit.Aer.get_backend('qasm_simulator')
  backend.shots = 8192
  def execute_circ(beta):
    circuit = QAOA(edges, adj)
    counts = backend.run(circuit, seed_simulator = 10, shots = 8192).result()
    counts = majorityVote(counts)
    
    return lossFunction(counts)
  return execute_circ

exp = getExpVal(edges, adj)
exp(betas)

start = time.time()
optimizer = SPSA(maxiter=1000)
expVal = getExpVal(edges, adj)
optimized_parameters, final_cost, number_of_circuit_calls = optimizer.optimize(8, expVal, initial_point=np.random.rand(3) * np.pi / 10 )
end = time.time()
print('Optimization terminated within {:.3f} seconds' , end-start)
# %% Unit Tests

sys.argv = ['Tests\\unitTest.py','n=3']
execfile('Tests\\unitTest.py')

