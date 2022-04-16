# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 17:44:00 2022

@author: burak
"""

import pennylane as qml
import numpy as np
import networkx as nx
from scipy.linalg import expm, logm
import time
import qiskit

n = 3
N = n**2
z = np.array([[1,0] , [0j, -1]])
y = np.array([[0,-1j] , [1j, 0]])
x = np.array([[0,1] , [0j + 1, 0]])
I = np.eye(2)

Splus = np.array([[0j,0], [1,0]])
Splus = x + 1j*y
Sminus = x - 1j*y
Sminus = np.array([[0j,1], [0,0]])



createBitString = lambda x,y=N: str(bin(x)[2:].zfill(y))
int2bit = lambda x: str(bin(x)[2:].zfill(N))
bit2int = lambda b: int("".join(str(bs) for bs in b), base = 2)
quantumOuter = lambda i1,i2: np.outer(i1, i2.conj().T)

def commuteCheck(A,B):
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
    out = np.array([1])
    # idx = 0
    for i in range(N):
        if(i in listOfQubits):
            idx = listOfQubits.index(i)
            out = np.kron(out, paulis[idx])
        else:
            out = np.kron(out, I)
    return out


def findGroundStates(H, size = N):
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



def edgeCount(self, solution, G):
  edge_count = 0
  edges = G.edges()
  for edge in edges:
    edge_1, edge_2 = edge
    if(solution[edge_1] != solution[edge_2]):
      edge_count += 1
  return edge_count * -1    

# %% Graph
n = 3
N = n**2



G = nx.Graph()

for i in range(n):
    G.add_node(i)


G.add_edge(0,1, weight  = 4)
G.add_edge(0,2, weight  = 2)
# G.add_edge(1,2, weight  = 2)




nx.draw(G, with_labels=True, alpha=0.8)
adj = nx.adjacency_matrix(G).todense()
edges = list(G.edges)


# %% 

n = 3
N = n**2
backend = qiskit.BasicAer.get_backend('qasm_simulator')
unitary_backend = qiskit.BasicAer.get_backend('unitary_simulator')


def QAOA(edges, adj):
    '''
    Building the QAOA Ansatz, that includes U_Hc and encoding scheme

    Returns
    -------
    circ : qiskit circuit

    '''
    global n, N
    

    c = qiskit.ClassicalRegister(N) # stores the measurements
    q = qiskit.QuantumRegister(N)
    circ = qiskit.QuantumCircuit(q,c)
    b = np.ones((2**N)) / (np.sqrt(2) **N )
    circ.initialize(b , [q[i] for i in range(N)])
    beta = np.pi / 18 # initial parameter for HC
    
    N_index = N - 1
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
    
    # Encoder
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
            
            # Applying C_inv_toffoli
            #bitstring = apply_CinvToffoli(bitstring, 2, 0, 1, 3)
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
            
            '''
            bitstring = apply_X(bitstring, 0)
            bitstring = apply_X(bitstring, 1)
            bitstring = apply_Toffoli(bitstring, 0, 1, 2) # third qubit always set to 0
            bitstring = apply_X(bitstring, 0)
            bitstring = apply_X(bitstring, 1)  # first and second qubit always stay the same
            
            
            # last bit is determined by string from qubits 4 to 9
            # if binary number from 4-6 > 7-9, then last bit 1, otherwise it's 0
            # we must consider each case individually
            
            bitstring = apply_Toffoli(bitstring, 0, 4, 2)
            bitstring = apply_Toffoli(bitstring, 1, 3, 2)
            bitstring = apply_Toffoli(bitstring, 3, 7, 2)  # now third qubit tells us if 4-6 > 7-9
            
            
            # all qubits from 4 to 9 must be set to 0
            
            bitstring = apply_Toffoli(bitstring, 0, 2, 4)
            bitstring = apply_Toffoli(bitstring, 0, 2, 8)
            
            
            bitstring = apply_Toffoli(bitstring, 1, 2, 3)
            bitstring = apply_Toffoli(bitstring, 1, 2, 8)
            
            bitstring = apply_CinvToffoli(bitstring, 2, 0, 1, 3)
            bitstring = apply_CinvToffoli(bitstring, 2, 0, 1, 7) 
            
            bitstring = apply_X(bitstring, 2)
            
            bitstring = apply_Toffoli(bitstring, 0, 2, 5)
            bitstring = apply_Toffoli(bitstring, 0, 2, 7)
            
            bitstring = apply_Toffoli(bitstring, 1, 2, 5)
            bitstring = apply_Toffoli(bitstring, 1, 2, 6)
            
            bitstring = apply_CinvToffoli(bitstring, 2, 0, 1, 4)
            bitstring = apply_CinvToffoli(bitstring, 2, 0, 1, 6) 
            
            bitstring = apply_X(bitstring, 2)
            '''
    encoderCircuit(n)
    
    def discardQubits(n):
        #TODO: to be changed with ancilla qubits 
        if(n == 2):
            circ.reset(2)
            circ.reset(1)
            circ.reset(0)
        if(n==3):
            circ.reset(8)
            circ.reset(7)
            circ.reset(6)
            circ.reset(5)
            circ.reset(4)
            circ.reset(3)
    
    discardQubits(n)
    
    def decoderCircuit(n):
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
            
            
            
    decoderCircuit(n)

    
    
    circ.measure(q,c)
    
    
    return circ
circ = QAOA(edges, adj)
job = backend.run(qiskit.transpile(circ, backend))
result = job.result()
result.get_counts()
# %% 
N = 9

c = qiskit.ClassicalRegister(N) # stores the measurements
q = qiskit.QuantumRegister(N)
circ = qiskit.QuantumCircuit(q,c)

beta = np.pi / 18 # initial parameter for HC

psi = np.ones(2**N) /  2**((N/2))

circ.initialize(psi, [q[0], q[1], q[2] ,q[3]])

circ.cnot(3,0)
circ.x(3)
circ.cnot(3,2)
circ.cnot(3,1)
circ.x(3)


circ.reset(2)
circ.reset(1)
circ.reset(0)
circ.x(3).inverse
circ.cnot(3,2).inverse
circ.cnot(3,1).inverse

circ.x(3).inverse
circ.cnot(3,0).inverse
    
circ.measure(q,c)

job = backend.run(qiskit.transpile(circ, backend))
result = job.result()
result.get_counts()

for edge in edges:
    u, v = edge
    for step in range(n - 1):
        q_1 = (u * n) + step
        q_2 = (v * n) + step + 1 # (+1 because we want that node visited in the next step)
        
        q_1_rev = (u * n) + step + 1 # (reverse path from above)
        q_2_rev = (v * n) + step  
        
        circ.rzz(2 * beta, q_1 , q_2 ) # TODO: Change with edge weights
        circ.rzz(2 * beta, q_1_rev, q_2_rev)
circ.measure(q,c)

H = 0
for edge in edges:
    u, v = edge
    for step in range(n - 1):
        q_1 = (u * n) + step
        q_2 = (v * n) + step + 1 # (+1 because we want that node visited in the next step)
        
        q_1_rev = (u * n) + step + 1 # (reverse path from above)
        q_2_rev = (v * n) + step  
        
        z1 = KroneckerProductString([q_1, q_2], [z,z], N)
        z2 = KroneckerProductString([q_1_rev, q_2_rev], [z,z], N)
        
        H += (z1 + z2)

job = backend.run(qiskit.transpile(circ, backend))
result = job.result()
result.get_counts()


# qiskit output: 1000 means |0001>

unitary_job = unitary_backend.run(qiskit.transpile(circ, unitary_backend))
unitary_result = unitary_job.result()
U = unitary_result.get_unitary(circ, decimals=3)


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



print(TestCase2())
print(TestCase(edges,adj)[0])


# %% 
unitary_backend = qiskit.BasicAer.get_backend('unitary_simulator')

c = qiskit.ClassicalRegister(9) # stores the measurements
q = qiskit.QuantumRegister(9)
circ = qiskit.QuantumCircuit(q,c)

circ.rzz(2 * beta , 8 , 4)
circ.rzz(2 * beta , 1,  5)
circ.rzz(2 * beta , 2,  4)
circ.rzz(2 * beta , 1 , 3)

a = expm(-1j * beta * 
         (  (KroneckerProductString([0,4], [z,z], 9) )  + (KroneckerProductString([7,3], [z,z], 9) )  + (KroneckerProductString([6,4], [z,z], 9) )  + (KroneckerProductString([7,5], [z,z], 9) ) 
                         ))
unitary_job = unitary_backend.run(qiskit.transpile(circ, unitary_backend))
unitary_result = unitary_job.result()
U = unitary_result.get_unitary(circ, decimals=3)
np.allclose(U,a, 10e-4)
# %% 


commuteCheck( expm(-1j * beta * KroneckerProductString([0,1], [z,z], 4) * np.sqrt(11)/ 4) , expm(-1j * beta * KroneckerProductString([1,3], [z,z], 4) * (np.sqrt(5)/4 )) )

commuteCheck(expm(-1j * beta * KroneckerProductString([0,1], [z,z], 4)) , expm(-1j * beta * KroneckerProductString([2,3], [z,z], 4)  )) 

U = expm(-1j * beta * KroneckerProductString([2,3], [z,z], 4) 
U2 = expm(-1j * beta * KroneckerProductString([0,1], [z,z], 4)) 
commuteCheck(KroneckerProductString([0,1], [z,z], 4)  , KroneckerProductString([1,3], [z,z], 4)  )
commuteCheck()
H = 0
for edge in edges:
    u, v = edge
    for step in range(n - 1):
        q_1 = (u * n) + step
        q_2 = (v * n) + step + 1 # (+1 because we want that node visited in the next step)
        
        q_1_rev = (u * n) + step + 1 # (reverse path from above)
        q_2_rev = (v * n) + step  
        
        z1 = KroneckerProductString([q_1, q_2], [z,z], N)
        z2 = KroneckerProductString([q_1_rev, q_2_rev], [z,z], N)
        
        H += (z1 + z2)

# qiskit output: 1000 means |0001>

# %%  C_invToffoli

qml.MultiControlledX([0,1], [2], '00').matrix

qml.Toffoli(wires = [0,1,2]).matrix
CinvToffoli = np.eye(16)

unitary_backend2 = qiskit.BasicAer.get_backend('unitary_simulator')

c2 = qiskit.ClassicalRegister(4) # stores the measurements
q2 = qiskit.QuantumRegister(4)
circ2 = qiskit.QuantumCircuit(q2,c2)

circ2.cnot(3,2)
circ2.cnot(3,1)
circ2.mcx([3,2,1], 0)
circ2.cnot(3,1)
circ2.cnot(3,2)

unitary_job2 = unitary_backend2.run(qiskit.transpile(circ2, unitary_backend2))
unitary_result2 = unitary_job2.result()
U2 = unitary_result2.get_unitary(circ2, decimals=3)
