# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 00:37:47 2021

@author: burak
"""
from scipy.linalg import expm
import numpy as np
import qiskit
import sys


sys.path.append('..\\')
from QAutoencoder.Utils import KroneckerProductString, z
from QAutoencoder.QAOA_with_encoding import adj, edges



def TimeEvolutionTest(hamiltonian, t, states,dataset_size,ind):
    """ Finds the coefficients for the product state of pauli matrices that
    composes a hamiltonian, from a given hermitian

    Non-Keyword arguments:
        Hamiltonian- Hamiltonian of the system
        t- Time step for the time evolution
        states- Input state psi0
    
        
    Returns:
        Returns a boolean value whether the dataset includes data which are actually
        refers to the time evolved states of some inputs
    """
    
    return expm(hamiltonian[ind] * -1j * t) @ states[ind] == states[ind + dataset_size * t]




def TestCase(edges, adj,n = 3,):
    '''
    Checks if the exponential of the problem hamiltonian, and the 
    unitary gate resulting from Ising ZZ Coupling gate are the same 
    
    
    p.s qubits have MSB ( most significant bit) order for the qubits
    
    Returns
    -------
    Bool

    '''
    
    beta = np.pi / 18 # initial parameter for HC
    N = n**2
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
    backend = qiskit.BasicAer.get_backend('qasm_simulator')
    
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





def main(*args, **kwargs):
    for arg in args:
        
        n = arg.split("=")[0]
    if(type(edges) != list):
        raise('Wrong arguements passed')
    if(type(adj) != np.matrix):
        raise('Wrong arguements passed')
    if(len(edges) <= 0):
        raise('Missing edges')
    
    if(TestCase(edges,adj)[0] == True):
        print('Cost Hamiltonian is applied correctly as an Ising Coupling gate')
    else:
        raise('Value mismatch in the exponential of the Cost Hamiltonian')
    
    
    if(TestCase2() == True):
        print('Encoding is done correctly!')
    else:
        raise('Encoding failed')
        
if __name__ == '__main__':

    main(*sys.argv[1:])
    
