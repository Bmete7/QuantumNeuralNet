# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:51:21 2022

@author: burak
"""
import pennylane as qml
import numpy as np
import networkx as nx
import seaborn
from scipy.linalg import expm, logm
import time
import qiskit 
import sys

# %% Utils

n = 3
N = n**2
z = np.array([[1,0] , [0j, -1]])
y = np.array([[0,-1j] , [1j, 0]])
x = np.array([[0,1] , [0j + 1, 0]])
I = np.eye(2)

Splus = np.array([[0j,0], [1,0]])
Sminus = np.array([[0j,1], [0,0]])
G = nx.Graph()

for i in range(n):
    G.add_node(i)


G.add_edge(0,1, weight  = 4)
G.add_edge(0,2, weight  = 2)
G.add_edge(1,2, weight  = 2)



# for i in range(N):
#     for j in range(i + 1, N):
#         weight = np.random.randint(1,3)
#         if((i == 0 and j == 3) or (i == 1 and j==2) ):
#             weight = 1
#         else:
#             weight = 2
#         G.add_edge(i, j, weight = weight)

nx.draw(G, with_labels=True, alpha=0.8)
adj = nx.adjacency_matrix(G).todense()
edges = list(G.edges)

# %% 

createBitString = lambda x,y=N: str(bin(x)[2:].zfill(y))
int2bit = lambda x: str(bin(x)[2:].zfill(N))
bit2int = lambda b: int("".join(str(bs) for bs in b), base = 2)
quantumOuter = lambda inputs: np.outer(inputs.conj().T, inputs)

def commuteCheck(A,B):
    return (A@B == B@A).all()
    
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
    idx = 0
    for i in range(N):
        if(i in listOfQubits):
            out = np.kron(out, paulis[idx])
            idx += 1
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




commuteCheck(Hb121,Hb122)

A = KroneckerProduct([0,5], x, 9)
B = KroneckerProduct([0,5], y, 9)



# %% 

Hbprime = 0
for u in range(n):
    for v in range(n):
        for i in range(n):
            for j in range(n):
                if((u,v) in edges or (v,u) in edges):
                    if(i!=j):
                        orders = [(u * 3 + i), (v * 3 + j), (u * 3 + i), (v * 3 + i)]
                        
                        paulis = [Sminus, Splus, Splus, Sminus]
                        
                        result = KroneckerProductString(orders, paulis, N)
                        Hbprime += result
                        Hbprime += result.conj().T
                        

n = 3
N = n**2
Hbprime = 0

for i in range(n):
    for j in range(n):
        for u in range(n):
            for v in range(n):
                if(i==j or u == v):
                    continue
                
                orders = [(u * (n) + i), (v * (n) + j), (u * (n) + j), (v * (n) + i)]
                paulis = [Splus, Sminus, Sminus, Splus]
                
                result = KroneckerProductString(orders, paulis, n**2)
                Hbprime += result
                Hbprime += result.conj().T

solution , ground_state , solutions, index = findGroundStates(Hbprime, N)

def timeEvolution(local_hamiltonian, psi, timestamp = 1):
    # U = expm(-1j * H * t )
    U = expm(local_hamiltonian * -1j * timestamp)
    return U @ psi

int2bit = lambda x: str(bin(x)[2:].zfill(N))
bit2int = lambda b: int("".join(str(bs) for bs in b), base = 2)

valid_states_str = ['100010001',
 '100001010',
 '010100001',
 '010001100',
 '001010100',
 '001100010']

feasibleSubspace = [np.eye(2**N)[ : , (bit2int(val))] for val in valid_states_str]



# %% Cost Hamiltonian

HC = 0j
l = 0
for edge in edges:
    u, v = edge
    for j in range(n - 1):
        start = time.time()
        HC += (adj[u, v]) * (KroneckerProduct([(u * n + j), (v * n + j + 1)], z, N) + KroneckerProduct([(u * n + j + 1), (v * n + j)], z, N))
        end = time.time()
        
    l += (adj[u, v])

l *= (2 - n/2)  
HC += (l * np.eye(2**N))


# %% Analysis of the Cost Hamiltonian


ew, ev = np.linalg.eig(HC)
for i in range(6):
    print(ew[bit2int(valid_states_str[i])])




# %% 


# # # Preparing Hb
# # Hb010 = KroneckerProduct([0,3], x, 9) + KroneckerProduct([0,3], y, 9)
# # Hb011 = KroneckerProduct([1,4], x, 9) + KroneckerProduct([1,4], y, 9)
# # Hb012 = KroneckerProduct([2,5], x, 9) + KroneckerProduct([2,5], y, 9)

# # Hb020 = KroneckerProduct([0,6], x, 9) + KroneckerProduct([0,6], y, 9)
# # Hb021 = KroneckerProduct([1,7], x, 9) + KroneckerProduct([1,7], y, 9)
# # Hb022 = KroneckerProduct([2,8], x, 9) + KroneckerProduct([2,8], y, 9)

# # Hb120 = KroneckerProduct([3,6], x, 9) + KroneckerProduct([3,6], y, 9)
# # Hb121 = KroneckerProduct([4,7], x, 9) + KroneckerProduct([4,7], y, 9)
# # Hb122 = KroneckerProduct([5,8], x, 9) + KroneckerProduct([5,8], y, 9)
# # HbCheck = (Hb010 @ Hb011 @ Hb012) + (Hb020 @ Hb021 @ Hb022) + (Hb120 @ Hb121 @ Hb122)

# # ew, ev = np.linalg.eig(HbCheck)


# def returnPauliStrings():
    
#     y = np.array([[0,-1j] , [1j, 0]])
#     x = np.array([[0,1] , [0j + 1, 0]])
    
#     pauliTerms = []
#     orderTerms = []
#     strpauliTerms = []
#     MixerPaulis = [x,y]
#     strMixerPauli = ['x' , 'y']
    
    
#     for edge in edges:
#         u, v = edge
#         for j in range(n-1):
#             pauliStrings = []
#             pauliOrder = []
#             strpauliStrings = []
            
#             pauliOrder.append(u*n)
#             pauliOrder.append(v*n)
            
#             pauliStrings.append(MixerPaulis[j])
#             pauliStrings.append(MixerPaulis[j])
            
#             strpauliStrings.append(strMixerPauli[j])
#             strpauliStrings.append(strMixerPauli[j])
            
#             for k in range(n-1):
#                 pauliOrder1 = pauliOrder.copy()
#                 pauliStrings1 = pauliStrings.copy()
#                 strpauliStrings1 = strpauliStrings.copy()
                
#                 pauliOrder1.append(u*n +  1)
#                 pauliOrder1.append(v*n + 1)
                
#                 pauliStrings1.append(MixerPaulis[k])
#                 pauliStrings1.append(MixerPaulis[k])
                
#                 strpauliStrings1.append(strMixerPauli[k])
#                 strpauliStrings1.append(strMixerPauli[k])
                
#                 # orderTerms.append(pauliOrder1)
#                 # pauliTerms.append(pauliStrings1)
#                 # strpauliTerms.append(strpauliStrings1)
                
#                 for m in range(n-1):
#                     pauliOrder2 = pauliOrder1.copy()
#                     pauliStrings2 = pauliStrings1.copy()
#                     strpauliStrings2 = strpauliStrings1.copy()
                    
#                     pauliOrder2.append(u*n + 2)
#                     pauliOrder2.append(v*n + 2)
                    
#                     pauliStrings2.append(MixerPaulis[m])
#                     pauliStrings2.append(MixerPaulis[m])
                    
#                     strpauliStrings2.append(strMixerPauli[m])
#                     strpauliStrings2.append(strMixerPauli[m])
            
#                     orderTerms.append(pauliOrder2)
#                     pauliTerms.append(pauliStrings2)
#                     strpauliTerms.append(strpauliStrings2)

    

    
#     sortedorderTerms = []
#     sortedpauliTerms = []
#     sortedstrpauliTerms = []
#     for idx, od in enumerate(orderTerms):
#         sortedorderTerms.append(sorted(od))
#         sortedpauliTerms.append([y for _, y in sorted(zip(od, pauliTerms[idx]))])
#         sortedstrpauliTerms.append([y for _, y in sorted(zip(od, strpauliTerms[idx]))])
    
#     return sortedorderTerms, sortedpauliTerms, sortedstrpauliTerms

# sortedorderTerms, sortedpauliTerms, sortedstrpauliTerms = returnPauliStrings()

# HB = 0

# for idx, a in enumerate(sortedorderTerms):
#     HB += KroneckerProductString(a, sortedpauliTerms[idx], N)


# HbCheck = (Hb010 @ Hb011 @ Hb012) + (Hb120 @ Hb121 @ Hb122)
# (HbCheck == HB).all()


# # %% 
# ew , ev = np.linalg.eig(HbCheck)

# np.argwhere(np.abs(ev[:,207] )>= 10e-3)
# np.argwhere(ew == np.amin(ew))


# res = expm(-1j * HB) @ ev[:,20]
# (res - ev[:,20])
# np.argwhere(np.abs(res) >= 10e-3)
# np.argwhere(np.abs(ev[:,20] )>= 10e-3)
# res
# # %% check if holds

# KroneckerProduct([0,3], x, 9) + KroneckerProduct([0,3], y, 9)
# KroneckerProduct([0,3], x, 9) + KroneckerProduct([0,3], y, 9)
# KroneckerProduct([0,3], x, 9) + KroneckerProduct([0,3], y, 9)




# # %% 
# for idx in range(2**9):
#     vals = (np.argwhere(ev[:,idx] != 0 ))    
#     validity = [checkSolution(int2bit(val[0])) for val in vals]
#     if(np.array(validity).all() == True):

#         # print(ew[np.ndarray.item(vals)], ' eigenvalue')
#         # print(ev[np.ndarray.item(vals), idx])
#         print('*')
        

# # %% the mixer holds for 2 node setting, the low energy eigenspace encodes the valid solutions


# minihb = KroneckerProductString([0,1,2,3], [x,x,x,x], 4) +  KroneckerProductString([0,1,2,3], [x,y,x,y], 4) +  KroneckerProductString([0,1,2,3], [y,x,y,x], 4) +  KroneckerProductString([0,1,2,3], [y,y,y,y], 4) 
# ew, ev = np.linalg.eig(minihb)

# np.where(ew== np.amin(ew))
# ev[:, 3]

# Hbprime = 0
# for u in range(2):
#     for v in range(u + 1, n):
#         for i in range(2):
#             for j in range(u + 1, n):
                
#                 orders = [(u * 3 + i), (u * 3 + j), (v * 3 + i), (v * 3 + j)]
                
#                 paulis = [Sminus, Splus, Splus, Sminus]
                
#                 result = KroneckerProductString(orders, paulis, 4)
#                 Hbprime += result
#                 Hbprime += result.conj().T

# ew, ev = np.linalg.eig(Hbprime)
                


# #%%



# valid_states = [bit2int(b) for b in valid_states_str]

# # %% 



# def checkSolution(sol):
#     if(type(sol) == int):
#         sol= int2bit(sol)
    
#     for u in range(n):
#         first_check = 0
#         for i in range(n):
#             first_check += int(sol[u*3 + i])
#         if first_check != 1:
#             return False
#     for i in range(n):
#         second_check = 0
#         for u in range(n):
#             second_check += int(sol[u*3 + i])
#         if second_check != 1:
#             return False
#     return True

# def checkSolutionTests():
#     ctr = 0
#     for i in range(2**N):
#         case = int2bit(i)
#         if((checkSolution(case))):
#             ctr += 1
#             print((case))
#     return ctr

# ctr = checkSolutionTests()
# # %% 



# class CircuitRun:

#   def __init__(self, number_of_qubits = 9):
#     self.number_of_qubits = number_of_qubits
  

#   def edgeCount(self, solution, G):
#     edge_count = 0
#     edges = G.edges()
#     for edge in edges:
#       edge_1, edge_2 = edge
#       if(solution[edge_1] != solution[edge_2]):
#         edge_count += 1
#     return edge_count * -1

#   def expVal(self, counts, G):
#     exp = 0
#     total_val = 0
#     for sol in counts.items():
#       solution, count = sol
#       edge_count = self.edgeCount(solution[::-1], G)
#       exp += (edge_count * count)
#       total_val += count
#     return exp/total_val

#   def QAOA(self, G, params):
#     param_idx = 5 #number of layers
#     beta = params[:param_idx]
#     gamma = params[param_idx:]
#     circuit = qiskit.QuantumCircuit(self.number_of_qubits)
#     edge_list = list(G.edges())
#     for i in range(self.number_of_qubits):
#       circuit.h(i)
#     #circuit.barrier()
#     for p in range(param_idx):
#       for edge in edge_list:
#         node_1, node_2 = edge
#         #circuit.rz(2 * gamma[p] , node_1)
#         #circuit.rz(2 * gamma[p] , node_2)
#         circuit.rzz(2 * gamma[p], node_1, node_2)
#     #circuit.barrier()
    
#       for i in range(self.number_of_qubits):
#         circuit.rx(2* beta[p], i)
      
#     circuit.measure_all()

#     return circuit

#   def getExpVal(self, G):
#     backend = qiskit.Aer.get_backend('qasm_simulator')
#     backend.shots = 8192
#     def execute_circ(theta):
#       circuit = self.QAOA(G, theta)
#       counts = backend.run(circuit, seed_simulator = 10, shots = 8192).result().get_counts()
#       return self.expVal(counts, G)
#     return execute_circ

#   def measureCircuit(self, G, theta):
#     circuit = self.QAOA(G, theta)
#     backend = qiskit.Aer.get_backend('aer_simulator')
#     counts = backend.run(circuit, seed_simulator = 10, shots = 8192).result().get_counts()
#     return counts

#   def getResult(self, G, theta):
#     counts = self.measureCircuit(G, theta)
#     return max(counts, key = counts.get)[::-1], counts # state which has been measured the most frequently