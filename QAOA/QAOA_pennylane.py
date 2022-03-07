# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 20:07:23 2022

@author: burak
"""

import numpy as np
import qiskit
import time
import networkx as nx
import pennylane as qml


'''
QAOA Implementation on Qiskit using SPSA

The task is to implement a QAOA solution for a given random fully connected graph with 12 nodes.

Key points of QAOA:

We have to define 2 Hamiltonians,  HC , a Cost Hamiltonian whose ground state(eigenspace that corresponds to the lowest eigenvalue) is encoded with the problem solution.

The second Hamiltonian  HM , is called a Mixer Hamiltonian, and it has to anti-commute with the Cost Hamiltonian. The QAOA algorithm, consist of, preparing an initial state, then applying an ansatz which consists of the time evolution that corresponds to the hamiltonians.  e−iβHC  and  e−iγHM , while  β  and  γ  are the parameters to optimize over.

The overall goal is to lower the expected value of the measurement over the cost hamiltonian, which would ideally converge to the lowest energy state, which describes a solution.


'''


number_of_qubits = 11
G = nx.erdos_renyi_graph(number_of_qubits, 0.4, seed=123, directed=False)
nx.draw(G, with_labels=True, alpha=0.8)

# Creating an adjacency matrix for the Graph G
adj = nx.adjacency_matrix(G).todense()
adj
#%% Implementation with qiskit

class CircuitRun:

  def __init__(self, number_of_qubits = 12):
    self.number_of_qubits = number_of_qubits
  

  def edgeCount(self, solution, G):
    edge_count = 0
    edges = G.edges()
    for edge in edges:
      edge_1, edge_2 = edge
      if(solution[edge_1] != solution[edge_2]):
        edge_count += 1
    return edge_count * -1

  def expVal(self, counts, G):
    exp = 0
    total_val = 0
    for sol in counts.items():
      solution, count = sol
      edge_count = self.edgeCount(solution[::-1], G)
      exp += (edge_count * count)
      total_val += count
    return exp/total_val

  def QAOA(self, G, params):
    param_idx = 5 #number of layers
    beta = params[:param_idx]
    gamma = params[param_idx:]
    circuit = qiskit.QuantumCircuit(self.number_of_qubits)
    edge_list = list(G.edges())
    for i in range(self.number_of_qubits):
      circuit.h(i)
    #circuit.barrier()
    for p in range(param_idx):
      for edge in edge_list:
        node_1, node_2 = edge
        #circuit.rz(2 * gamma[p] , node_1)
        #circuit.rz(2 * gamma[p] , node_2)
        circuit.rzz(2 * gamma[p], node_1, node_2)
    #circuit.barrier()
    
      for i in range(self.number_of_qubits):
        circuit.rx(2* beta[p], i)
      
    circuit.measure_all()

    return circuit

  def getExpVal(self, G):
    backend = qiskit.Aer.get_backend('qasm_simulator')
    backend.shots = 8192
    def execute_circ(theta):
      circuit = self.QAOA(G, theta)
      counts = backend.run(circuit, seed_simulator = 10, shots = 8192).result().get_counts()
      return self.expVal(counts, G)
    return execute_circ

  def measureCircuit(self, G, theta):
    circuit = self.QAOA(G, theta)
    backend = qiskit.Aer.get_backend('aer_simulator')
    counts = backend.run(circuit, seed_simulator = 10, shots = 8192).result().get_counts()
    return counts

  def getResult(self, G, theta):
    counts = self.measureCircuit(G, theta)
    return max(counts, key = counts.get)[::-1], counts # state which has been measured the most frequently

    

QAOA_Circuit = CircuitRun(number_of_qubits)


from qiskit.algorithms.optimizers import SPSA
start = time.time()
optimizer = SPSA(maxiter=1000)
expVal = QAOA_Circuit.getExpVal(G)
optimized_parameters, final_cost, number_of_circuit_calls = optimizer.optimize(8, expVal, initial_point=np.random.rand(10)* np.pi * 2 )
end = time.time()
print('Optimization terminated within {:.3f} seconds' , end-start)
solution,counts = QAOA_Circuit.getResult(G, optimized_parameters)
final_cost

qiskit.visualization.plot_histogram(counts)

#Testing: Classically check number of edges for any possible bitstring

createBitString = lambda x: str(bin(x)[2:].zfill(number_of_qubits))
solution_dictionary = {}

for i in range(2**number_of_qubits):
  solution_dictionary[createBitString(i)] = QAOA_Circuit.edgeCount(createBitString(i), G)
  
  
min_count = 0
min_index = 0
for i in range(2 ** number_of_qubits):
  res = solution_dictionary[createBitString(i)]
  if ( min_count > res ):
    min_count = res
    min_index = i
min_index, min_count



#Test
def checkResult(solution,solution_dictionary, min_count):
  return solution_dictionary[solution] == min_count

 
print(checkResult(solution,solution_dictionary, min_count))



# %% 
def MixerHamiltonian(beta, layer):
    for wire in range(number_of_qubits):
        qml.RX(2 * beta[layer], wires=wire)
        
def CostHamiltonian(gamma, layer):
    for idx, edge in enumerate(list(G.edges)):
        qml.MultiRZ(2 * gamma[layer], wires = edge)

# %%
dev = qml.device("default.qubit", wires=number_of_qubits, shots= 1024)
@qml.qnode(dev)
def QAOA(beta, gamma, layers = 5, test = False):
    # print((gamma))
    
    for i in range(number_of_qubits):
        qml.Hadamard(wires = i) 
        
    for l in range(layers):
        CostHamiltonian(gamma, l)
        MixerHamiltonian(beta, l)
        
    if(test == True):
        return qml.sample(wires= range(0, number_of_qubits))
    
    return qml.probs(wires = range(0, number_of_qubits))
QAOA(optimized_parameters[:5], optimized_parameters[5:])
sol_index = np.argmax(QAOA(optimized_parameters[:5], optimized_parameters[5:]))
print(createBitString((sol_index)))

np.save('edges_list.npy' , list(G.edges()) )
np.save('optimized_parameters.npy', optimized_parameters)

# QAOA( optimized_parameters[:int (len(optimized_parameters) / 2)], optimized_parameters[int(len(optimized_parameters) / 2):])

#%% Same implementation with pennylane
# from pennylane import numpy as np

# createBitString = lambda x: str(bin(x)[2:].zfill(number_of_qubits))

# dev = qml.device("default.qubit", wires=number_of_qubits, shots=2 ** number_of_qubits)

# def edgeCount(solution, G):
#     edge_count = 0
#     edges = G.edges()
#     for edge in edges:
#         edge_1, edge_2 = edge
#         if(solution[edge_1] != solution[edge_2]):
#             edge_count += 1
#     return edge_count * -1

# def expVal(params):

#     beta  = params[0]
#     gamma = params[1]
#     # print(beta)
#     counts = (QAOA(beta, gamma))
#     exp = 0
#     total_val = 0
#     for solution, count in enumerate(counts):
#         solution = createBitString(solution)
#         edge_count = edgeCount(solution[::-1], G)
#         exp += (edge_count * count)
#         total_val += count
#     return exp/total_val

# def MixerHamiltonian(beta, layer):
#     for wire in range(number_of_qubits):
#         qml.RX(2 * beta[layer], wires=wire)
        
# def CostHamiltonian(gamma, layer):
#     for idx, edge in enumerate(list(G.edges)):
#         qml.CRZ(gamma[layer], wires = edge)

# number_of_layers = 3


# init_params = np.random.rand(2 , number_of_layers, requires_grad = True) * np.pi * 2

# params = init_params


# @qml.qnode(dev)
# def QAOA(beta,gamma, test = False):
#     # print((gamma))
#     layers = len(beta)
#     for i in range(number_of_qubits):
#         qml.Hadamard(wires = i) 
#     for l in range(layers):
#         CostHamiltonian(gamma, l)
#         MixerHamiltonian(beta, l)
#     if(test == True):
#         return qml.sample(wires= range(0, number_of_qubits))
#     return qml.probs(wires = range(0, number_of_qubits))

# # print(params)
# cost_fn = expVal
# opt = qml.AdagradOptimizer(stepsize= 0.5 )
# steps = 30
# for i in range(steps):
#     params = opt.step(cost_fn, params)
#     if (i + 1) % 5 == 0:
#         print("Objective after step {:5d}: {: .7f}".format(i + 1, -expVal(params)))
# QAOA(params[0] , params[1], True)
