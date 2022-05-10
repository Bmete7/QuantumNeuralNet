# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:58:29 2022

@author: burak
"""

# encoding with pennylaneimport numpy as np
import networkx as nx
from scipy.linalg import expm
import time
import qiskit
from QAutoencoder.Utils import *
from qiskit.algorithms.optimizers import SPSA
from copy import deepcopy
import pennylane as qml
from pennylane import numpy as np



n = 3  # number of nodes in the graph
N = n**2 # number of qubits to represent the one-hot-encoding of TSP

G = nx.Graph()

for i in range(n):
    G.add_node(i)

n_wires = N

G.add_edge(0,1, weight  = 1)
G.add_edge(0,2, weight  = 1)
G.add_edge(1,2, weight  = 100)

pos=nx.spring_layout(G) # pos = nx.nx_agraph.graphviz_layout(G)
nx.draw_networkx(G,pos)
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
adj = nx.adjacency_matrix(G).todense()
edges = list(G.edges)




# %% 

def KroneckerProduct(listOfQubits, pauli, N):
    out = np.array([1])
    for i in range(N):
        if(i in listOfQubits):
            out = np.kron(out, pauli)
        else:
            out = np.kron(out, I)
    return out
int2bit = lambda x,y=9: str(bin(x)[2:].zfill(y))
bit2int = lambda b: int("".join(str(bs) for bs in b), base = 2) 
# %% 

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

# psi1 = np.zeros(2**9)
# psi1[390] = 1
# psi2 = np.zeros(2**9)
# psi2[496] = 1
# quantumOuter = lambda i1,i2: np.outer(i1, i2.conj().T) # braket

# Ham1 = quantumOuter(psi1,psi1) * 40
# Ham2 = quantumOuter(psi2,psi2) * 40
# HC += Ham1
# HC += Ham2

# HC = np.eye(512)
# HC[84,84] = -50
# %% 

input_states = [[0,0,1,0,1,0,1,0,0],
                [0,0,1,1,0,0,0,1,0],
                [0,1,0,0,0,1,1,0,0],
                [0,1,0,1,0,0,0,0,1],
                [1,0,0,0,0,1,0,1,0],
                [1,0,0,0,1,0,0,0,1]
                ] # 2-1-0 , 1-2-0,  2-0-1, 1-0-2, 0-2-1, 0-1-2
                # 000      001       010    011   100    101 
input_states_str = ['001010100', 
                   '001100010',
                   '010001100',
                   '010100001',
                   '100001010',
                   '100010001']
input_states_int = [bit2int(s) for s in input_states_str]

for i in input_states_int:
    print(HC[i,i])
# %% 
layers = 1

dev = qml.device("default.qubit", wires= 9 + (6 * layers) , shots=1024)

n_layers = layers
statevec = np.zeros(2**9)
for st in input_states_int:
    statevec[st] += 1 / np.sqrt(6)

@qml.qnode(dev)
def circuit(params, edge=None, feature_vector = None, test_mode = False):
    global n_layers
    
    # qml.BasisEmbedding(features=feature_vector, wires=range(9))
    # initialize the qubits in the latent space
    for wire in range(0):
        qml.Hadamard(wires=wire)
        
    qml.Toffoli(wires = [0, 1, 3])
    qml.CNOT(wires = [3, 0])        
    qml.PauliX(3)
    # qml.QubitStateVector(statevec, wires=range(9))
    qml.PauliX(2).inv()
    
    qml.MultiControlledX([2,0,1], [6], '100').inv()
    qml.MultiControlledX([2,0,1], [4], '100').inv()
    
    qml.Toffoli(wires = [1, 2, 6]).inv()
    qml.Toffoli(wires = [1, 2, 5]).inv()
    
    qml.Toffoli(wires = [0, 2, 7]).inv()
    qml.Toffoli(wires = [0, 2, 5]).inv()
    
    qml.PauliX(2).inv()
    
    qml.MultiControlledX([2,0,1], [7], '100').inv()
    qml.MultiControlledX([2,0,1], [3], '100').inv()

    qml.Toffoli(wires = [1, 2, 8]).inv()
    qml.Toffoli(wires = [1, 2, 3]).inv()
    
    qml.Toffoli(wires = [0, 2, 8]).inv()
    qml.Toffoli(wires = [0, 2, 4]).inv()

    qml.Toffoli(wires = [3, 7, 2]).inv()
    qml.Toffoli(wires = [1, 3, 2]).inv()
    qml.Toffoli(wires = [0, 4, 2]).inv()    
    for l in range(n_layers):
        
        for edge in edges:
            u,v = edge
            for step in range(n - 1):
                q_1 = (u * n) + step
                q_2 = (v * n) + step + 1 # (+1 because we want that node visited in the next step)
                
                q_1_rev = (u * n) + step + 1 # (reverse path from above)
                q_2_rev = (v * n) + step  
                
                
                qml.CNOT(wires=[q_1, q_2])
                qml.RZ(params[l], wires= q_2)
                qml.CNOT(wires=[q_1, q_2])
                
                qml.CNOT(wires=[q_1_rev, q_2_rev])
                qml.RZ(params[l], wires= q_2_rev)
                qml.CNOT(wires=[q_1_rev, q_2_rev])    
        
        
        #encoder
        qml.MultiControlledX([0,1], [2], '00')
        
        qml.Toffoli(wires = [0, 4, 2])
        qml.Toffoli(wires = [1, 3, 2])
        qml.Toffoli(wires = [3, 7, 2])
    
        qml.Toffoli(wires = [0, 2, 4])
        qml.Toffoli(wires = [0, 2, 8])
        
        qml.Toffoli(wires = [1, 2, 3])
        qml.Toffoli(wires = [1, 2, 8])
    
        qml.MultiControlledX([2,0,1], [3], '100')
        qml.MultiControlledX([2,0,1], [7], '100')
        
        qml.PauliX(2)
        
        qml.Toffoli(wires = [0, 2, 5])
        qml.Toffoli(wires = [0, 2, 7])
        
        qml.Toffoli(wires = [1, 2, 5])
        qml.Toffoli(wires = [1, 2, 6])
        
        qml.MultiControlledX([2,0,1], [4], '100')
        qml.MultiControlledX([2,0,1], [6], '100')
        
        qml.PauliX(2)
        
        #discarder
        for i in range(6):
            qml.SWAP(wires = [(6*l) + 3 + i, (6*l)+ 9 + i])
            
            
        for i in range(3):
            qml.RX(params[n_layers + l], wires = [i])
            
        qml.Toffoli(wires = [0, 1, 3])
        qml.CNOT(wires = [3, 0])        
        qml.PauliX(3)
            
        if(test_mode == False or (test_mode == True and l != (n_layers - 1))):
            
            #decoder
            qml.PauliX(2).inv()
            
            qml.MultiControlledX([2,0,1], [6], '100').inv()
            qml.MultiControlledX([2,0,1], [4], '100').inv()
            
            qml.Toffoli(wires = [1, 2, 6]).inv()
            qml.Toffoli(wires = [1, 2, 5]).inv()
            
            qml.Toffoli(wires = [0, 2, 7]).inv()
            qml.Toffoli(wires = [0, 2, 5]).inv()
            
            qml.PauliX(2).inv()
            
            qml.MultiControlledX([2,0,1], [7], '100').inv()
            qml.MultiControlledX([2,0,1], [3], '100').inv()
        
            qml.Toffoli(wires = [1, 2, 8]).inv()
            qml.Toffoli(wires = [1, 2, 3]).inv()
            
            qml.Toffoli(wires = [0, 2, 8]).inv()
            qml.Toffoli(wires = [0, 2, 4]).inv()
        
            qml.Toffoli(wires = [3, 7, 2]).inv()
            qml.Toffoli(wires = [1, 3, 2]).inv()
            qml.Toffoli(wires = [0, 4, 2]).inv()
            
            qml.MultiControlledX([0,1], [2], '00').inv()
        else:
            return qml.probs(range(0,3))

    return qml.expval(qml.Hermitian(HC, wires=[0,1,2,3,4,5,6,7,8]))


    

    
# %% 
from noisyopt import minimizeSPSA
from noisyopt import minimizeCompass

flat_shape = layers * 2
param_shape = (layers*2,1)
init_params = np.random.normal(scale=0.1, size=param_shape, requires_grad=True)

init_params_spsa = init_params.reshape(flat_shape)
init_params_spsa[0]= 2
init_params_spsa[1]= 2
niter_spsa = 10



store_costs = []
device_execs = []
start = time.time()
def callback_fn(init_params_spsa):
    global start
    cost_val = circuit(init_params_spsa, n_layers = 1)
    print(cost_val, init_params_spsa)
    store_costs.append(cost_val)
    num_executions = int(dev.num_executions / 2)
    device_execs.append(num_executions)
    end = time.time()
    print('time elapsed : {} s '.format(end - start))
    start = time.time()
        
res = minimizeSPSA(
    circuit,
    x0=init_params_spsa.copy(),
    niter=niter_spsa,
    paired=False,
    c=0.015,
    a=0.01,
    callback = callback_fn
)


circuit(res.x, test_mode = True)



# %% 

def guessState(out):
    majority_vote = np.argmax(out[:-2])
    return int2bit(majority_vote, 3)

def validationError(pred):
    # calculate the accuracy w.r.t costly paths. 
    # Disregard the optimal paths.(since they are always multiple)
    error = 0
    ground_truth = [0,0,1,1,0,0,0,0]
    for i in range(len(ground_truth)):
        if(i== 2 or i == 3):
            continue
        else:
            error += np.abs(ground_truth[i] - pred[i])
    return 1- error

errors = []
for i in range(len(guesses)):
    errors.append(validationError(guesses[i]))
    
plt.plot(errors)
objective_func_evals
np.save('losses.npy', losses)
np.save('guesses.npy', guesses)

np.save('errors.npy', errors)
np.save('objective_func_evals.npy', objective_func_evals)


np.load('objective_func_evals.npy')
# %% 

losses = []
guesses = []
opt= qml.RMSPropOptimizer(stepsize = 0.03)
for i in range(30):
    init_params_spsa = opt.step(circuit, init_params_spsa)
    loss = circuit(init_params_spsa) 
    print(loss)
    guess = (circuit(init_params_spsa, test_mode = True) )
    guesses.append(guess)
    print(guess)
    
    losses.append(loss)
    circuit(init_params_spsa, test_mode = True)
    
    
    
# %% 

from matplotlib import pyplot as plt
running_avg = []
running_loss= 0
for idx,cost in enumerate(losses):
    running_loss += cost
    running_avg.append(running_loss / (idx+1))
plt.plot(running_avg)
running_avg

# %% objetive function graph
layer1params = (2,1)
layer1p = np.random.normal(scale=0.1, size=layer1params, requires_grad=True)

layer1p = layer1p.reshape(2)


epsilon= np.arange(-np.pi, np.pi, 0.25)



import time
# %% 
objective_func_evals = np.zeros((epsilon.shape[0], epsilon.shape[0] ))

for idx1, epsilon1 in enumerate(epsilon):
    for idx2, epsilon2 in enumerate(epsilon):
    
        
        layer1p[0] = epsilon1
        layer1p[1] = epsilon2
        
        start = time.time()
        res = circuit(layer1p)
        objective_func_evals[idx1,idx2] = res
        print(res)
        end = time.time()
        print('time elapsed : {} s '.format(end - start))

# %% Visualize 

mesh = np.meshgrid(epsilon,epsilon)

fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height]) 

cp = plt.contourf(epsilon, epsilon, objective_func_evals )
plt.colorbar(cp)

ax.set_title('Expected energy landscape, p=1')
ax.set_xlabel('gamma')
ax.set_ylabel('beta')
plt.show()
#%% 
fig = plt.figure(figsize=(6, 3.2))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(objective_func_evals)
ax.set_aspect('equal')
cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()


# %% 
circuit([epsilon[10], epsilon[11]], test_mode = True)

np.sum(HC)
diagHC= np.diag(HC)


sorteddiag = np.sort(diagHC)
sss = np.argsort(diagHC)
bbb = []
for i in sss:
    bbb.append(int2bit(i, 9))
    
bbb = np.array(bbb)
for i in graph3_dict:
    print(bit2int(i))



#%% 
opt = qml.GradientDescentOptimizer(stepsize=0.04)

steps = 10
params = init_params_spsa.copy()

for i in range(steps):
    # update the circuit parameters
    params = opt.step(circuit, params)


    


# %% 
opt_params = np.array([-1.457, 0.0006], requires_grad = True)
params = np.array([np.pi, np.pi], requires_grad = True)
optimizer = qml.RMSPropOptimizer()
steps = 70
for i in range(steps):
    gamma,beta = optimizer.step(circuit, params)
    print(circuit(params))
# for i in range(6):
#     res = (circuit(gamma,beta, feature_vector =  input_states[i]))
#     print(res)
#     # print(int2bit(np.argmax(res), 15))

par = [gamma, beta ]
print(circuit(par, test_mode = True))

