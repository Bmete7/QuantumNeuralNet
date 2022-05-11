# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:58:29 2022

@author: burak
"""

# Libraries
import networkx as nx
from scipy.linalg import expm
import time
import qiskit
from qiskit.algorithms.optimizers import SPSA
from copy import deepcopy
import pennylane as qml
from pennylane import numpy as np
from noisyopt import minimizeSPSA
from matplotlib import pyplot as plt



from QAutoencoder.Utils import *
from Visualization.visualizations import *
from Test.benchmark import *
from Ansatz import *
# %% Problem Setting

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

# %% Formulating the Problem Hamiltonian

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

## Hamiltonian as a pennylane operator
# ops = []
# coeffs = []

# for edge in edges:
#     u, v = edge
#     for j in range(n - 1):
#         op = []
        
#         coeffs.append(adj[u,v])
#         coeffs.append(adj[u,v])
#         ops.append(qml.PauliZ(u*n + j) @ qml.PauliZ(v*n + j + 1))
#         ops.append(qml.PauliZ(u*n + j + 1) @ qml.PauliZ(v*n + j ))
        
# HC = qml.Hamiltonian(coeffs, ops).eigvals

    

# %% Ansatz 
n_layers = 1


ancillary = calculateRequiredAncillaries(n)
dev = qml.device("default.qubit", wires= N + ((N-ancillary) * n_layers), shots=1024)
                
@qml.qnode(dev)
def circuit(params, edge = None, feature_vector = None, test_mode = False):
    '''
    QAOA ansatz interleaved with an encoder, for solving constrained optimization
    problems on 3-vertex graphs. Even though it is implemented for 3-vertices, 
    the ansatz can easily be generalized into a d-vertex setting

    Parameters
    ----------
    params : list
        gammas and betas, parameters for cost and mixer unitaries
    
    edge : list, optional
        list of edges in the graph
    feature_vector : array, optional    
        quantum state vector for state preparation
    test_mode : bool
        if true, it returns a valid solution encoding, that can be converted into a route
        if not, it returns the expectation value of the cost hamiltonian
    Returns
    -------
    TYPE
        either the measurement of the cost hamiltonian, or the state vector of the latent space, depending on the test_mode flag

    '''
    
    global n_layers, ancillary, edges, adj, n, N, HC
    
    # qml.BasisEmbedding(features=feature_vector, wires=range(9))
    # initialize the qubits in the latent space

    # Initial State Preparation
    for wire in range(0):
        qml.Hadamard(wires=wire)
    
    def reduceFalsePaths():
        qml.Toffoli(wires = [0, 1, 3])
        qml.CNOT(wires = [3, 0])        
        qml.PauliX(3)
        
    reduceFalsePaths()
    # qml.QubitStateVector(statevec, wires=range(9))
    
    U_D(n)
    for l in range(n_layers):
    # Cost unitary
        U_C(params, n, l, edges)
        # Encoder
        U_E(n)
        
        # Discarding the subsystem B, via swapping with a reference system
        discardQubits(n, l)
            
        # Mixer Unitary
        U_M(params, ancillary, n_layers, l)
   
        reduceFalsePaths()
            
        # Decoder, if it is the last QAOA iteration in a test mode, it does not run, since then a state vector in latent
        # space is returned instead of an expectation value of the cost hamiltonian
        if(test_mode == False or (test_mode == True and l != (n_layers - 1))):            
            U_D(n)
        else:
            return qml.probs(range(0,ancillary))
    # return qml.expval(HC)
    return qml.expval(qml.Hermitian(HC, wires=range(N)))



# %% Training with SPSA 

lr = 0.03 
losses = []
optimizers = []
preds = []
running_avgs = []
accuracies = []
opt_params = []
flat_shape = n_layers * 2
param_shape = (n_layers * 2, 1)
initial_params = np.random.normal(scale=0.1, size=param_shape, requires_grad=True)
initial_params = initial_params.reshape(flat_shape)

# %% 

init_params = initial_params.copy()
device_execs = []
spsa_losses = []
spsa_preds = []

ctr = 0
niter_spsa = 2


current_model = circuit
current_dev = dev

def callback_fn(params):
    global start, ctr
    
    loss = current_model(params)
    pred = current_model(params, test_mode = True)
    
    spsa_preds.append(pred)
    spsa_losses.append(loss)
    
    num_executions = int(dev_mixers.num_executions / 2)
    device_execs.append(num_executions)
    
    end = time.time()
    ctr += 1
    print('Epoch {} elapsed in {}s,  Loss: {}'.format( ctr , end - start, loss))
        
start = time.time()

res = minimizeSPSA(
    current_model,
    x0=init_params.copy(),
    niter=niter_spsa,
    paired=False,
    c=0.15,
    a=0.2,
    callback = callback_fn
)

# SPSA outputs
losses.append(spsa_losses)
optimizers.append('SPSA')
preds.append(spsa_preds)
spsa_running_avgs = calculateRunningAverage(spsa_losses)
running_avgs.append(spsa_running_avgs)

spsa_optimal_params = res.x

spsa_accuracies = []
for i in range(len(spsa_preds)):
    spsa_accuracies.append(validationError(spsa_preds[i]))
    
accuracies.append(spsa_accuracies)
opt_params.append(spsa_optimal_params)


# %% SGD

rmsOptimizer = qml.RMSPropOptimizer(stepsize = lr)
adamOptimizer = qml.AdamOptimizer(stepsize = lr)

opts = [rmsOptimizer, adamOptimizer]
opt_names = ['RMSProp' , 'ADAM']

for idx, opt in enumerate(opts):
    init_params = initial_params.copy()
    cur_losses = []
    cur_preds = []
    for i in range(60):
        init_params = opt.step(circuit, init_params)
        loss = circuit(init_params)
        pred = (circuit(init_params, test_mode = True))
        print(loss)
        cur_preds.append(validationError(pred))
        cur_losses.append(loss)
    running_avg = calculateRunningAverage(cur_losses)
    losses.append(cur_losses)
    preds.append(cur_preds)
    opt_params.append(init_params)
    

# %%  Save Results

np.save('opt_params.npy', opt_params)


# %% Visualization
# losses = [spsa_running_avg, rms_running_avg , adam_running_avg]
optimizers = ['SPSA', 'RMSProp', 'Adam']
loss_plot(losses, optimizers, 'loss')
loss_plot(preds, optimizers, 'acc')


# %% Energy Landscape

search_params = initial_params.copy()
param_interval = 0.25
epsilon= np.arange(-np.pi, np.pi, param_interval)

# %% Visualize Energy Landscapes

objective_fn_evals = np.zeros((epsilon.shape[0], epsilon.shape[0] ))

for idx1, epsilon1 in enumerate(epsilon):
    for idx2, epsilon2 in enumerate(epsilon):
        search_params[0] = epsilon1
        search_params[1] = epsilon2
        objective_fn_evals[idx1,idx2] = circuit(search_params)
        
energyLandscape(epsilon, fn_eval)



# %% Implementing QAOA-M https://doi.org/10.1145/3149526.3149530

dev_mixers = qml.device("default.qubit", wires= N , shots= 1024)

@qml.qnode(dev_mixers)
def circuit_mixers(params, edge=None, feature_vector = None, test_mode = False):
    '''
    QAOA ansatz interleaved with an encoder, for solving constrained optimization
    problems on 3-vertex graphs. Even though it is implemented for 3-vertices, 
    the ansatz can easily be generalized into a d-vertex setting

    Parameters
    ----------
    params : list
        gammas and betas, parameters for cost and mixer unitaries
    
    edge : list, optional
        list of edges in the graph
    feature_vector : array, optional
    
        quantum state vector for state preparation
    test_mode : bool
        if true, it returns a valid solution encoding, that can be converted into a route
        if not, it returns the expectation value of the cost hamiltonian
    Returns
    -------
    TYPE
        either the measurement of the cost hamiltonian, or the state vector of the latent space, depending on the test_mode flag

    '''
    
    global n_layers
    
    # qml.BasisEmbedding(features=feature_vector, wires=range(9))
    # initialize the qubits in the latent space

    for edge in edges:
        u,v = edge
        for step in range(n - 1):
            
            q_1 = (u * n) + step
            q_2 = (v * n) + step + 1 # (+1 because we want that node visited in the next step)
            
            q_1_rev = (u * n) + step + 1 # (reverse path from above)
            q_2_rev = (v * n) + step  

            qml.CNOT(wires=[q_1, q_2])
            qml.RZ(params[1], wires= q_2)
            qml.CNOT(wires=[q_1, q_2])
            
            qml.CNOT(wires=[q_1_rev, q_2_rev])
            qml.RZ(params[1], wires= q_2_rev)
            qml.CNOT(wires=[q_1_rev, q_2_rev])    
            
   
    for edge in edges:
        u,v = edge
        
        q1 = u * n 
        q2 = v * n 
        for s1 in [True, False]:
            
            for s2 in [True, False]:
                for s3 in [True, False]:
                    
                    if(s1):
                        qml.Hadamard(q1)
                        qml.Hadamard(q2)
                    else:
                        qml.RX(params[0], wires = q1)
                        qml.RX(params[0], wires = q2)
                    if(s2):
                        qml.Hadamard(q1 + 1)
                        qml.Hadamard(q2 + 1)
                    else:
                        qml.RX(params[0], wires = q1 + 1)
                        qml.RX(params[0], wires = q2 + 1)
                    
                    if(s3):
                        qml.Hadamard(q1 + 2)
                        qml.Hadamard(q2 + 2)
                    else:
                        qml.RX(params[0], wires = q1 + 2)
                        qml.RX(params[0], wires = q2 + 2)
                    
                    qml.CNOT(wires= [q1, q1+1])
                    qml.CNOT(wires= [q1+1, q1+2])
                    qml.CNOT(wires= [q1+2, q2])
                    qml.CNOT(wires= [q2, q2+1])
                    qml.CNOT(wires= [q2+1, q2+2])
                    
                    qml.RZ(params[0], wires = q2 + 2)
                    
                    qml.CNOT(wires= [q1, q1+1])
                    qml.CNOT(wires= [q1+1, q1+2])
                    qml.CNOT(wires= [q1+2, q2])
                    qml.CNOT(wires= [q2, q2+1])
                    qml.CNOT(wires= [q2+1, q2+2])
                    
                    if(s3):
                        qml.Hadamard(q1 + 2).inv()
                        qml.Hadamard(q2 + 2).inv()
                    else:
                        qml.RX(params[0], wires = q1 + 2).inv()
                        qml.RX(params[0], wires = q2 + 2).inv()    
                    
                    if(s2):
                        qml.Hadamard(q1 + 1).inv()
                        qml.Hadamard(q2 + 1).inv()
                    else:
                        qml.RX(params[0], wires = q1 + 1).inv()
                        qml.RX(params[0], wires = q2 + 1).inv()
                    
                    if(s1):
                        qml.Hadamard(q1).inv()
                        qml.Hadamard(q2).inv()
                    else:
                        qml.RX(params[0], wires = q1).inv()
                        qml.RX(params[0], wires = q2).inv()
                            
    if(test_mode == True):
        return qml.probs(range(9))
    return qml.expval(qml.Hermitian(HC, wires=[0,1,2,3,4,5,6,7,8]))


# %% Optimizing the QAOA-M

mixer_params = initial_params.copy()

mixer_losses = []
mixer_preds = []
mixer_opt = qml.RMSPropOptimizer(stepsize = 0.03)
for i in range(60):
    start = time.time()
    mixer_params = mixer_opt.step(circuitnasa, mixer_params)
    
    mixer_loss = circuit(mixer_params)
    mixer_pred = (circuit(mixer_params, test_mode = True))
    
    mixer_preds.append(mixer_pred)
    mixer_losses.append(mixer_loss)
    end = time.time()
    if(i % 10):
        print('Epoch {} elapsed in {}s,  Loss: {}'.format( i , end - start, mixer_loss))

# %% Circuits Specs for benchmarking
n_layers = 1

dev = qml.device("default.qubit", wires= N + ((N-ancillary) * n_layers), shots=1024)
n_settings = [3,4]


ansatz_list = [circuit_3_benchmark, circuit_4_benchmark, circuit_3_mixers_benchmark, circuit_4_mixers_benchmark ]
ansatz_names = ['QAOA-E1-3', 'QAOA-E1-4', 'QAOA-M1-3', 'QAOA-M1-4']
results = []
durations = []

  
@qml.qnode(dev_benchmark)
def circuit_benchmark(params, n):
    U_E(n)
    # Discarding the subsystem B, via swapping with a reference system
    discardQubits(n, 0)
        
    return qml.probs(range(0,1))  
  
for n in n_settings:
    N = n**2
    ancillary = calculateRequiredAncillaries(n)
    dev_benchmark = qml.device("default.qubit", wires= N + ((N-ancillary) * n_layers), shots=1024)
    start = time.time()
    results.append(qml.specs(circuit_benchmark)(init_params))
    end = time.time()
    durations.append(end-start)

start = time.time()
results.append(qml.specs(circuit_3_mixers_benchmark)(init_params))
end = time.time()
durations.append(end-start)

n = 4  # number of nodes in the graph
N = n**2 # number of qubits to represent the one-hot-encoding of TSP

G = nx.Graph()

for i in range(n):
    G.add_node(i)

G.add_edge(0,1, weight  = 1)
G.add_edge(0,2, weight  = 1)
G.add_edge(1,2, weight  = 100)
G.add_edge(0,3, weight  = 11)
G.add_edge(1,2, weight  = 1)
G.add_edge(1,3, weight  = 100)
G.add_edge(2,3, weight  = 100)

adj = nx.adjacency_matrix(G).todense()
edges = list(G.edges)

start = time.time()
results.append(qml.specs(circuit_4_mixers_benchmark)(init_params))
end = time.time()
durations.append(end-start)

# %% Plot the depth

x = np.arange(2, 9)
y = [2**(i-1) * (i-1) * (i-1) + (i * (i-1) / 2 + 1) * (i-1)       for i in x]
y2 = [ ( 2**(i) * (i) * (i-1) * i / 2) * (4* i+1)        for i in x]


fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(2,1,1)
ax.set_title('QAOA Computational Depth ')
ax.set_xlabel('Graph Size (# of vertices)')
ax.set_ylabel('Depth of the Circuit')
ax.legend(bbox_to_anchor=(1.05, 1), loc='lower right', borderaxespad=0.)
line, = ax.semilogy(x, y, color='blue', lw=1, label = 'QAOA-E1')
line2, = ax.semilogy(x, y2, color='red', lw=1, label = 'QAOA-M1')
plt.legend()
plt.show()

    


    plt.title(title, fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    for idx, evals in enumerate(evals):
        ax.plot(evals, label= labels[idx])
        
    # Create a legend for the first line.
    
    # Add the legend manually to the Axes.
    
    