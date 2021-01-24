# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:01:43 2020

@author: burak
"""

# %% 
import numpy as np
import torch
from torch.autograd import Function

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import pennylane as qml




import timeit
from torch.utils.tensorboard import SummaryWriter


import qiskit
import torchvision


from torch.utils.data import Dataset, DataLoader
import sys

sys.path.append('../dataLoad')
sys.path.append('C:/Users/burak/OneDrive/Desktop/Quantum Machine Learning/CodeBank/QuantumNeuralNet/dataLoad')
sys.path.append('C:/Users/burak/OneDrive/Desktop/Quantum Machine Learning/CodeBank/QuantumNeuralNet/pennyLane')
sys.path.append('C:/Users/burak/OneDrive/Desktop/Quantum Machine Learning/CodeBank/QuantumNeuralNet/representation')
sys.path.append('../pennyLane')
sys.path.append('../representation')
from IrisLoad import importIRIS, IRISDataset
from MNISTLoad import importMNIST
from qCircuit import Net, latNet, genNet

from visualizations import visualize, visualize_state_vec
from copy import deepcopy
import seaborn

# %%
def Fidelity_loss(mes):
    tot  =0
    for i in mes[0]:
        tot += i[0]
    fidelity = (2 * (tot) / len(mes[0])  - 1.00)
    return torch.log(1- fidelity)

n_qubits = 2
dev2 = qml.device("default.qubit", wires=4,shots = 1000)
model = Net(dev2, 1, 4, 2, 1)

# new_model = Net(dev2, 1, 4, 2, 1)
model.load_state_dict(torch.load('checkpoint.pth.tar'))

learning_rate = 0.01
epochs = 5
loss_list = []
opt = torch.optim.Adam(model.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
loss_func = Fidelity_loss
test_loss = nn.MSELoss()
train_qubits = []
n_train_samples = 200
for i in range(n_train_samples):
    zero_amp  = np.random.rand()
    one_amp = np.sqrt(1 - (zero_amp ** 2))
    zero_amp_two  = np.random.rand()
    one_amp_two = np.sqrt(1 - (zero_amp_two ** 2))
    
    qubit_one = np.array([zero_amp, one_amp])
    qubit_two = np.array([zero_amp_two, one_amp_two])
    
    train_qubits.append(torch.from_numpy(np.kron(qubit_one,qubit_two)) )

test_qubits = []
n_test_samples = 100

for i in range(n_test_samples):
    zero_amp  = np.random.rand()
    one_amp = np.sqrt(1 - (zero_amp ** 2))
    zero_amp_two  = np.random.rand()
    one_amp_two = np.sqrt(1 - (zero_amp_two ** 2))
    
    qubit_one = np.array([zero_amp, one_amp] )
    qubit_two = np.array([zero_amp_two, one_amp_two] )
    
    test_qubits.append(torch.from_numpy(np.kron(qubit_one,qubit_two)))
    


# %%    
    
batch_id = np.arange(n_train_samples)
np.random.shuffle(batch_id)

for epoch in range(epochs):
    total_loss = []
    start_time = timeit.time.time()
    for i in batch_id:
        opt.zero_grad()
        # for iris dataseti
        # data = datas['data']
        
        data  = train_qubits[i]
        
        # They do not have to be normalized since AmplitudeEmbeddings does that
        # But we do it anyways for visualization
        
        normalized = np.abs(nn.functional.normalize((data).view(1,-1)).numpy()).reshape(-1)
        normalized = torch.Tensor(normalized).view(1,-1)


        # if(pad['padding_op']):
            
        #     new_arg = torch.cat((normalized[0], pad['pad_tensor']), dim=0)    
        #     normalized = torch.Tensor(new_arg).view(1,-1)
        
        
        out = model(normalized,True)
        
        loss = loss_func(out)
        loss.backward()
        
        if(i%10 == 0):
            print(out)
        opt.step()
        
    
        total_loss.append(loss.item())
    end_time = timeit.time.time()
    print('Time elapsed for the epoch' +  str(epoch)  + ' : {:.2f}'.format(end_time-start_time))
    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(100. * (epoch + 1) / epochs, loss_list[-1]))    

# %% 
model(data.view(1,-1),False,False,False)
dev3 = qml.device("default.qubit", wires=4,shots = 1000)
dev4 = qml.device("default.qubit", wires=4,shots = 1000)
paramserver = model.paramServer()
latModel = latNet(dev3, 1,4,2,paramserver,1)
genModel = genNet(dev4, 1,4,2,paramserver,1)


# %% 

def qubitsToStateVectorNew(vec, n):
    a = np.sum(vec[0,:(2**(n-1))].numpy())
    
    if(n == 3):
        b=  (vec[0,0] + vec[0,1] + vec[0,4] + vec[0,5]).numpy()
    elif(n==4):
        b=  (vec[0,0] + vec[0,1] + vec[0,2] +  vec[0,3]  + vec[0,8] + vec[0,9] + vec[0,10] + vec[0,11] ).numpy()
    
    
    if(n==3):
        c  = np.sum(vec[0,::2].numpy())
    else:
        c =  (vec[0,0] + vec[0,1] + vec[0,4] +  vec[0,5]  + vec[0,8] + vec[0,9] + vec[0,12] + vec[0,13] ).numpy()
        

    d = np.sum(vec[0,::2].numpy())
    s0 = np.zeros(2)
    s0[0] = 1
    
    s1 = np.zeros(2)
    s1[1] = 1
    
    lat1 = s0 * np.sqrt(a)  +s1 * np.sqrt((1-a))
    lat2 = s0 * np.sqrt(b)  +s1 * np.sqrt((1-b))
    lat3 = s0 * np.sqrt(c)  +s1 * np.sqrt((1-c))
    lat4 = s0 * np.sqrt(d)  +s1 * np.sqrt((1-d))
    if  ( n== 3):
        return lat1,lat2,lat3
    else:
        return lat1,lat2,lat3,lat4

# %% 


def qubitsToStateVector(vec, n):
        
    s0 = np.zeros(2)
    s0[0] = 1
    
    s1 = np.zeros(2)
    s1[1] = 1
    vals = ((vec + 1 )/2)
    state_1_vals = 1- deepcopy(vals)
    
    vals = np.sqrt(vals)
    state_1_vals = np.sqrt(state_1_vals)
    
    lat1 = s0 * vals[0][0].numpy()  +s1 * (state_1_vals[0][0].numpy())
    lat2 = s0 * vals[0][1].numpy() + s1 * (state_1_vals[0][1].numpy())
    lat3 = s0 * vals[0][2].numpy() + s1 * (state_1_vals[0][2].numpy())
    if(n == 4):
        lat4 = s0 * vals[0][3].numpy() + s1 * (state_1_vals[0][3].numpy())
        return lat1,lat2,lat3,lat4
    else:
        return lat1,lat2,lat3

# %%
        
,
total_losses = np.zeros((50) , float)
# %%
batches = np.arange(n_train_samples)
total_loss = []
total_losses = np.zeros(n_train_samples)
with torch.no_grad():
    correct = 0     
    for i in range(n_train_samples):
    # for i in new_batches:
        data = train_qubits[i]
        
        normalized = np.abs(nn.functional.normalize((data).view(1,-1)).numpy()).reshape(-1)
        normalized = torch.Tensor(normalized).view(1,-1)
        
    
    
        output = model(data.view(1,-1), training_mode = False, return_latent =False) 
        
        output2 = model(data.view(1,-1), training_mode = True) 


        
        # lat_out = latModel(data.view(1,-1))
        # gen_out = genModel(np.sqrt(lat_out.detach()))
        
        # visualize_state_vec(output.detach() , 'output ' + str(i) , 2)
        # visualize_state_vec(normalized**2, 'data ' + str(i),2)
        
        total_losses[i] = loss_func(output2)
        
new_batches = np.argsort(total_losses)[:25]
# %%


with torch.no_grad():
    correct = 0     
    for i in new_batches:
    
        data = train_qubits[i]
        
        normalized = np.abs(nn.functional.normalize((data).view(1,-1)).numpy()).reshape(-1)
        normalized = torch.Tensor(normalized).view(1,-1)
        
    

        
        output2 = model(data.view(1,-1), training_mode = True)
        output = model(data.view(1,-1), training_mode = False, return_latent =False) 

        
        # lat_out = latModel(data.view(1,-1))
        # print(lat_out)
        # gen_out = genModel(np.sqrt(lat_out.detach()))
        
        # visualize_state_vec(output.detach() , 'output ' + str(i) , 2)
        # visualize_state_vec(normalized**2, 'data ' + str(i),2)
        
        # total_losses[i] = loss_func(output2)


# %% 
first_rots , final_rots ,cnots ,wires_list = model.paramServer()
# %%
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram, plot_bloch_vector
from math import sqrt, pi
import qiskit
# %% 
lat_dev_qml = qml.device("default.qubit", wires=3,shots = 1000)
gen_qml = qml.device("default.qubit", wires=3,shots = 1000)
lat_loss = np.zeros((200) , float)
output_loss = np.zeros((200,4) , float)
@qml.qnode(dev_qml)
def penny(inputs,fir,cnot,last):
    qml.templates.AmplitudeEmbedding(inputs, wires = [1,2], normalize = True,pad=(0.j))
    
    qml.QubitUnitary(np.matrix(fir[0]), wires = 1)
    qml.QubitUnitary(np.matrix(fir[1]), wires = 2)
    
    qml.QubitUnitary(np.matrix(cnot[0]), wires = [1,2])
    qml.QubitUnitary(np.matrix(cnot[1]), wires = [2,1])
    
    qml.QubitUnitary(np.matrix(last[0]), wires = 1)
    qml.QubitUnitary(np.matrix(last[1]), wires = 2)
    
    return qml.probs([1]) , qml.probs([2]) 

@qml.qnode(lat_dev_qml)
def generate_penny(inputs):
    qml.templates.AmplitudeEmbedding(inputs, wires = [1, 2], normalize = True,pad=(0.j))

    qml.QubitUnitary(first_rots[0] , wires = 1)
    qml.QubitUnitary(first_rots[1] , wires = 2)
    
    qml.QubitUnitary(np.matrix(cnots[0]), wires = [1,2])
    qml.QubitUnitary(np.matrix(cnots[1]), wires = [2,1])
    
    qml.QubitUnitary(np.matrix(final_rots[0]), wires = 1)
    qml.QubitUnitary(np.matrix(final_rots[1]), wires = 2)
    
    qml.SWAP(wires = [0,1])
    qml.QubitUnitary(np.matrix(final_rots[0]).H , wires = 1)
    qml.QubitUnitary(np.matrix(final_rots[1]).H, wires = 2)
    
    qml.QubitUnitary(np.matrix(cnots[1]).H, wires = [2,1])
    qml.QubitUnitary(np.matrix(cnots[0]).H, wires = [1,2])
    
    
    qml.QubitUnitary( np.matrix(first_rots[0]).H, wires = 1)
    qml.QubitUnitary(np.matrix(first_rots[1]).H , wires = 2)
    
    return qml.probs([1,2])

@qml.qnode(gen_qml)
def gen_penny(inputs):
    qml.templates.AmplitudeEmbedding(inputs, wires = [ 2], normalize = True,pad=(0.j))

    
    qml.QubitUnitary(np.matrix(final_rots[0]).H , wires = 1)
    qml.QubitUnitary(np.matrix(final_rots[1]).H, wires = 2)
    
    qml.QubitUnitary(np.matrix(cnots[1]).H, wires = [2,1])
    qml.QubitUnitary(np.matrix(cnots[0]).H, wires = [1,2])
    
    
    qml.QubitUnitary(np.matrix(first_rots[0]).H, wires = 1)
    qml.QubitUnitary(np.matrix(first_rots[1]).H , wires = 2)
    
    return qml.probs([1,2])
        
for i in range(n_train_samples):
    data = train_qubits[i]    
    lat_loss[i] = 1 - penny(data.detach().numpy(),first_rots, cnots, final_rots)[0][0]
    output_loss[i] = data - np.sqrt(gen_penny( penny(data.detach().numpy() , first_rots, cnots, final_rots) [1]))    

out_losses = np.sum(np.abs(output_loss) , axis = 1)/4 

x = np.argsort(out_losses)[:50]
y = np.argsort(lat_loss)[:50]

selected_qubits = []


for el in x:
    pres= False
    for j in y:
        if j == el:
            pres = True 
            break
    if(pres == True):
        selected_qubits.append(el)
        

selected_features = []
selected_latent_features = []
        
for q in (selected_qubits):
    selected_latent_features.append(penny(train_qubits[q].detach().numpy()   ,first_rots, cnots, final_rots)[1])
    selected_features.append(train_qubits[q].detach().numpy() )
    
features= tf.convert_to_tensor(selected_features)
latents = tf.convert_to_tensor(selected_latent_features)

# np.sqrt(gen_penny(penny(train_qubits[6].detach().numpy() ,first_rots , cnots, final_rots)[1]))

generate_penny.draw()
# np.argsort(lat_loss)

# selected_feature_qubits_actual = []
# selected_feature_qubits_latent = []
# selected_qubits = np.argsort(lat_loss)[:20]
# for i in selected_qubits:
    
# %%
'''
lat_loss = np.zeros((200) , float)

# for i in new_batches:
for i in range(n_train_samples):
    data = train_qubits[i]


    initial_state = data.detach().numpy()
    
    
    rot0 = qiskit.extensions.UnitaryGate(first_rots[0].H)
    rot1 = qiskit.extensions.UnitaryGate(first_rots[1].H)
    
    cnot2 = qiskit.extensions.UnitaryGate(cnots[0].H)
    cnot3 = qiskit.extensions.UnitaryGate(cnots[1].H)
    
    rot4 = qiskit.extensions.UnitaryGate(final_rots[0].H)
    rot5 = qiskit.extensions.UnitaryGate(final_rots[1].H)
    
    
    inv_rot0 = qiskit.extensions.UnitaryGate(final_rots[0])
    inv_rot1 = qiskit.extensions.UnitaryGate(final_rots[1])
    
    
    inv_cnot2 = qiskit.extensions.UnitaryGate(cnots[1])
    inv_cnot3 = qiskit.extensions.UnitaryGate(cnots[0])
    
    inv_rot4 = qiskit.extensions.UnitaryGate(first_rots[0])
    inv_rot5 = qiskit.extensions.UnitaryGate(first_rots[1])





    # qc = QuantumCircuit(n_qubits + 1)
    lat_qc = QuantumCircuit(n_qubits + 1) 

    # qc.initialize(initial_state, [1,2]) # Apply initialisation operation to the 0th qubit
    lat_qc.initialize(initial_state, [1,2])
    
    # qc.append(rot0, [1])
    # qc.append(rot1, [2])
    
    # qc.append(cnot2, [1,2])
    # qc.append(cnot3, [2,1])
    
    # qc.append(rot4, [1])
    # qc.append(rot5 , [2])
    
    
    lat_qc.append(rot0, [1])
    lat_qc.append(rot1, [2])
    
    lat_qc.append(cnot2, [1,2])
    lat_qc.append(cnot3, [2,1])
    
    lat_qc.append(rot4, [1])
    lat_qc.append(rot5 , [2])
    
    
    
    # qc.swap(0, 1)
    # lat_qc.swap(0, 1)
    
    # qc.append(inv_rot0, [1])
    # qc.append(inv_rot1, [2])
    
    # qc.append(inv_cnot2, [2,1])
    # qc.append(inv_cnot3, [1,2])
    
    # qc.append(inv_rot4, [1])
    # qc.append(inv_rot5 , [2])
    
 
    
    # qc.draw()
    backend = Aer.get_backend('statevector_simulator') # Tell Qiskit how to simulate our circuit
    
    # result = execute(qc,backend).result()
    lat_result = execute(lat_qc,backend).result()
    
    # res = result.get_statevector()
    lat_res = lat_result.get_statevector()
    
    # counts = result.get_counts()
    lat_counts = lat_result.get_counts()
    
    # sv = np.array([0.194695439185007,0.017163135438262,0.707821628904329,0.080319796472402,0,0,0,0])
    
    # dev_qml = qml.device("default.qubit", wires=3,shots = 1000)
    # @qml.qnode(dev_qml)
    # def qnode(inputs):
    #     qml.templates.AmplitudeEmbedding(inputs, wires = [2,1,0], normalize = True,pad=(0.j))
    #     return qml.probs([2,1])
    #     return qml.expval(qml.PauliZ(2)), qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliZ(0))
    
    lat_dev_qml = qml.device("default.qubit", wires=3,shots = 1000)
    @qml.qnode(lat_dev_qml)
    def lat_qnode(inputs):
        qml.templates.AmplitudeEmbedding(inputs, wires = [2,1,0], normalize = True,pad=(0.j))
        
        return qml.probs([1])
    print(np.round(lat_qnode(lat_res),3) , ' qiskit lat')
    #3. qubit yine 0 a denk
    print(np.round((data ** 2).numpy(), 3) , 'original data')
    print(np.round(qnode(res),3) , 'qiskit output')
    
    lat_out = latModel(data.view(1,-1))
    gen_out = genModel(np.sqrt(lat_out.detach()))
    print(np.round(lat_out.numpy(),3) , ' model lat') 
    print(np.round(lat_qnode(lat_res),3) , ' qiskit lat')
    
    lat_loss[i] = l(torch.Tensor(lat_qnode(lat_res)),lat_out.view(2))
    output2 = model(data.view(1,-1), training_mode = True) 
    print(np.round(output2.detach().numpy(),3) , 'model discarded qubit') 
    print('-')



selected_feature_qubits_actual = []
selected_feature_qubits_latent = []
selected_qubits = np.argsort(lat_loss)[:10]


for j in selected_qubits:
    data = train_qubits[j]
    initial_state = data.detach().numpy()
    lat_qc = QuantumCircuit(n_qubits + 1) 
    lat_qc.initialize(initial_state, [1,2])
    
    lat_qc.append(rot0, [1])
    lat_qc.append(rot1, [2])
    
    lat_qc.append(cnot2, [1,2])
    lat_qc.append(cnot3, [2,1])
    
    lat_qc.append(rot4, [1])
    lat_qc.append(rot5 , [2])
    
    backend = Aer.get_backend('statevector_simulator')     
    lat_result = execute(lat_qc,backend).result()
    lat_res = lat_result.get_statevector()
    
    selected_feature_qubits_actual.append(train_qubits[j])
    selected_feature_qubits_latent.append(lat_qnode(lat_res))
'''
                                          
# %%