# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:01:43 2020

@author: burak
"""





# %%

# TASK 1-  Try different embedding (1 to 1 )

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
sys.path.append('../PennyLane')
sys.path.append('../representation')
from IrisLoad import importIRIS, IRISDataset
from MNISTLoad import importMNIST
from qCircuit import Net, latNet, genNet

from visualizations import visualize, visualize_state_vec
from copy import deepcopy
import seaborn
import tensorflow as tf

#%% numpy implementations for the gates


I  = np.array(((0j + 1, 0), (0, 1)))
pauli_x  = np.array(((0j, 1), (1, 0)))
pauli_y = np.array(((0, -1j), (1j, 0)))
pauli_z = np.array(((1+0j, 0), (0, -1)))

# %%
def Fidelity_loss(mes):
    tot  =0
    for i in mes:
        
        tot += i[0]
    fidelity = (2 * (tot) / len(mes[0])  - 1.00)
    return torch.log(1- fidelity)

# %%
    '''
n_qubits = 2
dev2 = qml.device("default.qubit", wires=4,shots = 1000)
model = Net(dev2, 1, 4, 2, 1)

# new_model = Net(dev2, 1, 4, 2, 1)
# Import the autoencoder model, which is trained
model.load_state_dict(torch.load('21jan.pth.tar')) 


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
    
    test_qubits.append(torch.from_numpy(np.kron(qubit_one,qubit_two))) ''' 
# %% 
dev_embed = qml.device("default.qubit", wires=6,shots = 1000)   
embed_model = EmbedNet(dev_embed, 1, 6, 4, 1)
learning_rate = 0.1
epochs = 20
loss_list = []

loss_func = Fidelity_loss
opt = torch.optim.Adam(embed_model.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
n_embed_samples = 10
embed_features = np.random.rand(n_embed_samples,4)* np.pi
embed_features = torch.Tensor(embed_features)
embed_model(embed_features[0], True)


# %%    
    
batch_id = np.arange(n_embed_samples)
np.random.shuffle(batch_id)

for epoch in range(epochs):
    total_loss = []
    start_time = timeit.time.time()
    for i in batch_id:
        opt.zero_grad()
        # for iris dataseti
        # data = datas['data']
        
        #data  = train_qubits[i]
        normalized = embed_features[i]
        # They do not have to be normalized since AmplitudeEmbeddings does that
        # But we do it anyways for visualization
        
        # normalized = np.abs(nn.functional.normalize((data).view(1,-1)).numpy()).reshape(-1)
        # normalized = torch.Tensor(normalized).view(1,-1)


        # if(pad['padding_op']):
            
        #     new_arg = torch.cat((normalized[0], pad['pad_tensor']), dim=0)    
        #     normalized = torch.Tensor(new_arg).view(1,-1)
        
        
        out = embed_model(normalized,True)
        
        loss = loss_func(out)
        loss.backward()
        
        if(i%100 == 0):
            print(out)
        opt.step()
        
    
        total_loss.append(loss.item())
    end_time = timeit.time.time()
    print('Time elapsed for the epoch' +  str(epoch)  + ' : {:.2f}'.format(end_time-start_time))
    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(100. * (epoch + 1) / epochs, loss_list[-1]))    

# %% 
model(data.view(1,-1),False,False,False)
embed_model(normalized,False,False,False)
dev3 = qml.device("default.qubit", wires=4,shots = 1000)
dev4 = qml.device("default.qubit", wires=4,shots = 1000)
paramserver = model.paramServer()
latModel = latNet(dev3, 1,4,2,paramserver,1)
genModel = genNet(dev4, 1,4,2,paramserver,1)



# %%


total_losses = np.zeros((50) , float)
# %%
# batches = np.arange(n_train_samples)
# total_loss = []
# total_losses = np.zeros(n_train_samples)
# with torch.no_grad():
#     correct = 0     
#     for i in range(n_train_samples):
#     # for i in new_batches:
#         data = train_qubits[i]
        
#         normalized = np.abs(nn.functional.normalize((data).view(1,-1)).numpy()).reshape(-1)
#         normalized = torch.Tensor(normalized).view(1,-1)
        
    
    
#         output = model(data.view(1,-1), training_mode = False, return_latent =False) 
        
#         output2 = model(data.view(1,-1), training_mode = True) 

#         print(output2)
#         output = model(data.view(1,-1), training_mode = False, return_latent =False) 
#         # lat_out = latModel(data.view(1,-1))
#         # gen_out = genModel(np.sqrt(lat_out.detach()))
        
#         # visualize_state_vec(output.detach() , 'output ' + str(i) , 2)
#         # visualize_state_vec(normalized**2, 'data ' + str(i),2)
        
#         total_losses[i] = loss_func(output2)
        
# new_batches = np.argsort(total_losses)[:25]
model(data.view(1,-1), training_mode = False, return_latent =False) 

# %% Extracting the parameters, to select suitable training data

# Retrieve fine-tuned model parameters
first_rots , final_rots ,cnots ,wires_list = model.paramServer()
first_rots , final_rots ,cnots ,wires_list = embed_model.paramServer()
# %%
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram, plot_bloch_vector
from math import sqrt, pi
import qiskit
# %%  This part is implemented to extract some qubits, which can be 
# succesfully d/encoded. Thus, we make sure that if the second network in the
# pipeline works, whole system should work

dev_qml = qml.device("default.qubit", wires=1,shots = 1000)
lat_dev_qml = qml.device("default.qubit", wires=3,shots = 1000)
gen_qml = qml.device("default.qubit", wires=3,shots = 1000)
lat_loss = np.zeros((200) , float)
output_loss = np.zeros((200,4) , float)


@qml.qnode(dev_qml)
def penny(fir,cnot,last,inputs= False):
    print(inputs)
    # Get the encoded qubits
    #qml.templates.AmplitudeEmbedding(inputs, wires = [1,2], normalize = True,pad=(0.j))
    qml.templates.embeddings.AngleEmbedding(inputs,wires = [1,2], rotation = 'X')
    #qml.templates.embeddings.AngleEmbedding(inputs,wires = [1,2], rotation = 'X')
    
    qml.QubitUnitary(np.matrix(fir[0]), wires = 1)
    qml.QubitUnitary(np.matrix(fir[1]), wires = 2)
    
    qml.QubitUnitary(np.matrix(cnot[0]), wires = [1,2])
    qml.QubitUnitary(np.matrix(cnot[1]), wires = [2,1])
    
    qml.QubitUnitary(np.matrix(last[0]), wires = 1)
    qml.QubitUnitary(np.matrix(last[1]), wires = 2)
    
    return qml.probs([1]) , qml.probs([2]) 

@qml.qnode(lat_dev_qml)
def generate_penny(inputs):
    # Get Reconstructed output
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
    # Generate data from latent
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
    
features = tf.convert_to_tensor(selected_features[:7] , dtype_hint=tf.complex128)
latents = tf.convert_to_tensor(selected_latent_features[:7]  , dtype_hint=tf.complex128)

targets = tf.convert_to_tensor(selected_features[7:] , dtype_hint=tf.complex128)
target_latents = tf.convert_to_tensor(selected_latent_features[7:] , dtype_hint=tf.complex128)



np.save('features.npy', features)
np.save('latents.npy', latents)
np.save('targets.npy', targets)
np.save('target_latents.npy', target_latents)
# %% 

#data generation for basis embedding

penny([3], first_rots ,cnots, final_rots)


# %% 
# Get hand-tailored data, which are proven to be working with autoencoders

features = np.load('features.npy') 
latents = np.load('latents.npy')
targets = np.load('targets.npy' )
target_latents = np.load('target_latents.npy')

gen_penny(penny(features[-4] , first_rots, cnots, final_rots)[1])
features[-4].numpy() ** 2


generate_penny(features[-1])


# %% 



# dev = qml.device('default.qubit', wires=2 , shots = 10000)
class ExpMat(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.pauli_x  = tf.Variable(
                initial_value=pauli_x,
                trainable=False,
                dtype = tf.complex128
            )   
        self.pauli_y  = tf.Variable(
                initial_value=pauli_y,
                trainable=False,
                dtype = tf.complex128
            )   
        self.pauli_z  = tf.Variable(
                initial_value=pauli_z,
                trainable=False,
                dtype = tf.complex128
            )  

        
        self.b  = tf.Variable(
                initial_value=2,
                trainable=True,
                dtype = tf.complex128,
                name = 'b'
            )

        self.c  = tf.Variable(
                initial_value=3,
                trainable=True,
                dtype = tf.complex128,
                name = 'c'
            )
        self.d  = tf.Variable(
                initial_value=1,
                trainable=True,
                dtype =tf.complex128,
                name = 'd'
            )

        self.zerotensor  = tf.Variable(
                initial_value=0,
                trainable=False,
                dtype = tf.float64,
                name = 'dz'
            )
        phasor_qml = qml.device("default.qubit", wires=1,shots = 1000)
        @qml.qnode(phasor_qml)
        def get_probs(inputs):
            qml.templates.AmplitudeEmbedding(inputs, wires = [0], normalize = True,pad=(0.j))    
            return qml.probs([0])
        weight_shapes = {}
        self.qlayer = qml.qnn.KerasLayer(get_probs, weight_shapes, output_dim=2)
    def __call__(self, x):
        """returns a matrix multiplication
        
        
        Parameters
        ----------
        x : tensor
        input
        
        Returns
        -------
        tensor
        output
        
        """
        
        
        l = tf.linalg.expm(-1j * (self.pauli_x * self.b + self.pauli_y * self.c + self.pauli_z * self.d ) * 1)
        l2 = tf.linalg.expm(-1j * (self.pauli_x * self.b + self.pauli_y * self.c + self.pauli_z * self.d ) * 2)
        l3 = tf.linalg.expm(-1j * (self.pauli_x * self.b + self.pauli_y * self.c + self.pauli_z * self.d ) * 3)
        
        l = tf.linalg.matvec(l, x)
        l2 = tf.linalg.matvec(l2, x)
        l3 = tf.linalg.matvec(l3, x)
        
        return l,l2,l3
        # return self.qlayer((l[0] , l[1]))

   

# %%  Network to get rid of the phasor and getting the probabilities,
# TODO: Relocate it inside the ExpMat class
phasor_qml = qml.device("default.qubit", wires=2,shots = 1000)
@qml.qnode(phasor_qml)
def get_probs(inputs):
    qml.templates.AmplitudeEmbedding(inputs, wires = [0,1], normalize = True,pad=(0.j))    
    return qml.probs([0,1])


# %%  
expMat = ExpMat(name="expmat")
# %%
optimizer = tf.keras.optimizers.SGD(learning_rate=0.004)
loss_history = []

t_count = 1
def train_step(psi_in, psi_out, psi_out_t2, psi_out_t3 , H= 0 ):
    with tf.GradientTape() as tape:

        
        res = expMat(tf.math.sqrt(psi_in))
        # print(res)
        
        # loss_value = tf.reduce_sum(tf.pow( tf.math.real(res) - tf.math.real(psi_out),2))/(2) + tf.reduce_sum(tf.pow( tf.math.imag(res) - tf.math.imag(psi_out),2))/(2)
        loss_value = tf.pow(psi_out  - res[0] ,2)/(2) + tf.pow(psi_out_t2  - res[1] ,2)/(2)# + tf.reduce_sum(tf.pow(psi_out_t3  - res[2] ,2)) /(2) 
        print(loss_value)
        # print(tf.math.real(loss_value))
    
    loss_history.append(loss_value.numpy().mean())
    
    grads = tape.gradient(loss_value, expMat.trainable_variables)
    optimizer.apply_gradients(zip(grads,expMat.trainable_variables))
    # expMat.b.assign(tf.complex(tf.math.real(expMat.b), expMat.zerotensor))
    # expMat.c.assign(tf.complex(tf.math.real(expMat.c), expMat.zerotensor))
    # expMat.d.assign(tf.complex(tf.math.real(expMat.d), expMat.zerotensor))
    return loss_value

    
    

for j in range(10):

    for i in range(7):
    # for i in range(len(features)):
        loss_value_tf = train_step(latents[i], target_latents[i] , target_latents[i+len(features)] , target_latents[i+len(features) +len(features)])
    if(tf.math.real(loss_value_tf[0]) <= 0.00001):
        break
        
psi_in  = latents[i]
psi_out = target_latents[i+14]




# %% Creating the dataset pt2: hamiltonians
# Hamilton class, gets an input psi_in, and creates 3 output with t=1,2,3
# Then tries to find the suitable circuit for all.
# After finding the circuit, by taking the logarithm with proper t values
# One could find the hamiltonian




class Hamilton(nn.Module):
    def __init__(self,w_s):
    
        super(Hamilton, self).__init__()
        
        
        
        self.dev1 = qml.device('default.qubit', wires=6 , shots = 1000)

        
        @qml.qnode(self.dev1)
        def q_circuit(weights_r , weights_cr  ,inputs = False):
            
            self.t_level_embedding(inputs)
            self.a = qml.Rot(*weights_r[0, 0], wires = 0)
            self.b = qml.Rot(*weights_r[0, 1], wires = 1)
            
            self.c = qml.CRot(*weights_cr[0, 0], wires = [0,1])
            self.d = qml.CRot(*weights_cr[0, 1], wires = [1, 0])
            
            self.e = qml.Rot(*weights_r[1, 0], wires = 0)
            self.f = qml.Rot(*weights_r[1, 1], wires = 1)
            
            
            
            qml.Rot(*weights_r[0, 0], wires = 2)
            qml.Rot(*weights_r[0, 1], wires = 3)
            
            qml.CRot(*weights_cr[0, 0], wires = [2,3])
            qml.CRot(*weights_cr[0, 1], wires = [3, 2])
            
            qml.Rot(*weights_r[1, 0], wires = 2)
            qml.Rot(*weights_r[1, 1], wires = 3)
            
            qml.Rot(*weights_r[0, 0], wires = 2)
            qml.Rot(*weights_r[0, 1], wires = 3)
            
            qml.CRot(*weights_cr[0, 0], wires = [2,3])
            qml.CRot(*weights_cr[0, 1], wires = [3, 2])
            
            qml.Rot(*weights_r[1, 0], wires = 2)
            qml.Rot(*weights_r[1, 1], wires = 3)
            
            
            qml.Rot(*weights_r[0, 0], wires = 4)
            qml.Rot(*weights_r[0, 1], wires = 5)
            
            qml.CRot(*weights_cr[0, 0], wires = [4,5])
            qml.CRot(*weights_cr[0, 1], wires = [5,4])
            
            qml.Rot(*weights_r[1, 0], wires = 4)
            qml.Rot(*weights_r[1, 1], wires = 5)
            
            qml.Rot(*weights_r[0, 0], wires = 4)
            qml.Rot(*weights_r[0, 1], wires = 5)
            
            qml.CRot(*weights_cr[0, 0], wires = [4,5])
            qml.CRot(*weights_cr[0, 1], wires = [5,4])
            
            qml.Rot(*weights_r[1, 0], wires = 4)
            qml.Rot(*weights_r[1, 1], wires = 5)
            
            qml.Rot(*weights_r[0, 0], wires = 4)
            qml.Rot(*weights_r[0, 1], wires = 5)
            
            qml.CRot(*weights_cr[0, 0], wires = [4,5])
            qml.CRot(*weights_cr[0, 1], wires = [5,4])
            
            qml.Rot(*weights_r[1, 0], wires = 4)
            qml.Rot(*weights_r[1, 1], wires = 5)
            
            return qml.probs([0,1,2,3,4,5])


        
        self.qlayer = qml.qnn.TorchLayer(q_circuit, w_s)



    @qml.template
    def embedding(self,inputs):
        qml.templates.AmplitudeEmbedding(inputs[:2], wires = range(0,2), normalize = True,pad=(0.j))
    @qml.template
    def t_level_embedding(self,inputs):
        qml.templates.AmplitudeEmbedding(inputs, wires = range(0,6), normalize = True,pad=(0.j))
        

    def forward(self, x):
        
        return self.qlayer(x)


    


# %%
learning_rate = 0.01
loss_list_ham = []

def L2_loss(target,data):
    return torch.sum((target - data)**2)/4
loss_func_ham = L2_loss

loss_hist_ham = []

w_s = {"weights_r": (2 , 2, 3),"weights_cr": (1,2 ,3)}

hamNet = Hamilton(w_s)
opt_ham = torch.optim.Adam(hamNet.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

hamNets = []
hamiltons = []

for i in range(7):
    hamNets.append(Hamilton(w_s))


hamiltons = []
for i in range(7):
    
    hamNet = hamNets[i]
    opt_ham = torch.optim.Adam(hamNet.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    for j in range(30):
        
        opt_ham.zero_grad()
        
        # data = torch.Tensor(tf.cast(features[i] , dtype = tf.float64).numpy())
        # data = torch.Tensor(np.kron(tf.cast(features[i] , dtype = tf.float64).numpy() , tf.cast(features[i] , dtype = tf.float64).numpy() ))
        data = torch.Tensor(np.kron( tf.cast(features[i] , dtype = tf.float64).numpy() , np.kron(tf.cast(features[i] , dtype = tf.float64).numpy() , tf.cast(features[i] , dtype = tf.float64).numpy() )))
        
        target = (torch.Tensor(tf.cast(targets[i] , dtype = tf.float64).numpy())) ** 2
        target2 = (torch.Tensor(tf.cast(targets[i+7] , dtype = tf.float64).numpy())) ** 2
        target3 = (torch.Tensor(tf.cast(targets[i+14] , dtype = tf.float64).numpy())) ** 2
        
        
        out = hamNet(data)
        two_l_target = torch.Tensor(np.kron(np.kron(target, target2) , target3))
        
        loss = loss_func_ham(two_l_target,out) # + loss_func_ham(target2,out[1])  #+ loss_func_ham(target3,out[2]) #+ loss_func_ham(target4,out[3]) 
        
        loss.backward()
        
        if(j%6 == 0):
            print(loss)
        
        loss_hist_ham.append(loss)
        opt_ham.step()
    hUf = deepcopy(np.kron(hamNet.a.matrix , hamNet.b.matrix))
    hUc1 = deepcopy(hamNet.c.matrix)
    hUc2t = deepcopy(hamNet.d.matrix)
    
    
    cnotss = np.copy( hUc2t )
    cnotss[1,1] = np.copy( hUc2t[2,2] )
    cnotss[1,3] = np.copy( hUc2t[2,3] )
    cnotss[3,1] = np.copy( hUc2t[3,2] )
    cnotss[3,2] = 0
    cnotss[2,3] = 0
    cnotss[2,2] = 1
    
    hUfin = deepcopy(np.kron(hamNet.e.matrix , hamNet.f.matrix))
    
    newU = hUfin @ cnotss @ hUc1 @ hUf
    hamiltons.append(newU)
    # hamNets.append(hamNet)
    print('loss is :',  get_probs(hamiltons[i] @ features[i].numpy()))

# %% 
for h in range(len(hamNets)):
    torch.save(hamNets[h].state_dict() ,'circ' + str(h) + '.pth.tar')
    np.save('circ' + str(h) + '.npy', hamiltons[h])


hamiltonian_matrices = []  # they are hermitians, refer to the hamiltonian of the system
for h in range(len(hamNets)):
    hamiltonian_matrices.append( logm(np.array(hamiltons[h])) * 1j )

# we found the hamiltonians !!

get_probs(expm(-2j* hamiltonian_matrices[0]) @ features[0].numpy())
targets[7] ** 2

load_hamilton_data(hamiltons,hamNets)
hamNets[0].load_state_dict(torch.load('circ' + str(0) + '.pth.tar'))

# %% Loading hamiltonian info:
for i in range(len(hamNets)):
    hamNets[i].load_state_dict(torch.load('circ' + str(i) + '.pth.tar'))
    hamiltons.append( np.load('circ' + str(i) + '.npy'))
    
# def load_hamilton_data(hamilton, hamNets):
#     for i in range(len(hamNets)):
#         hamNets[i].load_state_dict(torch.load('circ' + str(i) + '.pth.tar'))
#         hamiltons.append( np.load('circ' + str(i) + '.npy'))
#     print(hamNets)
    
     


# %% bring all together
from scipy.linalg import expm, sinm, cosm,logm
for i in range(7):
    print(get_probs(expm(-1j* hamiltonian_matrices[i] * 2) @ features[i].numpy()) - (targets[i+7]**2).numpy())
    # print(get_probs( hamiltonians[i] @ features[i].numpy()) - (targets[i]**2).numpy())

decoded_original_qubit = (np.sqrt(gen_penny(  penny(features[i].numpy() , first_rots, cnots, final_rots)[1])))
for i in range(7):
    print(get_probs(expm(-1j* hamiltonian_matrices[i] * 2) @ (np.sqrt(gen_penny(  penny(features[i].numpy() , first_rots, cnots, final_rots)[1])))) - (targets[i+7]**2).numpy())
    
for i in range(7):
    print(get_probs(expm(-1j* hamiltonian_matrices[i] * 2) @ (np.sqrt(gen_penny(latents[i].numpy() )))) - (targets[i+7]**2).numpy())
    
    
    print(features[i].numpy())
