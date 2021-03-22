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
from qCircuit import EmbedNet

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

model_saved = True


# %%
def Fidelity_loss(mes):
    tot  =0
    for i in mes:
        
        tot += i[0]
    fidelity = (2 * (tot) / len(mes[0])  - 1.00)
    return torch.log(1- fidelity)
    # return torch.log(1-mes)


# %% 
 
dev_embed = qml.device("default.qubit", wires=4+2+2,shots = 1000)
embed_model = EmbedNet(dev_embed, 1, 4, 2, 1)

learning_rate = 0.018
epochs = 10
loss_list = []

loss_func = Fidelity_loss
opt = torch.optim.Adam(embed_model.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# %% 
n_embed_samples = 2000
embed_features = np.random.rand(n_embed_samples,2)* np.pi
if(model_saved):
    embed_features  =np.load('features_temp.npy')
else: 
    np.save('embed_features.npy' ,embed_features)    
embed_features = torch.Tensor(embed_features)


# %% 
def time_evolution_simulate(i, t):
    return sp.linalg.expm(hamiltonians[i] * -1j * t)
def find_rx_params(cur_q_val):
    amp_res = (cur_q_val[0] + cur_q_val[1])
    amp_res /= cur_q_val[3] + cur_q_val[2]
    
    amp_res2 = (cur_q_val[0] + cur_q_val[2])
    amp_res2 /= cur_q_val[1] + cur_q_val[3]
    
    amp_res = np.abs(amp_res)    
    amp_res2 = np.abs(amp_res2)
    a2 = 1 / (1 + (amp_res) ** 2)
    b2 = 1 / (1 + (amp_res2) ** 2)
    a1 = 1- a2
    b1 = 1-b2
    return np.arccos(np.sqrt(a1)) * 2 , np.arccos(np.sqrt(b1)) * 2

pauli_list = [I,pauli_x, pauli_y, pauli_z]
observables = [np.kron(i,j) for i in pauli_list for j in pauli_list]
import numpy as np
hamiltonians = []
for i in range(500):
    coefs = 2.5 * np.random.randn(16) + 0.5
    ham = coefs[0] * observables[0]
    for i in range(1,16):
        ham += coefs[i] * observables[i] 
    hamiltonians.append(ham)
amps_d = qml.device("default.qubit", wires=2,shots = 1000)
@qml.qnode(amps_d)
def circ_amp(inputs):
    qml.templates.embeddings.AngleEmbedding(inputs,wires = range(0,2), rotation = 'X')
    return qml.probs([0,1])

q_values =  [] 
for i in range(500):
    circ_amp(embed_features[i])
    q_values.append(amps_d._state)
import scipy as sp
x_embed_params = []    
x2_embed_params = [] 

for i in range(500):
    cur_q_val = q_values[i].reshape(4,)
    
    U_t1 = time_evolution_simulate(i, 1)
    U_t2 = time_evolution_simulate(i, 2)
    U_t3 = time_evolution_simulate(i, 3)
    
    cur_q_val_t1 = U_t1 @ cur_q_val
    cur_q_val_t2 = U_t2 @ cur_q_val
    cur_q_val_t3 = U_t3 @ cur_q_val
    
    rx_1_t1, rx_2_t1 =  find_rx_params(cur_q_val_t1)
    rx_1_t2, rx_2_t2 = find_rx_params(cur_q_val_t2)
    rx_1_t3, rx_2_t3 = find_rx_params(cur_q_val_t3)
    
    x_embed_params.append(rx_1_t1)
    x2_embed_params.append(rx_2_t1)
    x_embed_params.append(rx_1_t2)
    x2_embed_params.append(rx_2_t2)
    x_embed_params.append(rx_1_t3)
    x2_embed_params.append(rx_2_t3)




x_params = torch.from_numpy(np.array(x_embed_params))
x2_params = torch.from_numpy(np.array(x2_embed_params))

embed_features_hams = torch.stack((x_params,x2_params)).T

embed_features[500:] = embed_features_hams

amps_d = qml.device("default.qubit", wires=2,shots = 1000)
@qml.qnode(amps_d)
def circ_amp(inputs):
    qml.templates.embeddings.AngleEmbedding(inputs,wires = range(0,2), rotation = 'X')
    return qml.probs([0,1])

time_evolution_simulate(47,1) @ 

circ_amp(embed_features[47].detach().numpy())

circ_amp(embed_features[47*3 + 500].detach().numpy())
# %%  Training for the Autoencoder   
model_saved = False
if(model_saved == False):
    batch_id = np.arange(n_embed_samples)
    np.random.shuffle(batch_id)
    
    for epoch in range(epochs):
        total_loss = []
        start_time = timeit.time.time()
        for i in batch_id:
            opt.zero_grad()
    
            normalized = embed_features[i]
            
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
# %% fidelities plot 
batch_id = np.arange(n_embed_samples)
np.random.shuffle(batch_id)
losses = np.zeros((n_embed_samples,1))
for i in batch_id:
    normalized = embed_features[i]
    out = embed_model(normalized,True)
    losses[i] = loss_func(out).detach().numpy()
import matplotlib
import matplotlib.pyplot as plt
w = 4
h = 3
d = 70
plt.figure(figsize=(w, h), dpi=d)
plt.hist(losses, bins=np.arange(0,0.5, 0.00525))

# %%
first_rot_params_0 = opt.param_groups[0]['params'][0][0][0]
first_rot_params_1 = opt.param_groups[0]['params'][0][0][1]

crot_params_01 = opt.param_groups[0]['params'][1][0]
crot_params_10 = opt.param_groups[0]['params'][1][1]

final_rot_params_0 = opt.param_groups[0]['params'][0][1][0]
final_rot_params_1 = opt.param_groups[0]['params'][0][1][1]

# %% SAVE AND LOAD THE MODULE 
PATH = './autoencoder.npy'



if(model_saved == True):
    embed_model.load_state_dict(torch.load(PATH))
else:
    torch.save(embed_model.state_dict(), PATH)
print('model parameters: ',  embed_model.state_dict)    


# %% 
amp_dev = qml.device("default.qubit", wires=4,shots = 1000)
@qml.qnode(amp_dev)
def get_amps(count, inputs = False):
    qml.templates.AngleEmbedding(inputs, wires = range(0,count), rotation = 'X' )
    return qml.probs(range(0,count))

# %% 
succesfull_train_sample_index = []

# Checks if the encode/decode phase is well 
for i in range(500):
    if((embed_model(embed_features[i])[0][0].item()) >= 0.985 and (embed_model(embed_features[i*3 +500])[0][0].item()) >= 0.985 and (embed_model(embed_features[i*3+501])[0][0].item()) >= 0.985  ):
        succesfull_train_sample_index.append(i)

        
initial_succesfull_train_sample_index = deepcopy(succesfull_train_sample_index)





len(initial_succesfull_train_sample_index)
# np.save('initial_succesfull_train_sample_index.npy' , initial_succesfull_train_sample_index)
# %% 


succesfull_train_samples_all = deepcopy(embed_features[succesfull_train_sample_index])



# Their decoded versions are also quite similar

succesfull_train_sample_index = []
succesfull_train_samples_latent = torch.zeros_like(succesfull_train_samples_all)
for i in range(len( succesfull_train_samples_all)):
    succesfull_train_samples_latent[i] = embed_model(succesfull_train_samples_all[i],training_mode = False, return_latent = True)
    similarity = embed_model(succesfull_train_samples_all[i],training_mode = False) - embed_model(succesfull_train_samples_latent[i].detach(), run_latent = True)    
    if (((torch.sum(similarity**2)) / 4) < 0.01):
        succesfull_train_sample_index.append(i)
        print(i)
input_indices = int(len(succesfull_train_sample_index)/3)
    
succesfull_train_samples_input = deepcopy(succesfull_train_samples_all[succesfull_train_sample_index[:input_indices]])
succesfull_train_samples_evolved = deepcopy(succesfull_train_samples_all[succesfull_train_sample_index[input_indices:]])    


# %%  Check data performs well overall, and save them
inp_ind  = np.random.randint(input_indices)
# Real amplitudes
torch.from_numpy(get_amps(2 , (succesfull_train_samples_input[inp_ind].detach().numpy() ) )) 
# Decoder results
embed_model(succesfull_train_samples_input[inp_ind].detach(),False) 
# Decoded output only using their latent output
embed_model(embed_model(succesfull_train_samples_all[inp_ind],training_mode = False, return_latent = True).detach(), run_latent = True)    

# Save the succesful data points
np.save('succesfull_train_samples_input.npy' ,succesfull_train_samples_input)
np.save('succesfull_train_samples_evolved.npy' ,succesfull_train_samples_evolved)


# Find the latents for the data
succesfull_train_samples_input_latent = torch.zeros_like(succesfull_train_samples_input)
succesfull_train_samples_evolved_latent = torch.zeros_like(succesfull_train_samples_evolved)

for i in range(len( succesfull_train_samples_input )):
    succesfull_train_samples_input_latent[i] = embed_model(succesfull_train_samples_input[i],training_mode = False, return_latent = True)
succesfull_train_samples_input_latent = succesfull_train_samples_input_latent.detach().numpy()

for i in range(len( succesfull_train_samples_evolved )):
    succesfull_train_samples_evolved_latent[i] = embed_model(succesfull_train_samples_evolved[i],training_mode = False, return_latent = True)
    
succesfull_train_samples_input_latent = succesfull_train_samples_input_latent
succesfull_train_samples_evolved_latent = succesfull_train_samples_evolved_latent

# Save the latent vectors
np.save('succesfull_train_samples_input_latent.npy' ,succesfull_train_samples_input_latent)
np.save('succesfull_train_samples_evolved_latent.npy' ,succesfull_train_samples_evolved_latent.detach().numpy())

# %%  This part is implemented to extract some qubits, which can be 
# succesfully d/encoded. Thus, we make sure that if the second network in the
# pipeline works, whole system should work



succesfull_train_samples_input = np.load('succesfull_train_samples_input.npy')
succesfull_train_samples_evolved = np.load('succesfull_train_samples_evolved.npy' )
succesfull_train_samples_input_latent = np.load('succesfull_train_samples_input_latent.npy')
succesfull_train_samples_evolved_latent = np.load('succesfull_train_samples_evolved_latent.npy')
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
        self.I  = tf.Variable(
                initial_value=I,
                trainable=False,
                dtype = tf.complex128
            )  
        self.coefs = tf.Variable(
                initial_value=[1,1,1,1,1,1,1,1,1,1,1,1],
                trainable=True,
                dtype = tf.complex128
            )  
        self.zerotensor  = tf.Variable(
                initial_value=0,
                trainable=False,
                dtype = tf.float64,
                name = 'dz'
            )
        self.paulis = [I, self.pauli_x, self.pauli_y,self.pauli_z]
        phasor_qml = qml.device("default.qubit", wires=2,shots = 1000)
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
        hamilton = [tf.contrib.kfac.utils.kronecker_product(i,j)  for i in self.paulis for j in self.paulis]
        
        
        
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
