# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 13:01:45 2020

@author: burak
"""


import numpy as np
 

import torch
from torch.autograd import Function

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import pennylane as qml
import qiskit




import timeit
from torch.utils.tensorboard import SummaryWriter



import torchvision

# %%



w  = torch.rand((2,2),dtype=torch.cfloat)


class myLinear(nn.Module):
    def __init__(self):
            super().__init__()

            #self.weight = torch.nn.Parameter(torch.randn((2,2), dtype= torch.cfloat))
            self.weight_11 = torch.nn.Parameter(torch.randn(1, dtype= torch.cfloat))
            self.weight_12 = torch.nn.Parameter(torch.randn(1, dtype= torch.cfloat))
            self.weight_21 = torch.nn.Parameter(torch.randn(1, dtype= torch.cfloat))
            self.weight_22 = torch.nn.Parameter(torch.randn(1, dtype= torch.cfloat))
            
    def forward(self, input):
        
        
        #output = self.weight @ input
        output = torch.cat((self.weight_11 *  input[0,0] + self.weight_12 *  input[1,0] , self.weight_21 *  input[0,0] + self.weight_22 *  input[1,0]), 0)
        return output

loss_
#%%

custom = myLinear()
a = torch.zeros((2,1), dtype = torch.cfloat)
a[0,0] = 1

b = torch.zeros((2,1), dtype = torch.cfloat)
b[1,0] = 1

custom.forward(a)




# %%
class HamiltonianNet(nn.Module):
    def __init__(self):
        
        super(HamiltonianNet, self).__init__()
        
        
        
        @qml.qnode(dev)
        def q_circuitGenerateLat(inputs = False):
            self.embeddingLatent(inputs)
            
            self.first_rots  =deepcopy(self.first_rots_lat)
            self.final_rots = deepcopy(self.final_rots_lat)
            self.cnot = deepcopy(self.cnot_lat)
            self.wires_list = deepcopy(self.wires_list_lat)
            # In the testing, SWAP the Reference Bit and the trash states
            
            
            for i in range(self.latent_space_size + self.auxillary_qubit_size , self.n_qubits):
                ind = i - (self.latent_space_size + self.auxillary_qubit_size)
                qml.QubitUnitary(self.final_rots[ind], wires = i)
                
            for i in range(len(self.cnot)):
                qml.QubitUnitary(self.cnot.pop() , wires = self.wires_list.pop())
            
            for i in range(self.latent_space_size + self.auxillary_qubit_size ,self.n_qubits):
                ind = i - (self.latent_space_size + self.auxillary_qubit_size)
                qml.QubitUnitary(self.first_rots[ind], wires = i)
                
            
            return qml.probs(range(self.auxillary_qubit_size+self.latent_space_size ,self.n_qubits))
        
        weight_shapes = {}
        

        weight_shapes = {"weights_r": (2 , training_qubits_size, 3),"weights_cr": (self.training_qubits_size,self.training_qubits_size-1 ,3), "weights_st":  (3,self.training_qubits_size,3)}
        self.qlayerGenerateLatent = qml.qnn.TorchLayer(q_circuitGenerateLat, weight_shapes)
    @qml.template
    def embeddingLatent(self,inputs):
        
        qml.templates.AmplitudeEmbedding(inputs, wires = range(self.latent_space_size + self.latent_space_size+ self.auxillary_qubit_size,self.n_qubits), normalize = True,pad=(0.j))
        #qml.QubitStateVector(inputs, wires = range(n_qubits))    
    def forward(self, x):
        print(x)
        x =  self.qlayerGenerateLatent(x)
        print(x)
        print(self.qlayerGenerateLatent.qnode.draw())
    
        
        return x



# %%
H = qml.Hamiltonian(
    [1, 1, 0.5],
    [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)])
print(H)


dev = qml.device('default.qubit', wires=2)

t = 1
n = 2

@qml.qnode(dev)
def circuit():
    qml.templates.ApproxTimeEvolution(H, t, n)
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]

circuit()
print(circuit.draw())



cnot = np.array(((1,0,0,0) ,(0,1,0,0) ,(0,0,0,1),(0,0,1+0j,0)))





pauli_y = np.array(((0, -1j), (1j, 0) ))
pauli_z = np.array(((1+0j, 0), (0, -1)))
pauli_x  = np.array(((0j, 1), (1, 0)))
I  = np.array(((0j + 1, 0), (0, 1)))

U = np.kron(pauli_x, pauli_x)
U_gate = np.kron(pauli_y, pauli_z)


hadamard = np.array(((1+0j,1), (1,-1)))
hadamard /= np.sqrt(2)


state = torch.zeros(4) + 0.j
state[0] = 1
state[2] = 1


state = np.abs(nn.functional.normalize((state).view(1,-1)).numpy()).reshape(-1)


kronproduct @ state


# %%
q1 = np.zeros(2)
q1[1] = 1 

q1 = hadamard @ q1
#q2 = np.arange(2)
q2 = np.zeros(2)
q2[0] = 1 
data = np.kron(q1,q2)



# %% 

H = np.array([[ 2.,    0.,    0. ,   0.05],[ 0. ,   0. ,   0.05,  0.  ],[ 0. ,   0.05 , 0.  ,  0.  ], [ 0.05 , 0.,    0.  , -2.  ]])


#%% 
lim = 20
expo = lambda m,lim: [pow(m,n) for n in(np.arange(lim))]
factorial = lambda lim: [np.math.factorial(n) for n in np.arange(lim)]
#lim = 5
taylor = lambda a,b: [x/y for x,y in zip(a,b)]
res= taylor(expo(5,lim), factorial (lim))
np.sum(res)
mat_pow  = lambda b,x : np.linalg.matrix_power(b,x)
matrix_expo = lambda m,lim: [taylor(mat_pow(m,n),factorial(n)) for n in(np.arange(lim))]
from scipy.linalg import expm, sinm, cosm,logm




cur_res = np.zeros((2,2))
for i in range(20):
    
    res = np.eye(2)
    for j in range( i):
        res = res @ H
    cur_res += res/np.math.factorial(i)
    
np.round(expm(H * -1j * np.pi/2) , 7)


np.linalg.eig(H)






# %% 

H = np.array(([-1,1], [1,-1]))
T = logm(expm(H* 1j))
# e^-iHt = U 

expm(H)


(np.exp(-2j)+ 1 ) / 2 
U_gate @ data



np.kron(pauli_y @ q1,pauli_z @ q2 )
