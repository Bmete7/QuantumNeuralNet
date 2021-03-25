# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 00:43:09 2021

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
import timeit
import torchvision
from scipy.linalg import expm, sinm, cosm,logm
import math
# %%

'''
# TODO: 
    + Run a torch optimization with complex matrices
    - See if exponentiating via Pauli matrices hold
    - Run a torch optimization after exponentiating a hermitian
    - Create a simple trainable QCircuit with 1 Unitary Gate
    - Try to create unitary gates with complex numbers, and run them on pennyLane
    - Try to exponentiate a hermitian matrix, then simulate the resulting unitary
    15.01.2021

'''
    
    


# %%

def StateL2Loss(s1,s2):
    return ((s1[0].real-s2[0].real) * (s1[0].real-s2[0].real)) + ((s1[0].imag-s2[0].imag) * (s1[0].imag-s2[0].imag)) + ((s1[1].real-s2[1].real) * (s1[1].real-s2[1].real)) + ((s1[1].imag-s2[1].imag) * (s1[1].imag-s2[1].imag))
    # return torch.abs((((s1[0]-s2[0])* (s1[0]-s2[0]))).real  + (((s1[0]-s2[0])* (s1[0]-s2[0]))).imag + ((s1[1]-s2[1]) * (s1[1]-s2[1])).real + ((s1[1]-s2[1]) * (s1[1]-s2[1])).imag)
# %% 

plus_state  = minus_state.clone().detach()
plus_state[1,0] *= -1

# minus_state = torch.ones((1,2), dtype = torch.cfloat)
minus_state = mat.detach().clone()

minus_state[0] = plus_state[0,0].clone().detach()
minus_state[1] = plus_state[0,1].clone().detach() * -1
H = torch.zeros((2,2),dtype = torch.cfloat)
H[0,0] = -1 
H[0,1] = 1

H[1,0] = 1 
H[1,1] = -1

pauli_y = np.array(((0, -1j), (1j, 0)))
pauli_z = np.array(((1+0j, 0), (0, -1)))
pauli_x  = np.array(((0j, 1), (1, 0)))
I  = np.array(((0j + 1, 0), (0, 1)))

# %%

dev = qml.device('default.qubit', wires=1)


class HamiltonianNet(nn.Module):
    def __init__(self, n =2):
        
        super(HamiltonianNet, self).__init__()
        
        self.H = torch.zeros((2,2),dtype = torch.cfloat,requires_grad=True)
        self.H[0,0] = -1 
        self.H[0,1] = 1
        
        self.H[1,0] = 1 
        self.H[1,1] = -1
        @qml.qnode(dev)
        def OptimPauliY(weights, inputs = False):
            self.embeddingLatent(inputs)
            print(weights, inputs)
            for i in range(0,math.floor(np.sqrt(2))):
                qml.QubitUnitary( np.matrix([[1,0], [0,1]]) ,wires = i)
            qml.PauliZ(0)
            
            
            return qml.expval(qml.PauliZ(0))
            
        
        
        

        weight_shapes = {"weights": (n,n)}
        self.optimPauliY = qml.qnn.TorchLayer(OptimPauliY, weight_shapes)
    @qml.template
    def embeddingLatent(self,inputs):
        qml.templates.AmplitudeEmbedding(inputs, wires = range(0,math.floor(np.sqrt(2))), normalize = True,pad=(0.j))
        
    def forward(self, x):
        print(x)
        x =  self.optimPauliY(x)
        print(x)
        #print(self.optimPauliY.qnode.draw())    
        
        return x
    
model = HamiltonianNet()
print(model.forward(plus_state))
# %%
@qml.qnode(dev)
def OptimPauliY( weights, inputs = False):
    
    qml.templates.AmplitudeEmbedding(inputs, wires = range(0,math.floor(np.sqrt(2))), normalize = True,pad=(0.j))
    # for i in range(0,math.floor(np.sqrt(2))):
    #     qml.QubitUnitary(weights, wires = i) 
    qml.PauliZ(0)                   
    return qml.expval(qml.PauliZ(0))


print(OptimPauliY(0,plus_state))
/
/
# %% 





u,s,vh   = np.linalg.svd(A)
u @ np.diag(s) @ np.matrix(vh).H


eigvals, eigvecs = np.linalg.eig(A)
eig1,eig2  = eigvals 
euler_eigs = np.array([np.cos(eig1) + 1j * np.sin(eig1) , np.cos(eig2)  + 1j * np.sin(eig2)])

np.round(u @ np.diag(euler_eigs) @ np.matrix(vh).H , 3)

np.round(expm(A*1j) , 3)
np.exp(1j*-4)
(A@ a)/4 == a

np.cos(1) * np.eye(2) + 1j * np.sin(1) * A
pauli_y = np.array(((0, -1j), (1j, 0) ))
pauli_z = np.array(((1+0j, 0), (0, -1)))

np.round(expm(1j * A  * np.pi/2) , 2)


# %% 

# Pauli matrices commutation relations
y = np.array(((0, -1j), (1j, 0)))
z = np.array(((1+0j, 0), (0, -1)))
x  = np.array(((0j, 1), (1, 0)))
I  = np.array(((0j + 1, 0), (0, 1)))
c1 = 0.6
c2 = 0.8
c3 = 0.4
H = c1*x + c2*y + c3*z 
eh = expm(-1j * (H))
ex = expm(-1j * 0.6 * (x))
ey = expm(-1j * 0.8 * (y))
ez = expm(-1j * 0.4 * (z))
ex@ey@ez

np.cos(1) - 1j*np.sin(1)*c3

-c2 * np.sin(1) - 1j* np.sin(1) * 2

x@y - y@x = 2z
expm(x) @ expm(y)
expm(y) @ expm(x)
# %% 

# TODO 2: Exponentiating the Hamiltonian via Pauli Matrices
# e^-iH ?= e^-i\sigma_x * . . . . * e-^iI
# I does not commute, so fix it to 1
pauli_y = np.array(((0, -1j), (1j, 0)))
pauli_z = np.array(((1+0j, 0), (0, -1)))
pauli_x  = np.array(((0j, 1), (1, 0)))
I  = np.array(((0j + 1, 0), (0, 1)))

np.round(expm( 1j * pauli_x),4)
u,s,vh   = np.linalg.svd(pauli_x)

eigvals, eigvecs = np.linalg.eig(pauli_x)
eig1,eig2  = eigvals 
euler_eigs = np.array([np.cos(eig1) + 1j * np.sin(eig1) , np.cos(eig2)  + 1j * np.sin(eig2)])

np.round(u @ np.diag(euler_eigs) @ np.matrix(vh).H , 3)

pauli_y = torch.tensor(pauli_y)
pauli_z = torch.tensor(pauli_z)
pauli_x = torch.tensor(pauli_x)

I = torch.tensor(I)
a1 = torch.tensor(0)
a2 = torch.tensor(0, dtype = torch.float, requires_grad=True)
a3 = torch.tensor(0.5, dtype = torch.float, requires_grad=True)
a4 = torch.tensor(0.5, dtype = torch.float, requires_grad=True)
H = a1*I + a2*pauli_x+ a3*pauli_y+ a4*pauli_z

#%%

# TODO 1: Optimizing a Complex Valued matrix

pauli_y = np.array(((0, -1j), (1j, 0)))

pauli_z = np.array(((1+0j, 0), (0, -1)))
pauli_x  = np.array(((0j, 1), (1, 0)))
pauli_y = torch.tensor(pauli_y)
pauli_z = torch.tensor(pauli_z)
pauli_x = torch.tensor(pauli_x)
I  = np.array(((0j + 1, 0), (0, 1)))
a1 = torch.tensor(0)
a2 = torch.tensor(0, dtype = torch.float, requires_grad=True)
a3 = torch.tensor(0.1, dtype = torch.float, requires_grad=True)
a4 = torch.tensor(0.9, dtype = torch.float, requires_grad=True)

optimizer = optim.SGD([a2,a3,a4], lr=0.01, momentum=0.9)
   

for i in range(500):
    optimizer.zero_grad()
    
    #r = custom.forward([np.sqrt(1/2), np.sqrt(1/2)])
    #loss = ComplexL2(r, b)
    #loss.backward()
    H = a1*I + a2*pauli_x+ a3*pauli_y+ a4*pauli_z
    mat = H[0,0]* plus_state[0,0] + H[0,1]* plus_state[0,1]
    mat2 = H[1,0]* plus_state[0,0] + H[1,1]* plus_state[0,1]
    mat = torch.vstack((mat,mat2))
    loss = StateL2Loss(mat,minus_state)
    loss.backward()
    print(loss)
    if(i%100 == 0):
        print(a1,a2,a3,a4)

    optimizer.step()
# %%

a = torch.zeros((2,1), dtype = torch.cfloat)
a[0,0] = 1

b = torch.zeros((2,1), dtype = torch.cfloat)
b[1,0] = 1
custom = myLinear(a)


B= torch.randn((2,2),requires_grad=True)

def dummyEigenLoss(eig_o,eig_t):
    return torch.sum(torch.square(eig_o-eig_t))
t = torch.zeros((2,))
t[0] = 2
t[1] = 1
print(B)
for i in range(1000):
    optimizer.zero_grad()
    #r = custom.forward([np.sqrt(1/2), np.sqrt(1/2)])
    #loss = ComplexL2(r, b)
    #loss.backward()
    #eigenvalues,eigenvectors = torch.symeig(B,eigenvectors = True)
    out= 
    
    if(i%100 == 0):
        print(loss)
    optimizer.step()

# %%  TENSORFLOW

H[0,0] = 4
H[0,1] = 0
H[1,0] = 0
H[1,1] = -4
# tH = tf.Variable(H, dtype = tf.complex128, trainable = False )
# psi_in = tf.Variable(plus_state, dtype = tf.complex128, trainable = False )
# psi_out = tf.Variable(minus_state, dtype = tf.complex128, trainable = False )
# t = tf.Variable([1], dtype = tf.complex128)
# extH = tf.linalg.expm(-1j* tH * t, name=None)
#res = tf.linalg.matmul(extH,psi_in)
tH = tf.Variable(H, dtype = tf.complex128, trainable = False )
psi_in = tf.Variable(plus_state, dtype = tf.complex128, trainable = False )
psi_out = tf.Variable(minus_state, dtype = tf.complex128, trainable = False )
t = tf.Variable([1], dtype = tf.complex128 , name='t')

with tf.GradientTape() as tape:
    
grad = tape.gradient(res, t)
    
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


loss_history = []

# %% 
def train_step(psi_in, psi_out,tH):
    with tf.GradientTape() as tape:
        res = expMat(psi_in)
        
        loss_value = tf.reduce_sum(tf.pow( tf.math.real(res) - tf.math.real(psi_out),2))/(2) + tf.reduce_sum(tf.pow( tf.math.imag(res) - tf.math.imag(psi_out),2))/(2)
        # loss_value = tf.reduce_sum(tf.pow(psi_out  - res,2))/(2)
        
    loss_history.append(loss_value.numpy().mean())
    
    grads = tape.gradient(loss_value, expMat.trainable_variables)
    optimizer.apply_gradients(zip(grads,expMat.trainable_variables))

for i in range(200):
    train_step(psi_in, psi_out,tH)

# %% 

#loss
loss = tf.reduce_sum(tf.square(psi_in - res))
tf.train.GradientDescentOptimizer(1.0).minimize(loss)

#tf.linalg.matmul(extH, tf.transpose(extH, conjugate = True))
# %% 

class ExpMat(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.h  = tf.Variable(
                initial_value=H,
                trainable=False,
                dtype = tf.complex128,
                name = 'h'
            )
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
        self.t  = tf.Variable(
                initial_value=5,
                trainable=True,
                dtype = tf.complex128,
                name = 't'
            )
        self.b  = tf.Variable(
                initial_value=2,
                trainable=True,
                dtype = tf.complex128
            )
        self.c  = tf.Variable(
                initial_value=3,
                trainable=True,
                dtype = tf.complex128
            )
        self.d  = tf.Variable(
                initial_value=4,
                trainable=True,
                dtype = tf.complex128
            )
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
        return tf.matmul((-1j *self.h * self.t), x)

expMat = ExpMat(name="expmat")


# %% 
import tensorflow as tf

x = tf.Variable(2, name='x', dtype=tf.float32)
log_x = tf.math.log(x)
log_x_squared = tf.square(log_x)

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(log_x_squared)

init = tf.initialize_all_variables()

def optimize():
  with tf.Session() as session:
    session.run(init)
    print("starting at", "x:", session.run(x), "log(x)^2:", session.run(log_x_squared))
    for step in range(10):  
      session.run(train)
      print("step", step, "x:", session.run(x), "log(x)^2:", session.run(log_x_squared))
        

optimize()