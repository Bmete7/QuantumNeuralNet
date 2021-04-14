# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 15:03:42 2021

@author: burak
"""

import tensorflow as tf
import pennylane as qml
import scipy as sp

# %% 


# %%        
device = qml.device("default.qubit", wires=6,shots = 1000)
hamNet = HamiltonianNet(device)        
hamOpt = torch.optim.Adam(hamNet.parameters() , lr = 0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
for j in range(0,1):
    for i in range(70):
        
        train_data = torch.cat((embed_features[initial_succesfull_train_sample_index[j]] , embed_features[initial_succesfull_train_sample_index[j+200] ]))
        y = hamNet(train_data)
        
        loss = loss_func(y)
        loss.backward()
    
        hamOpt.step()
        print(loss)


hamNet(train_data,True)




# %% 


amps_prob = qml.device("default.qubit", wires=2,shots = 1000)
@qml.qnode(amps_prob)
def circ_ampprob(inputs):
    qml.templates.embeddings.AmplitudeEmbedding(inputs,wires = range(0,2), normalize = True,pad=(0.j))
    return qml.probs([0,1])

circ_ampprob(qubit_state_tf_input[0])

amps_dev = qml.device("default.qubit", wires=3,shots = 1000)
@qml.qnode(amps_dev)
def circ(inputs):
    qml.templates.embeddings.AngleEmbedding(inputs,wires = range(1,3), rotation = 'X')
    
    qml.Rot(*first_rot_params_0, wires = 1)
    qml.Rot(*first_rot_params_1, wires = 2)
    
    qml.CRot(*crot_params_01 , wires = [1,2])
    qml.CRot(*crot_params_10 , wires = [2,1])
    
    qml.Rot(*final_rot_params_0, wires = 1)
    qml.Rot(*final_rot_params_1, wires = 2)
    
    qml.SWAP(wires= [0,1])
    
    qml.Rot(*final_rot_params_0, wires = 1).inv()
    qml.Rot(*final_rot_params_1, wires = 2).inv()
    
    qml.CRot(*crot_params_10 , wires = [2,1]).inv()
    qml.CRot(*crot_params_01 , wires = [1,2]).inv()
    
    qml.Rot(*first_rot_params_0, wires = 1).inv()
    qml.Rot(*first_rot_params_1, wires = 2).inv()
    
    return qml.probs([1,2])
amps_dev_full = qml.device("default.qubit", wires=3,shots = 1000)
@qml.qnode(amps_dev_full)
def circ2(inputs):
    qml.templates.embeddings.AngleEmbedding(inputs,wires = range(1,3), rotation = 'X')
    qml.Rot(*first_rot_params_0, wires = 1)
    qml.Rot(*first_rot_params_1, wires = 2)
    qml.CRot(*crot_params_01 , wires = [1,2])
    qml.CRot(*crot_params_10 , wires = [2,1])
    qml.Rot(*final_rot_params_0, wires = 1)
    qml.Rot(*final_rot_params_1, wires = 2)
    return qml.probs([1,2])

qubit_state = []

amps_d_tek = qml.device("default.qubit", wires=1,shots = 1000)
@qml.qnode(amps_d_tek)
def circ_amp_tek(inputs):
    qml.templates.embeddings.AngleEmbedding(inputs,wires = range(0,1), rotation = 'X')
    return qml.probs([0])  

circ_amp_tek([1.0392919])
circ_amp_tek([0.89708031])

circ_amp(embed_features[0])

q_values
amps_d_tek._state

print(circ_amp(embed_features[initial_succesfull_train_sample_index[0]].detach().numpy()))
amps_d._state
torch.Tensor()
for j in range(len(initial_succesfull_train_sample_index)): 
    
    print(circ(embed_features[initial_succesfull_train_sample_index[j]].detach().numpy()))
    print(circ_amp(embed_features[initial_succesfull_train_sample_index[j]].detach().numpy()))
    qubit_state.append(amps_d._state)
    
qubit_state_tf_input = tf.reshape(tf.convert_to_tensor(qubit_state[:133]), [133, 4])
qubit_state_tf_target =  tf.reshape(tf.convert_to_tensor(qubit_state[133:393]), [393-133, 4])


    

# %% Create 2 qubit paulis with coeffs

# dev = qml.device('default.qubit', wires=2 , shots = 10000)
class ExpMat(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.observables  = tf.Variable(
                initial_value=observables,
                trainable=False,
                dtype = tf.complex128,
                name = 'obs'
            )   
        self.coefs = tf.Variable(
                initial_value=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                trainable=True,
                dtype = tf.complex128,
                name = 'coefs'
            )  
        self.zerotensor  = tf.Variable(
                initial_value=0,
                trainable=False,
                dtype = tf.float64,
                name = 'dz'
            )
    def __call__(self, x):

        out = self.coefs[0]*self.observables[0]
        for i in range(1, 16):
            out += self.coefs[i]*self.observables[i]
        
        
        l = tf.linalg.expm(-1j * out)
        l2 = tf.linalg.expm(-1j * out * 2)
        
        
        # print( tf.matmul(l, l , adjoint_b = True))
        
        return tf.linalg.matvec(l, x) ,  tf.linalg.matvec(l2, x) 
        # l2 = 
        # l3 = tf.linalg.matvec(l3, x)
        # out = x * self.coefs[0]
        # for i in range(16):
        #     out += x * self.coefs[i]
        # return out
        # return l,l2,l3
        # return self.qlayer((l[0] , l[1]))


# %% 
        
# import scipy as sp
# m  = np.matrix([[2,1-1j] , [1+1j, 8]])
# t = tf.convert_to_tensor(m)
# tf.math.conj(t) @ t
# m.H@ m
# tf.linalg.matmul(t,t,adjoint_a = True)
# mm = np.matrix(sp.linalg.expm(-1j * m))
# tt = tf.convert_to_tensor(mm)
# res = tf.linalg.matvec(tt,tf.convert_to_tensor([1,0], dtype = tf.complex128))
# np.sum(np.abs((res ** 2)))

# tf.convert_to_tensor(mm) *  tf.math.conj( tf.convert_to_tensor(mm)) + tf.convert_to_tensor(mm)[0,1] *  tf.math.conj( tf.convert_to_tensor(mm))[1,0]
# mm[0,1]
# mm[0,1] * mm.H[1,0]
# xx[0][1]
# %% Modularize the code. 


expMat = ExpMat(name="expmat")



optimizer = tf.keras.optimizers.SGD(learning_rate=0.11)
loss_history = []

def complex_norm(num):
    return tf.math.sqrt(tf.math.real(num)**2 + tf.math.imag(num) **2)

def train_step(psi_in, psi_out, psi_out_t2):
    with tf.GradientTape() as tape:
        res = expMat(psi_in)
        
        
        loss_value = tf.reduce_sum(res[0] - psi_out) + tf.reduce_sum(res[1] - psi_out_t2) 
        
        #+ (res[1] - psi_out_t2)
        #loss_value = tf.tensordot(earn_param,res , axes = 0) + res
        #loss_value = (psi_out_t2[0] - tf.norm(res[1][0])) 
        # loss_value = tf.reduce_sum(tf.pow(psi_out  - res[0] ,2))/(2) + tf.reduce_sum(tf.pow(psi_out_t2  - res[1] ,2))/(2)
        
        
    loss_history.append(loss_value.numpy().mean())
    
    grads = tape.gradient(loss_value, expMat.trainable_variables)
    optimizer.apply_gradients(zip(grads,expMat.trainable_variables))
    for i in range(16):
            expMat.coefs[i].assign(tf.complex(tf.math.real(expMat.coefs[i]), expMat.zerotensor))
    return loss_value, res


    
        
for i in range(300):
    for j in range(1):
        loss_value_tf,res = train_step(qubit_state_tf_input[j], qubit_state_tf_target[j*2], qubit_state_tf_target[j*2+ 1])
print(loss_value_tf, res)
        



#%%

tf_succesfull_train_samples_input = deepcopy(succesfull_train_samples_input)
tf_succesfull_train_samples_evolved = deepcopy(succesfull_train_samples_evolved)


