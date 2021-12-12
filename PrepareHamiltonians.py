# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 16:17:28 2021

@author: burak
"""

# Different hamiltonians, and their ground states

import pennylane as qml
import torch
import numpy as np
import timeit

# %% 


norm = lambda x:np.linalg.norm(x)
normalize = lambda x: x/norm(x)

torchnorm = lambda x:torch.norm(x)
torchnormalize = lambda x: x/torchnorm(x)


import pickle
tensorlist = []
for n_qubits in range(0,14):
    
    tensorlist_nqubits = []
    start_time = timeit.time.time()
    for i in range(1000):
        # state = (torch.rand(2**n_qubits, dtype = torch.cfloat)).numpy()
        state = (torch.rand(2**n_qubits, dtype = torch.cfloat)) - (0.1 + 0.1j)
        ctr = 0
        # while(norm(state) != 1.0 and ctr <= 5):
        #     state = normalize(state)
        #     ctr += 1
        while(torchnorm(state) != 1.0 and ctr <= 5):
            state = torchnormalize(state)
            ctr += 1
        tensorlist_nqubits.append((state))
        
        if(i%100== 0):
            print(n_qubits, ' ' , i)
    tensorlist.append(tensorlist_nqubits)
    end_time = timeit.time.time()
    print('time elapsed for normalization {:.2f}'.format(end_time - start_time))
    
    


with open('states.npy', 'wb') as f:
    np.save(f, np_tensors)
# %% Import saved qubits

# res = np.load('states.npy',allow_pickle = True)
# res[-1][0].all() == np_tensors[-1][0].all()
# del res

# %% 


for n_qubits in range(1,10):
    vec_dev = qml.device('default.qubit', wires = n_qubits)
    
    @qml.qnode(vec_dev)
    def qCircuit(inputs):
        qml.QubitStateVector(inputs, wires = range(0,n_qubits))
        
        return qml.probs(range(0,n_qubits))
        return [qml.expval(qml.PauliZ(q)) for q in range(0,1)]
    
    print(qCircuit(np_tensors[n_qubits][0]))

# %% 



def fidelityLossMulti(mes):
    return sum(torch.log(1 - (2 * mes[i] -1)) for i in range(len(mes)))


import torch
import pennylane as qml
import torch.nn as nn
import numpy as np

from scipy.linalg import expm

class QuantumAutoencoder(nn.Module):
    
    '''
    Quantum Autoencoder Class
    
    Attributes
    ----------
    _training_mode: bool
        Whether the model is still being trained, or tested
    
    n_qubits: int
        Number of qubits
    
    n_latent_qubits: int
        Number of qubits that can be discarded during encoding
    
    dev: qml.device
        Pennylane device object for a circuit simulation
        
    decode_latent: bool
        Only runs the decoder given a latent input
    
    encode_qubits: bool
        Given an input, return only the latent space
    
    Methods
    ----------
    qCircuit()
        Quantum circuit that uses QNode decorator
        If _training_mode flag is on:
            Returns a fidelity measure during training,
        If encode_qubits flag is on:
            returns latent space
        If decode latent flag is on:
            returns the decoded output given a latent space
    
    swapTest()
        Implements SWAP Test for calculating the fidelity between 2 qubits
        see https://arxiv.org/pdf/quant-ph/0102001.pdf
    
    embedding(type: str)
        Implements several different embedding methods
    
    forward(input: Tensor)
        Does the forward pass of a quantum circuit iteration given an input
        
    '''
    
    
    
    def __init__(self, n_qubits, dev, n_latent_qubits = 1, n_auxillary_qubits = 1, input_size = 50):
        '''
        Parameters
        ----------
        
        n_qubits: int
            Number of qubits
        
        dev: qml.device
            Pennylane device object for a circuit simulation
        
        n_latent_qubits: int
            Number of qubits that can be discarded during encoding
        
        n_auxillary_qubits: int
            number of auxillary qubits to be used as ancillas in SWAP test
        
        '''
        
        
        super(QuantumAutoencoder, self).__init__()
        
        self._n_qubits = n_qubits
        self._dev = dev
        self._n_latent_qubits = n_latent_qubits
        self._n_auxillary_qubits = n_auxillary_qubits
        self._n_total_qubits = n_qubits + n_latent_qubits+  n_auxillary_qubits
        
        self._decode_latent = False
        self._encode_qubits = False
        self._training_mode = True
        self._draw_circuit = False
        
        self.ctr = 0
        self.input_size = input_size
        self.embedding_mode = 'amplitude'
        
        self.entanglingParameters()
        
        @qml.qnode(dev)
        def qCircuit(weights_r, weights_cr, inputs = False):
            if(self.training_mode == True):
                qml.QubitStateVector(inputs, wires = range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits))
                
                for idx, i in enumerate(range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)):
                    qml.Rot(*weights_r[0, idx] , wires = i)
                
                for idx, i in enumerate(range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)):
                    ctr=0
                    for jdx, j in enumerate(range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)):
                        if(i==j):
                            pass
                        else:
                            qml.CRot( *weights_cr[idx,ctr], wires= [i,j])
                            ctr += 1
                            
                for idx, i in enumerate(range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)):
                    qml.Rot(*weights_r[1, idx] , wires = i)
                
                    
                self.swapTestMulti()
                # return [qml.probs(i) for i in range(self.n_auxillary_qubits)]
                return [qml.expval(qml.PauliZ(q)) for q in range(0, self.n_auxillary_qubits )]
            else:
                self.crot_cont = []
                qml.QubitStateVector(inputs, wires = range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits))
                
                for idx, i in enumerate(range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)):
                    qml.Rot(*weights_r[0, idx] , wires = i)
                
                for idx, i in enumerate(range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)):
                    ctr=0
                    for jdx, j in enumerate(range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)):
                        if(i==j):
                            pass
                        else:
                            qml.CRot( *weights_cr[idx,ctr], wires= [i,j])
                            self.crot_cont.insert(0,[idx,ctr, i , j] )
                            ctr += 1
                            
                            
                for idx, i in enumerate(range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)):
                    qml.Rot(*weights_r[1, idx] , wires = i)
                for i in range(self.n_latent_qubits):
                    qml.SWAP(wires = [self.n_auxillary_qubits + i , self.n_auxillary_qubits + i + self.n_latent_qubits])
                    
                for i in range(self.n_latent_qubits + self.n_auxillary_qubits , self.n_total_qubits):
                    ind = i - (self.n_latent_qubits + self.n_auxillary_qubits)
                    qml.Rot(*weights_r[1, ind], wires = i).inv()
                    
                for i in range(len(self.crot_cont)):
                    vals = self.crot_cont[i]
                    print(weights_cr.size())
                    qml.CRot( *weights_cr[vals[0],vals[1]]  ,wires= [vals[2],vals[3]]).inv()
                
                for i in range(self.n_latent_qubits + self.n_auxillary_qubits , self.n_total_qubits):
                    ind = i - (self.n_latent_qubits + self.n_auxillary_qubits)
                    qml.Rot(*weights_r[0, ind], wires = i).inv()
                
                
                
                return qml.probs(range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits))
            
        weight_shapes = {"weights_r": (2 , self.n_qubits, 3),"weights_cr": (self.n_qubits,self.n_qubits-1 ,3)}
        self.qLayer = qml.qnn.TorchLayer(qCircuit, weight_shapes)
        
        
    def forward(self, x,training_mode = True, decode_latent = False, encode_qubits = False, draw_circuit= False): 
        
        self._training_mode = training_mode
        self._decode_latent = decode_latent
        self._encode_qubits = encode_qubits
        self._draw_circuit = draw_circuit
        
        x = self.qLayer(x)
        # Normalize exp. value between [0,1] for [1,-1]
        x +=  1
        x = x/2
        self.ctr += 1
        return x
    
    def swapTest(self):
        # SWAP Test measures the similarity between 2 qubits 
        # see https://arxiv.org/pdf/quant-ph/0102001.pdf
        for i in range(self.n_auxillary_qubits):
            qml.Hadamard(wires = i)
        for i in range(self.n_auxillary_qubits):
            qml.CSWAP(wires = [i, i + self.n_auxillary_qubits , self.n_auxillary_qubits + i + self.n_latent_qubits])
        for i in range(self.n_auxillary_qubits):
            qml.Hadamard(wires = i)
    
    
    def swapTestMulti(self):
        # SWAP Test measures the similarity between 2 qubits 
        # see https://arxiv.org/pdf/quant-ph/0102001.pdf
        for i in range(self.n_auxillary_qubits):
            qml.Hadamard(wires = i)
        for i in range(self.n_auxillary_qubits):
            qml.CSWAP(wires = [i, i+ self.n_auxillary_qubits , self.n_auxillary_qubits + i + self.n_latent_qubits])
        for i in range(self.n_auxillary_qubits):
            qml.Hadamard(wires = i)
    
    @qml.template
    def embedding(self, inputs, embedding_type ='angle'):
        if(embedding_type == 'angle'):
            qml.templates.embeddings.AngleEmbedding(inputs,wires = range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits ), rotation = 'X')
        elif(embedding_type == 'entangler'):
            # random_entangling_weights = qml.init.strong_ent_layers_normal(self.n_qubits,self.n_qubits)
            # qml.templates.layers.StronglyEntanglingLayers(random_entangling_weights, range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)) 
            qml.templates.layers.StronglyEntanglingLayers(self.entangling_parameters[self.ctr % self.input_size], range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)) 
        else:
            qml.templates.embeddings.AmplitudeEmbedding(inputs,wires = range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits ), normalize = True,pad_with=(0.j))
        
    
    def entanglingParameters(self):
        self.entangling_parameters = [qml.init.strong_ent_layers_normal(self.n_qubits,self.n_qubits) for i in range(self.input_size)]
        # self.entangling_parameters = [qml.init.strong_ent_layers_uniform(self.n_qubits,self.n_qubits, low = np.pi/3, high = np.pi*3/2) for i in range(self.input_size)]
    
    
    @property
    def training_mode(self):
        return self._training_mode
    
    @training_mode.setter
    def training_mode(self, flag):
        self._training_mode = flag
        
    ## Getter-Setter decode_latent
    @property
    def decode_latent(self):
        return self._decode_latent
    
    @decode_latent.setter
    def decode_latent(self, flag):
        self._decode_latent = flag    
    ## 
    @property
    def encode_qubits(self):
        return self._encode_qubits
    
    @encode_qubits.setter
    def encode_qubits(self, flag):
        self._encode_qubits = flag 
        
    ## 
    @property
    def draw_circuit(self):
        return self._draw_circuit
    
    @draw_circuit.setter
    def draw_circuit(self, flag):
        self._draw_circuit = flag 
    ##
    
    @property
    def n_qubits(self):
        return self._n_qubits
    
    @property
    def n_latent_qubits(self):
        return self._n_latent_qubits
    
    @property
    def n_auxillary_qubits(self):
        return self._n_auxillary_qubits
    
    @property
    def n_total_qubits(self):
        return self._n_total_qubits
    
    @property
    def dev(self):
        return self._dev
    
    
    

# %% 

n_qubit_sizes = list(range(3,12))
iteration_duration = []

for n in n_qubit_sizes:
    
    n_qubit_size = n
    latent_size = 2
    n_auxillary_qubits = latent_size
    
    
    dev = qml.device('default.qubit', wires = n_qubit_size + latent_size + n_auxillary_qubits)
    loss_func = fidelityLossMulti
    learning_rate = 0.02
    
    qAutoencoder = QuantumAutoencoder(n_qubit_size, dev, latent_size, n_auxillary_qubits)
    
    opt = torch.optim.Adam(qAutoencoder.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    average_fidelities = []
    batch_size = 4
    epochs = 30
    
    n_data = 10
    
    
    start_time = timeit.time.time()
    
    opt.zero_grad()
    out = qAutoencoder(tensorlist[n_qubit_size][0])
    
    loss = loss_func(out)
    print(out)
    loss.backward()
    opt.step()
    
    end_time = timeit.time.time()
    iteration_duration.append(end_time - start_time )
    print('Time elapsed for {} qubits:  {:.2f} '.format(n_qubit_size, end_time-start_time ))
# %% 

n_qubit_size = 10
latent_size = 1
n_auxillary_qubits = latent_size


dev = qml.device('default.qubit', wires = n_qubit_size + latent_size + n_auxillary_qubits)
loss_func = fidelityLossMulti
learning_rate = 0.01
n_data = 100
qAutoencoder = QuantumAutoencoder(n_qubit_size, dev, latent_size, n_auxillary_qubits, n_data)

opt = torch.optim.Adam(qAutoencoder.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# opt = torch.optim.RMSprop(qAutoencoder.parameters(), lr = learning_rate)

average_fidelities = []
avg_loss = 0
batch_size = 4
epochs = 15

# qAutoencoder(H)



#x = torch.stack([torch.Tensor(qml.init.strong_ent_layers_normal(n_qubit_size, qAutoencoder.n_total_qubits - (2* n_auxillary_qubits))) for i in range(n_data) ])
# %% Training

for epoch in range(epochs):
    running_loss = 0
    start_time = timeit.time.time()
    batch_id = np.arange(n_data)
    np.random.shuffle(batch_id)
    for i in batch_id:
        opt.zero_grad()
        out = qAutoencoder(tensorlist[n_qubit_size][i])
        loss = loss_func(out)
        loss.backward()
        # print(out)
        running_loss += loss
        opt.step()
    epoch_loss = running_loss / n_data
    avg_loss = (avg_loss * epoch + epoch_loss) / (epoch + 1)
    print(epoch_loss)
    end_time = timeit.time.time()
    
    print('Time elapsed for the epoch' +  str(epoch)  + ' : {:.2f}'.format(end_time-start_time))
    
    
# %% 

for i in range(10):
    state = (torch.rand(2**n_qubit_size, dtype = torch.cfloat)) - 0.013j
    ctr = 0
    while(torchnorm(state) != 1.0 and ctr <= 5):
        state = torchnormalize(state)
        ctr += 1
    
    
    # print(qAutoencoder(tensorlist[n_qubit_size][i]))
    print(qAutoencoder(state, training_mode = False))
    




    # input_state  = np.kron(np.kron([1,0], [1,0]),tensorlist[n_qubit_size][i].numpy() )
    input_state  = np.kron(np.kron([1,0], [1,0]),state.numpy() )
    result = np.array(dev.state.detach())
    
    # how similar is the output to the input
    similarity = sum(np.abs((result-input_state) ** 2)) / 1048
    print('similarity is {:.9f}'.format(similarity))

# %%  NEXT THINGS TO DO

# PREPARE THE DEV.STATE WITH THE INPUT THAT IS PREPARED WITH QARBITRARYSTATE - DONE
# Prepare examplary hamiltonians, 
# 