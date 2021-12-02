# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import pennylane as qml
import torch.nn as nn
import numpy as np
import timeit



class QAutoencoder(nn.Module):
    
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
    
    
    
    def __init__(self, n_qubits, dev, n_latent_qubits = 1, n_auxillary_qubits = 1):
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
        
        
        super(QAutoencoder, self).__init__()
        
        self._n_qubits = n_qubits
        self._dev = dev
        self._n_latent_qubits = n_latent_qubits
        self._n_auxillary_qubits = n_auxillary_qubits
        self._n_total_qubits = n_qubits + n_latent_qubits+  n_auxillary_qubits
        
        self._decode_latent = False
        self._encode_qubits = False
        self._training_mode = True
        self._draw_circuit = False
        
        
        @qml.qnode(dev)
        def qCircuit(weights_r, weights_cr, inputs = False):
            # pyTorch implementation
            if(self.training_mode):
                self.embedding(inputs, 'amplitude')
                
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
                
                    
                self.swapTest()
                # return [qml.probs(i) for i in range(self.n_auxillary_qubits)]
                return [qml.expval(qml.PauliZ(q)) for q in range(0, self.n_auxillary_qubits )]
            else:
                pass
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
        return x
    
    def swapTest(self):
        # SWAP Test measures the similarity between 2 qubits 
        # see https://arxiv.org/pdf/quant-ph/0102001.pdf
        for i in range(self.n_auxillary_qubits):
            qml.Hadamard(wires = i)
        for i in range(self.n_auxillary_qubits):
            qml.CSWAP(wires = [i, i+ self.n_auxillary_qubits , self.n_auxillary_qubits + i + self.n_latent_qubits])
        # for i in range(self.auxillary_qubit_size, self.latent_space_size + self.auxillary_qubit_size):
        #     qml.CSWAP(wires = [0, i, i + self.latent_space_size])
        for i in range(self.n_auxillary_qubits):
            qml.Hadamard(wires = i)
    
    
    def swapTestMulti(self):
        # SWAP Test measures the similarity between 2 qubits 
        # see https://arxiv.org/pdf/quant-ph/0102001.pdf
        for i in range(self.n_auxillary_qubits):
            qml.Hadamard(wires = i)
        for i in range(self.n_auxillary_qubits):
            qml.CSWAP(wires = [i, i+ self.n_auxillary_qubits , self.n_auxillary_qubits + i + self.n_latent_qubits])
        # for i in range(self.auxillary_qubit_size, self.latent_space_size + self.auxillary_qubit_size):
        #     qml.CSWAP(wires = [0, i, i + self.latent_space_size])
        for i in range(self.n_auxillary_qubits):
            qml.Hadamard(wires = i)
    
    
    
    
    
    
    
    
    
    
    
    @qml.template
    def embedding(self, inputs, embedding_type ='angle'):
        if(embedding_type == 'angle'):
            qml.templates.embeddings.AngleEmbedding(inputs,wires = range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits ), rotation = 'X')
        else:
            qml.templates.embeddings.AmplitudeEmbedding(inputs,wires = range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits ), normalize = True,pad=(0.j))
    
    
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


def fidelityLoss(mes):
    fidelity = 2 * mes  - 1.00
    return torch.log(1 - fidelity)


def fidelityLossMulti(mes):
    return sum(torch.log(1 - (2 * mes[i] -1)) for i in range(len(mes)))
    
def A(a):
    return sum([a[i] for i in range(3)])






# %% Calculating duration for one training iteration for x qubits

n_qubit_sizes = list(range(3,13))
iteration_duration = []

for n in n_qubit_sizes:
    
    n_qubit_size = n
    latent_size = 2
    auxillary_qubit_size = latent_size
    
    
    dev = qml.device('default.qubit', wires = n_qubit_size + latent_size + auxillary_qubit_size)
    loss_func = fidelityLossMulti
    learning_rate = 0.01
    
    qAutoencoder = QAutoencoder(n_qubit_size, dev, latent_size, auxillary_qubit_size)
    
    opt = torch.optim.Adam(qAutoencoder.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    average_fidelities = []
    batch_size = 4
    epochs = 100
    
    n_data = 100
    
    x = torch.rand(n_data, 2**n_qubit_size)
    
    start_time = timeit.time.time()
    
    opt.zero_grad()
    out = qAutoencoder(x[0].flatten())
    
    loss = loss_func(out)
    print(out)
    loss.backward()
    opt.step()
    
    end_time = timeit.time.time()
    iteration_duration.append(end_time - start_time )
    print('Time elapsed for {} qubits:  {:.2f} '.format(n_qubit_size, end_time-start_time ))

input('add smt')

# %% 

n_qubit_size = 10
latent_size = 2
auxillary_qubit_size = latent_size


dev = qml.device('default.qubit', wires = n_qubit_size + latent_size + auxillary_qubit_size)
loss_func = fidelityLossMulti
learning_rate = 0.01

qAutoencoder = QAutoencoder(n_qubit_size, dev, latent_size, auxillary_qubit_size)

opt = torch.optim.Adam(qAutoencoder.parameters() , lr = learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

average_fidelities = []
avg_loss = 0
batch_size = 4
epochs = 30

n_data = 100

x = torch.rand(n_data, 2**n_qubit_size)
x  = torch.rand(n_data, 2**n_qubit_size , dtype = torch.cfloat)


# %% 

for epoch in range(epochs):
    running_loss = 0
    start_time = timeit.time.time()
    batch_id = np.arange(n_data)
    np.random.shuffle(batch_id)
    for i in batch_id:
        opt.zero_grad()
        out = qAutoencoder(x[i].flatten())
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
# %% Testing
test_size = 10
test_data = torch.rand(test_size , 2**n_qubit_size , dtype = torch.cfloat)


for i in range(test_size):
    print(qAutoencoder(test_data[i]))