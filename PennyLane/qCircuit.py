# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:13:48 2020

@author: burak
"""
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml


# %% Whole network is defined within this class
class Net(nn.Module):
    def __init__(self,dev, latent_space_size, n_qubits,training_qubits_size,  auxillary_qubit_size = 1):
    
        super(Net, self).__init__()
        
        # inputs shpould be a keyword argument, for AmplitudeEmbedding !!
        # then it becomes non-differentiable and the network suits with autograd
        self.training_mode = True
        self.return_latent = False
        # This class constitutes the whole network, which includes a 
        # data embeddings, parametric quantum circuit (encoder), SWAP TEST(Calculating the fidelity)
        # Also the conj. transpose of the encoder(decoder)
        
        self.latent_space_size = latent_space_size
        self.training_qubits_size = training_qubits_size
        self.auxillary_qubit_size = auxillary_qubit_size
        self.n_qubits = n_qubits
        @qml.qnode(dev)
        def q_circuit(weights_r ,weights_cr,weights_st ,inputs = False):
            self.embedding(inputs)
            
            # qml.templates.StronglyEntanglingLayers(weights_st, range(latent_space_size+auxillary_qubit_size,n_qubits))
            
            
            # These lists holds the conj tranposes of the programmable gates
            # since we would need them in the testing
            self.first_rots = []
            self.final_rots = []
            self.cnot = []
            self.wires_list = []
            
            
            # Single rotation gates for each qubit- Number of gates = N
            # Number of parameters = N * 3 
            for i in range(self.latent_space_size + self.auxillary_qubit_size , self.n_qubits):
                ind = i - (self.latent_space_size + self.auxillary_qubit_size)
                self.first_rots.append(np.matrix((qml.Rot(*weights_r[0, ind], wires = i)).matrix).H)
                
            # Controlled rotation gates for each qubit pair- Number of gates = N(N-1)/2
            # Number of parameters = 3* N(N-1)/2
            for i in range(self.latent_space_size + self.auxillary_qubit_size , self.n_qubits):
                ind = i - (self.latent_space_size + self.auxillary_qubit_size)
                ctr=0
                for j in range(self.latent_space_size+self.auxillary_qubit_size , self.n_qubits):
                    ind_contr = j - (self.latent_space_size + self.auxillary_qubit_size)
                    if(i==j):
                        pass
                    else:
                        self.cnot.insert( len(self.cnot) , np.matrix(( qml.CRot( *weights_cr[ind,ctr]  ,wires= [i,j]).matrix )).H )
                        self.wires_list.insert( len(self.wires_list) , [i,j])
                        ctr += 1
            # Single rotation gates for each qubit- Number of gates = N
            # Number of parameters = N * 3                         
            for i in range(self.latent_space_size + self.auxillary_qubit_size , self.n_qubits):
                ind = i - (self.latent_space_size + self.auxillary_qubit_size)
                self.final_rots.append(np.matrix(qml.Rot(*weights_r[1, ind], wires = i).matrix).H)
                
            if(self.training_mode==True):
                self.SWAP_Test()
                return qml.probs(0)
            
            else:
                
                
                # In the testing, SWAP the Reference Bit and the trash states
                qml.SWAP(wires = [1,2])
                
                for i in range(self.latent_space_size + self.auxillary_qubit_size , self.n_qubits):
                    ind = i - (self.latent_space_size + self.auxillary_qubit_size)
                    qml.QubitUnitary(self.final_rots[ind], wires = i)
                    print('final', self.final_rots)
                for i in range(len(self.cnot)):
                    qml.QubitUnitary(self.cnot.pop() , wires = self.wires_list.pop())
                
                for i in range(self.latent_space_size + self.auxillary_qubit_size ,self.n_qubits):
                    ind = i - (self.latent_space_size + self.auxillary_qubit_size)
                    qml.QubitUnitary(self.first_rots[ind], wires = i)
                    print('rots', self.first_rots)
                
                return qml.probs(range(auxillary_qubit_size+latent_space_size,n_qubits ))
    
        weight_shapes = {"weights_r": (2 , training_qubits_size, 3),"weights_cr": (self.training_qubits_size,self.training_qubits_size-1 ,3), "weights_st":  (3,self.training_qubits_size,3)}
        weights_st = torch.tensor(qml.init.strong_ent_layers_uniform(3, self.training_qubits_size), requires_grad=True)
        self.qlayer = qml.qnn.TorchLayer(q_circuit, weight_shapes)
        self.DRAW_CIRCUIT_FLAG = True
        self.matrix_container = np.ndarray((self.training_qubits_size))
# =============================================================================
#         4 -Time elapsed: 0.50
#         5 -Time elapsed: 1.07
#         6 -Time elapsed: 1.92
#         7 -Time elapsed: 3.18
#         8- Time elapsed: 5.44
# =============================================================================
        # Those should be replaced with the swap test
        # self.clayer1 = torch.nn.Linear(64,2)
        # self.clayer2 = torch.nn.Linear(2,2)
        # self.softmax = torch.nn.Softmax(dim=1)
        #self.seq = torch.nn.Sequential(self.qlayer, self.clayer1)
        
    
    
    @qml.template
    def embedding(self,inputs):
        # Most Significant qubit is the ancilla, then reference, then subsystem B then subystem A
        # - Ancilla
        # - Refernce
        # - B (Trash State)
        # - A (Latent Space)
        # When normalize flag is True, features act like a prob. distribution
        
        qml.templates.AmplitudeEmbedding(inputs, wires = range(self.latent_space_size+self.auxillary_qubit_size,self.n_qubits), normalize = True,pad=(0.j))
        #qml.QubitStateVector(inputs, wires = range(n_qubits))
    @qml.template 
    
    @qml.template
    def SWAP_Test(self):
        # SWAP Test measures the similarity between 2 qubits 
        # see https://arxiv.org/pdf/quant-ph/0102001.pdf
        
        qml.Hadamard(wires = 0)
        for i in range(self.auxillary_qubit_size, self.latent_space_size + self.auxillary_qubit_size):
            qml.CSWAP(wires = [0, i, i + self.latent_space_size])
        qml.Hadamard(wires = 0)
        
    def forward(self, x, training_mode = True, return_latent = False):
        self.training_mode = training_mode
        self.return_latent = return_latent
        x =  self.qlayer(x)
        
        #printing once before training
        if(self.DRAW_CIRCUIT_FLAG):
            self.DRAW_CIRCUIT_FLAG = False
            print(self.qlayer.qnode.draw())
            # Within Torch Object, you reach the circuit with TorchObj.qnode
        
        if(training_mode == False):
            print(self.qlayer.qnode.draw())

        
        return x

