
import pennylane as qml
import torch.nn as nn


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
        
        
        
        ----------
        
        n_auxillary_qubits  : k
        
        ----------
        
        
        n_latent_qubits    : k 
        
        
        -----------
        
        
        n_qubits          : n
        
        
        ----------       
        n_total_qubits = 2 * k + n
        
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
        self._run_QAOA = False
        
        
        
        self.ctr = 0
        self.input_size = input_size
        self.embedding_mode = 'amplitude'
        self.n_layers = 3
        # self.entanglingParameters()
        
        @qml.qnode(dev)
        def qCircuit(weights_r, weights_cr, inputs = False):
            if(self.training_mode == True):
                # State Preparation
                qml.QubitStateVector(inputs, wires = range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits))
                for l in range(self.n_layers):                        
                    for idx, i in enumerate(range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)):
                        qml.Rot(*weights_r[l, 0, idx] , wires = i)
                    for idx, i in enumerate(range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)):
                        ctr=0
                        for jdx, j in enumerate(range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)):
                            if(i==j):
                                pass
                            else:
                                qml.CRot( *weights_cr[l, idx, ctr], wires= [i, j])
                                ctr += 1
                    for idx, i in enumerate(range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)):
                        qml.Rot(*weights_r[l, 1, idx] , wires = i)
                
                
                
                # if
                
                
                
                
                
                
                
                self.swapTestMulti()
                # return [qml.probs(i) for i in range(self.n_auxillary_qubits)]
                return [qml.expval(qml.PauliZ(q)) for q in range(0, self.n_auxillary_qubits )]
            else:
                
                
                self.crot_lay = []
                qml.QubitStateVector(inputs, wires = range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits))
                # for l in range(self.n_layers):
                for l in range(self.n_layers):
                    crot_cont = []
                    print(l , 'duz')
                    for idx, i in enumerate(range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)):
                        qml.Rot(*weights_r[l, 0, idx] , wires = i)
                    
                    for idx, i in enumerate(range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)):
                        ctr=0
                        for jdx, j in enumerate(range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)):
                            if(i==j):
                                pass
                            else:
                                qml.CRot( *weights_cr[l, idx, ctr], wires= [i, j])
                                crot_cont.insert(0,[idx,ctr, i , j] )
                                ctr += 1
                                
                    for idx, i in enumerate(range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)):
                        qml.Rot(*weights_r[l, 1, idx] , wires = i)
                    self.crot_lay.insert(0, crot_cont)
                
                self.swapTrashStates()
                    
                for l in range(self.n_layers-1, -1, -1):
                    print(l, 'ters')
                    for i in range(self.n_latent_qubits + self.n_auxillary_qubits , self.n_total_qubits):
                        ind = i - (self.n_latent_qubits + self.n_auxillary_qubits)
                        qml.Rot(*weights_r[l, 1, ind], wires = i).inv()
                    crot_cont = self.crot_lay[l]
                    for i in range(len(crot_cont)):
                        vals = crot_cont[i]
                        qml.CRot( *weights_cr[l, vals[0],vals[1]]  ,wires= [vals[2],vals[3]]).inv()
                    
                    for i in range(self.n_latent_qubits + self.n_auxillary_qubits , self.n_total_qubits):
                        ind = i - (self.n_latent_qubits + self.n_auxillary_qubits)
                        qml.Rot(*weights_r[l, 0, ind], wires = i).inv()
                
                # p.s in pennylane, the least logical qubit is the one at the last
                
                return qml.probs(range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits))
            
        weight_shapes = {"weights_r": (self.n_layers, 2, self.n_qubits, 3),"weights_cr": (self.n_layers, self.n_qubits, self.n_qubits-1 ,3)}
        self.qLayer = qml.qnn.TorchLayer(qCircuit, weight_shapes)
        
    
    
    def encodeQubits(self, inputs, encoding_type = 'encoder'):
        if(encoding_type == 'QAOA'):
            for i in range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits):
                qml.Hadamard(wires = i) ## QAOA starts in an equal superposition
            
        elif(encoding_type == 'encoder'):
            pass
            
    
    def forward(self, x, training_mode = True, decode_latent = False, encode_qubits = False, draw_circuit = False, run_QAOA = False): 
        
        self._training_mode = training_mode
        self._decode_latent = decode_latent
        self._encode_qubits = encode_qubits
        self._draw_circuit = draw_circuit
        self._run_QAOA = run_QAOA
        
        
        x = self.qLayer(x)
        # Normalize exp. value between [0,1] for [1,-1]
        if(training_mode == True):
            x +=  1
            x = x/2
        self.ctr += 1
        return x
    
    def hamiltonianSimulation(self, x): 
        x = self.qLayer(x)
        pass
    
    def swapTrashStates(self):
        for i in range(self.n_latent_qubits - 1, 0 - 1, -1):
            qml.SWAP(wires = [self.n_auxillary_qubits + i , self.n_auxillary_qubits + i + self.n_latent_qubits])
    
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
    
    # @qml.template
    # def embedding(self, inputs, embedding_type ='angle'):
    #     if(embedding_type == 'angle'):
    #         qml.templates.embeddings.AngleEmbedding(inputs,wires = range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits ), rotation = 'X')
    #     elif(embedding_type == 'entangler'):
    #         # random_entangling_weights = qml.init.strong_ent_layers_normal(self.n_qubits,self.n_qubits)
    #         # qml.templates.layers.StronglyEntanglingLayers(random_entangling_weights, range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)) 
    #         qml.templates.layers.StronglyEntanglingLayers(self.entangling_parameters[self.ctr % self.input_size], range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits)) 
    #     else:
    #         qml.templates.embeddings.AmplitudeEmbedding(inputs,wires = range(self.n_auxillary_qubits + self.n_latent_qubits, self.n_total_qubits ), normalize = True,pad_with=(0.j))
        
    
    # def entanglingParameters(self):
    #     self.entangling_parameters = [qml.init.strong_ent_layers_normal(self.n_qubits,self.n_qubits) for i in range(self.input_size)]
    #     # self.entangling_parameters = [qml.init.strong_ent_layers_uniform(self.n_qubits,self.n_qubits, low = np.pi/3, high = np.pi*3/2) for i in range(self.input_size)]
    
    
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
    def run_QAOA(self):
        return self._run_QAOA
    
    @run_QAOA.setter
    def run_QAOA(self, flag):
        self._run_QAOA = flag 
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