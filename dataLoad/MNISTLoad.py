# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:56:07 2020

@author: burak
"""
import numpy as np
import torch

from torchvision import datasets,transforms


def importMNIST(img_shape, n_samples, batch_size,latent_space_size,auxillary_qubit_size,train ):
    X_train = datasets.MNIST(root='./data', train=train, download=True,
    transform=transforms.Compose([transforms.Resize((img_shape,img_shape)),transforms.ToTensor() 
    ]))
    # We actually do not need the targets, since we are training an autoenc.
    X_train.data = X_train.data[:n_samples]
    X_train.targets = X_train.targets[:n_samples]
    
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)
    
    
    
    
    # To get number of qubits dynamically
    data,target = iter(train_loader).__next__()
    normalized = data.view(1,-1).numpy().reshape(-1)
    
    
    
    training_qubits_size = int(np.ceil(np.log2(normalized.shape[0])))
    n_qubits = training_qubits_size + latent_space_size  + auxillary_qubit_size
    
    #normalized = torch.Tensor(normalized).view(1,-1)
    
    normalized = torch.Tensor(normalized).view(1,-1)
    padding =  {}
    padding['pad_amount'] = 0
    padding['padding_op'] = False
    padding['pad_amount']  = int(2 ** (np.ceil(np.log2(training_qubits_size ** 2))) - normalized.view(-1,1).shape[0])
    
    if(padding['pad_amount'] > 0 ):
        padding['padding_op'] = True
        padding['pad_tensor'] = torch.zeros(padding['pad_amount'] )
        
    return train_loader, n_qubits, training_qubits_size, padding


