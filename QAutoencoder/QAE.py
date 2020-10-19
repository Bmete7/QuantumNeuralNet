# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:26:18 2020

@author: burak
"""



import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function
from torchvision import datasets,transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit.visualization import *


'''
TODO: 
    - import MNIST DATASET, transformit to 4x4 (Check if 8x8 has eligible compute efficiency also)
    - Do a Quantum State Embedding, Each pixel-a qubit(or think of a more intelligent solution)
    - Build the programmable circuit represented on the paper, use the rotation gate approach
    - Train the network and get some results
    Exp. 21.10.2020

'''