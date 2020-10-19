# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 23:03:42 2020

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