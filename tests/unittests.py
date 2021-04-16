# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 00:37:47 2021

@author: burak
"""
from scipy.linalg import expm, sinm, cosm
import numpy as np

def TimeEvolutionTest(hamiltonian, t, states,dataset_size,ind):
    """ Finds the coefficients for the product state of pauli matrices that
    composes a hamiltonian, from a given hermitian

    Non-Keyword arguments:
        Hamiltonian- Hamiltonian of the system
        t- Time step for the time evolution
        states- Input state psi0
    
        
    Returns:
        Returns a boolean value whether the dataset includes data which are actually
        refers to the time evolved states of some inputs
    """
    
    return expm(hamiltonian[ind] * -1j * t) @ states[ind] == states[ind + dataset_size * t]