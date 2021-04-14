# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 18:12:08 2021

@author: burak
"""
import pennylane as qml
from scipy.linalg import expm, sinm, cosm
def hamiltonian_simulate(H, t):
    """ Returns the unitary driven by some hamiltonian
    and time t
        
    Non-Keyword arguments:
        H- Hermitian Matrix
        t- Time parameter
        
    Returns:
        Unitary Matrix
    """
    return expm(H* -1j * t)
def time_evolution_simulate(H, t, psi):
    """ Returns the time evolution driven by some hamiltonian
    and time t on a specific state
        
    Non-Keyword arguments:
        H- Hermitian matrix
        i- Index of the hamiltonian in the array
        t- Time parameter
        psi- input state
        
    Returns:
        Another quantum state
    """
    return hamiltonian_simulate(H, t) @ psi