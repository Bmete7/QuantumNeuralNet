# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 14:25:10 2021

@author: burak
"""


import numpy as np
import itertools


def decompose_hamiltonian(hamiltonian,paulis):
    """ Finds the coefficients for the product state of pauli matrices that
    composes a hamiltonian, from a given hermitian

    Non-Keyword arguments:
        Hamiltonian- Hermitian matrix
    Keyword arguments:
        paulis: Pauli matrices with the given order: I,pX,pY,pZ
        
    Returns:
        coefs: Coefficients of pauli matrices, a1 III + a2 IIX + .. an ZZZ 
    """
    num_el = np.prod(list(hamiltonian.shape)) # number of elements of the hamiltonain
    assert np.log2(num_el) % 1 == 0, 'Hamiltonian is not a valid hermitian'
    
    n_qubits = np.log2(num_el) / 2
    assert n_qubits % 1 == 0, 'Hamiltonian is not a valid hermitian'
    
    
    n_qubits = int(n_qubits)
    num_el = int(num_el)
    
    pauli_prods = pauli_product_states(n_qubits, paulis)
    coefs = np.zeros(num_el , dtype = 'complex128')
    for i,prod in enumerate(pauli_prods):
        coefs[i] = np.trace(pauli_prods[i] @ hamiltonian) / 4
    
    for i in range(len(coefs)):
        assert coefs[i].imag == 0.0, 'Complex parameters are found in the coefficients, is the Hamiltonian a Hermitian matrix?'
    
    actual_coefs = np.zeros(num_el , dtype = 'float64')
    for i in range(len(coefs)):
        actual_coefs[i] = coefs[i].real
    return actual_coefs
    
def pauli_product_states(n_qubits,paulis):
    
    all_paulis = []
    
    for i in range(n_qubits):
        all_paulis.append(paulis)
    
    
    pauli_prods = []
    for combination in itertools.product(*all_paulis):
        
        prod_ind = combination[0]
        for i in range(1 , len(combination)):
            prod_ind  = np.kron(prod_ind , combination[i])
        pauli_prods.append(prod_ind)
    
    return pauli_prods