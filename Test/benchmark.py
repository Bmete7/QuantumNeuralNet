# -*- coding: utf-8 -*-
"""
Created on Wed May 11 19:42:18 2022

@author: burak
"""

import pennylane as qml
dev_3_mixers_benchmark = qml.device("default.qubit", wires= 9 , shots= 1024)
dev_4_mixers_benchmark = qml.device("default.qubit", wires= 16 , shots= 1024)

@qml.qnode(dev_3_mixers_benchmark)
def circuit_3_mixers_benchmark(params, edges):
    '''
    QAOA ansatz interleaved with an encoder, for solving constrained optimization
    problems on 3-vertex graphs. Even though it is implemented for 3-vertices, 
    the ansatz can easily be generalized into a d-vertex setting

    Parameters
    ----------
    params : list
        gammas and betas, parameters for cost and mixer unitaries
    
    edge : list, optional
        list of edges in the graph
    feature_vector : array, optional
    
        quantum state vector for state preparation
    test_mode : bool
        if true, it returns a valid solution encoding, that can be converted into a route
        if not, it returns the expectation value of the cost hamiltonian
    Returns
    -------
    TYPE
        either the measurement of the cost hamiltonian, or the state vector of the latent space, depending on the test_mode flag

    '''
    
    

    n = 3
    for edge in edges:
        u,v = edge
        
        q1 = u * n 
        q2 = v * n 
        for s1 in [True, False]:
            
            for s2 in [True, False]:
                for s3 in [True, False]:
                    
                    if(s1):
                        qml.Hadamard(q1)
                        qml.Hadamard(q2)
                    else:
                        qml.RX(params[0], wires = q1)
                        qml.RX(params[0], wires = q2)
                    if(s2):
                        qml.Hadamard(q1 + 1)
                        qml.Hadamard(q2 + 1)
                    else:
                        qml.RX(params[0], wires = q1 + 1)
                        qml.RX(params[0], wires = q2 + 1)
                    
                    if(s3):
                        qml.Hadamard(q1 + 2)
                        qml.Hadamard(q2 + 2)
                    else:
                        qml.RX(params[0], wires = q1 + 2)
                        qml.RX(params[0], wires = q2 + 2)
                    
                    qml.CNOT(wires= [q1, q1+1])
                    qml.CNOT(wires= [q1+1, q1+2])
                    qml.CNOT(wires= [q1+2, q2])
                    qml.CNOT(wires= [q2, q2+1])
                    qml.CNOT(wires= [q2+1, q2+2])
                    
                    qml.RZ(params[0], wires = q2 + 2)
                    
                    qml.CNOT(wires= [q1, q1+1])
                    qml.CNOT(wires= [q1+1, q1+2])
                    qml.CNOT(wires= [q1+2, q2])
                    qml.CNOT(wires= [q2, q2+1])
                    qml.CNOT(wires= [q2+1, q2+2])
                    
                    if(s3):
                        qml.Hadamard(q1 + 2).inv()
                        qml.Hadamard(q2 + 2).inv()
                    else:
                        qml.RX(params[0], wires = q1 + 2).inv()
                        qml.RX(params[0], wires = q2 + 2).inv()    
                    
                    if(s2):
                        qml.Hadamard(q1 + 1).inv()
                        qml.Hadamard(q2 + 1).inv()
                    else:
                        qml.RX(params[0], wires = q1 + 1).inv()
                        qml.RX(params[0], wires = q2 + 1).inv()
                    
                    if(s1):
                        qml.Hadamard(q1).inv()
                        qml.Hadamard(q2).inv()
                    else:
                        qml.RX(params[0], wires = q1).inv()
                        qml.RX(params[0], wires = q2).inv()
                    
    return qml.probs(wires = 0)


@qml.qnode(dev_4_mixers_benchmark)
def circuit_4_mixers_benchmark(params, edges):
    '''
    QAOA ansatz interleaved with an encoder, for solving constrained optimization
    problems on 3-vertex graphs. Even though it is implemented for 3-vertices, 
    the ansatz can easily be generalized into a d-vertex setting

    Parameters
    ----------
    params : list
        gammas and betas, parameters for cost and mixer unitaries
    
    edge : list, optional
        list of edges in the graph
    feature_vector : array, optional
    
        quantum state vector for state preparation
    test_mode : bool
        if true, it returns a valid solution encoding, that can be converted into a route
        if not, it returns the expectation value of the cost hamiltonian
    Returns
    -------
    TYPE
        either the measurement of the cost hamiltonian, or the state vector of the latent space, depending on the test_mode flag

    '''
    
    n = 4

   
    for edge in edges:
        u,v = edge
        
        q1 = u * n
        q2 = v * n
        for s1 in [True, False]:
            for s2 in [True, False]:
                for s3 in [True, False]:
                    for s4 in [True, False]:
                    
                        if(s1):
                            qml.Hadamard(q1)
                            qml.Hadamard(q2)
                        else:
                            qml.RX(params[0], wires = q1)
                            qml.RX(params[0], wires = q2)
                        if(s2):
                            qml.Hadamard(q1 + 1)
                            qml.Hadamard(q2 + 1)
                        else:
                            qml.RX(params[0], wires = q1 + 1)
                            qml.RX(params[0], wires = q2 + 1)
                        
                        if(s3):
                            qml.Hadamard(q1 + 2)
                            qml.Hadamard(q2 + 2)
                        else:
                            qml.RX(params[0], wires = q1 + 2)
                            qml.RX(params[0], wires = q2 + 2)
                        
                        if(s4):
                            qml.Hadamard(q1 + 3)
                            qml.Hadamard(q2 + 3)
                        else:
                            qml.RX(params[0], wires = q1 + 3)
                            qml.RX(params[0], wires = q2 + 3)
                            
                        qml.CNOT(wires= [q1, q1+1])
                        qml.CNOT(wires= [q1+1, q1+2])
                        qml.CNOT(wires= [q1+2, q1+3])
                        qml.CNOT(wires= [q1+3, q2])
                        qml.CNOT(wires= [q2, q2+1])
                        qml.CNOT(wires= [q2+1, q2+2])
                        qml.CNOT(wires= [q2+2, q2+3])
                        
                        qml.RZ(params[0], wires = q2 + 3)
                        
                        qml.CNOT(wires= [q1, q1+1])
                        qml.CNOT(wires= [q1+1, q1+2])
                        qml.CNOT(wires= [q1+2, q1+3])
                        qml.CNOT(wires= [q1+3, q2])
                        qml.CNOT(wires= [q2, q2+1])
                        qml.CNOT(wires= [q2+1, q2+2])
                        qml.CNOT(wires= [q2+2, q2+3])
                        
                        if(s4):
                            qml.Hadamard(q1 + 3).inv()
                            qml.Hadamard(q2 + 3).inv()
                        else:
                            qml.RX(params[0], wires = q1 + 3).inv()
                            qml.RX(params[0], wires = q2 + 3).inv()
                            
                        if(s3):
                            qml.Hadamard(q1 + 2).inv()
                            qml.Hadamard(q2 + 2).inv()
                        else:
                            qml.RX(params[0], wires = q1 + 2).inv()
                            qml.RX(params[0], wires = q2 + 2).inv()    
                        
                        if(s2):
                            qml.Hadamard(q1 + 1).inv()
                            qml.Hadamard(q2 + 1).inv()
                        else:
                            qml.RX(params[0], wires = q1 + 1).inv()
                            qml.RX(params[0], wires = q2 + 1).inv()
                        
                        if(s1):
                            qml.Hadamard(q1).inv()
                            qml.Hadamard(q2).inv()
                        else:
                            qml.RX(params[0], wires = q1).inv()
                            qml.RX(params[0], wires = q2).inv()
                    
    return qml.probs(wires = 0)
