# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 13:23:46 2021

@author: burak
"""
import pennylane as qml

angle_measurement_dev = qml.device("default.qubit", wires=2,shots = 1000)
@qml.qnode(angle_measurement_dev)
def AngleEmbeddingCircuit(inputs):
    """ Returns the measurement of real variables,
    which are used as RX gate parameters
        
    Non-Keyword arguments:
        Hamiltonian- inputs: List of Angles
        
    Returns:
        The probability of measurement outcomes of the system w.r.t PauliZ
    """
    qml.templates.embeddings.AngleEmbedding(inputs,wires = range(0,2), rotation = 'X')
    return qml.probs([0,1])

state_measurement_dev = qml.device("default.qubit", wires=2,shots = 1000)
@qml.qnode(state_measurement_dev)
def AmplitudeEmbeddingCircuit(inputs):
    """ Returns the measurement of state vectors
        
    Non-Keyword arguments:
        Hamiltonian- inputs: List of complex numbers whose sum of squares add up to 1
        
    Returns:
        The probability of measurement outcomes of the system w.r.t PauliZ
    """
    qml.templates.embeddings.AmplitudeEmbedding(inputs,wires = range(0,2), normalize = True,pad=(0.j))
    
    return qml.probs([0,1])

latents_dev = qml.device("default.qubit", wires=3,shots = 1000)
@qml.qnode(latents_dev)
def latent_circuit(inputs):
    qml.templates.embeddings.AngleEmbedding(inputs,wires = range(1,3), rotation = 'X')
    qml.Rot(*first_rot_params_0, wires = 1)
    qml.Rot(*first_rot_params_1, wires = 2)
    qml.CRot(*crot_params_01 , wires = [1,2])
    qml.CRot(*crot_params_10 , wires = [2,1])
    qml.Rot(*final_rot_params_0, wires = 1)
    qml.Rot(*final_rot_params_1, wires = 2)
    qml.SWAP(wires=[0,1])
    
    return qml.probs([2])

latents_exp_dev = qml.device("default.qubit", wires=3,shots = 1000)
@qml.qnode(latents_exp_dev)
def latent_circuit_exp(inputs):
    qml.templates.embeddings.AngleEmbedding(inputs,wires = range(1,3), rotation = 'X')
    qml.Rot(*first_rot_params_0, wires = 1)
    qml.Rot(*first_rot_params_1, wires = 2)
    qml.CRot(*crot_params_01 , wires = [1,2])
    qml.CRot(*crot_params_10 , wires = [2,1])
    qml.Rot(*final_rot_params_0, wires = 1)
    qml.Rot(*final_rot_params_1, wires = 2)
    qml.SWAP(wires=[0,1])
    
    return qml.expval(qml.PauliX(2))