# -*- coding: utf-8 -*-
"""
Created on Wed May 11 19:47:55 2022

@author: burak
"""
import pennylane as qml
from QAutoencoder.Utils import *


def U_C(params, n, l, edges):  
    for edge in edges:
        u,v = edge
        for step in range(n - 1):
            q_1 = (u * n) + step
            q_2 = (v * n) + step + 1 # (+1 because we want that node visited in the next step)
            
            q_1_rev = (u * n) + step + 1 # (reverse path from above)
            q_2_rev = (v * n) + step  

            qml.CNOT(wires=[q_1, q_2])
            qml.RZ(params[l], wires= q_2)
            qml.CNOT(wires=[q_1, q_2])
            
            qml.CNOT(wires=[q_1_rev, q_2_rev])
            qml.RZ(params[l], wires= q_2_rev)
            qml.CNOT(wires=[q_1_rev, q_2_rev])    

def U_M(params, ancillary, n_layers, l):
    for i in range(ancillary):
        qml.RX(params[n_layers + l], wires = [i])



def discardQubits(n, l):
    '''
    Sub-circuit to swap the trash state B and a reference state B prime

    Parameters
    ----------
    n : number of vertices
    l : number of QAOA iteration cycles

    '''
    
    ancillas = calculateRequiredAncillaries(n)
    N = n ** 2 
    offset = N - ancillas 
    for i in range(offset):
        qml.SWAP(wires = [(offset * l) + ancillas + i, (offset * l) + N + i])

def U_E(n):
    '''
    Encoder ansatz for n-vertex Graphs

    Parameters
    ----------
    n : Number of vertices

    '''
    # TODO: automatize encoder generation for n-vertex graph
    if(n == 3):
        qml.MultiControlledX([0,1], [2], '00')
        
        qml.Toffoli(wires = [0, 4, 2])
        qml.Toffoli(wires = [1, 3, 2])
        qml.Toffoli(wires = [3, 7, 2])
    
        qml.Toffoli(wires = [0, 2, 4])
        qml.Toffoli(wires = [0, 2, 8])
        
        qml.Toffoli(wires = [1, 2, 3])
        qml.Toffoli(wires = [1, 2, 8])
    
        qml.MultiControlledX([2,0,1], [3], '100')
        qml.MultiControlledX([2,0,1], [7], '100')
        
        qml.PauliX(2)
        
        qml.Toffoli(wires = [0, 2, 5])
        qml.Toffoli(wires = [0, 2, 7])
        
        qml.Toffoli(wires = [1, 2, 5])
        qml.Toffoli(wires = [1, 2, 6])
        
        qml.MultiControlledX([2,0,1], [4], '100')
        qml.MultiControlledX([2,0,1], [6], '100')
        
        qml.PauliX(2)
    
    elif(n == 4):
        
        qml.MultiControlledX([0,1,2], [3], '000')
            
        qml.Toffoli(wires = [4, 9, 3])
        qml.Toffoli(wires = [4, 10, 3])
        qml.Toffoli(wires = [4, 11, 3])
        qml.Toffoli(wires = [5, 10, 3])
        qml.Toffoli(wires = [5, 11, 3])
        qml.Toffoli(wires = [6, 11, 3])
        
        qml.MultiControlledX([8,9,10], [11], '000')
        
        qml.Toffoli(wires = [4, 13, 11])
        qml.Toffoli(wires = [4, 14, 11])
        qml.Toffoli(wires = [4, 15, 11])
        qml.Toffoli(wires = [5, 14, 11])
        qml.Toffoli(wires = [5, 15, 11])
        qml.Toffoli(wires = [6, 15, 11])
        
        qml.MultiControlledX([4, 6, 7], [5], '000')
        
        qml.Toffoli(wires = [8, 13, 5])
        qml.Toffoli(wires = [8, 14, 5])
        qml.Toffoli(wires = [8, 15, 5])
        qml.Toffoli(wires = [9, 14, 5])
        qml.Toffoli(wires = [9, 15, 5])
        qml.Toffoli(wires = [10, 15, 5])
               
        qml.MultiControlledX([0, 3, 11, 5], [15], '1111')
        qml.MultiControlledX([0, 3, 11, 5], [15], '1011')
        
        qml.MultiControlledX([0, 3, 11, 5], [14], '1110')
        qml.MultiControlledX([0, 3, 11, 5], [14], '1001')
        
        qml.MultiControlledX([0, 3, 11, 5], [13], '1100')
        qml.MultiControlledX([0, 3, 11, 5], [13], '1000')
        
        qml.MultiControlledX([0, 3, 11, 5], [10], '1111')
        qml.MultiControlledX([0, 3, 11, 5], [9], '1011')
        
        qml.MultiControlledX([0, 3, 11, 5], [9], '1001')
        qml.MultiControlledX([0, 3, 11, 5], [10], '1000')
        
        qml.MultiControlledX([0, 3, 11, 5], [6], '1011')
        qml.MultiControlledX([0, 3, 11, 5], [7], '1001')
        
        qml.MultiControlledX([0, 3, 11, 5], [6], '1100')
        qml.MultiControlledX([0, 3, 11, 5], [7], '1000')
        
        qml.MultiControlledX([1, 3, 11, 5], [15], '1111')
        qml.MultiControlledX([1, 3, 11, 5], [15], '1011')
        
        qml.MultiControlledX([1, 3, 11, 5], [14], '1110')
        qml.MultiControlledX([1, 3, 11, 5], [14], '1001')
        
        qml.MultiControlledX([1, 3, 11, 5], [12], '1100')
        qml.MultiControlledX([1, 3, 11, 5], [12], '1000')        
        
        qml.MultiControlledX([1, 3, 11, 5], [10], '1111')
        qml.MultiControlledX([1, 3, 11, 5], [8], '1011')
        
        qml.MultiControlledX([1, 3, 11, 5], [8], '1001')
        qml.MultiControlledX([1, 3, 11, 5], [10], '1000')        
        
        qml.MultiControlledX([1, 3, 11, 5], [4], '1111')
        qml.MultiControlledX([1, 3, 11, 5], [6], '1011')
        
        qml.MultiControlledX([1, 3, 11, 5], [4], '1110')
        qml.MultiControlledX([1, 3, 11, 5], [7], '1001')
        
        qml.MultiControlledX([1, 3, 11, 5], [6], '1100')
        qml.MultiControlledX([1, 3, 11, 5], [7], '1000')        
        
        qml.MultiControlledX([2, 3, 11, 5], [15], '1111')
        qml.MultiControlledX([2, 3, 11, 5], [15], '1011')
        
        qml.MultiControlledX([2, 3, 11, 5], [13], '1110')
        qml.MultiControlledX([2, 3, 11, 5], [13], '1001')
        
        qml.MultiControlledX([2, 3, 11, 5], [12], '1100')
        qml.MultiControlledX([2, 3, 11, 5], [12], '1000')        
        
        qml.MultiControlledX([2, 3, 11, 5], [9], '1111')
        qml.MultiControlledX([2, 3, 11, 5], [8], '1011')
        
        qml.MultiControlledX([2, 3, 11, 5], [8], '1001')
        qml.MultiControlledX([2, 3, 11, 5], [9], '1000')      
        
        qml.MultiControlledX([2, 3, 11, 5], [4], '1111')
        qml.MultiControlledX([2, 3, 11, 5], [4], '1110')
        
        qml.MultiControlledX([2, 3, 11, 5], [7], '1001')
        qml.MultiControlledX([2, 3, 11, 5], [7], '1000')      
        
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [14], '000111')
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [14], '000011')
        
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [13], '000110')
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [13], '000001')
        
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [12], '000100')
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [12], '000000')
        
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [9], '000111')
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [8], '000011')
        
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [10], '000110')
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [8], '000001')
        
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [10], '000100')
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [9], '000000')
                
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [4], '000111')
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [4], '000110')
        
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [6], '000001')
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [6], '000000')
        
        qml.SWAP(wires = [4,11])

def U_D(n):
    '''
    Decoding ansatz, basically the conjuagte transpose of encoder ansatz

    Parameters
    ----------
    n : number of qubits

    '''
    if(n == 3):
        qml.PauliX(2).inv()
        
        qml.MultiControlledX([2,0,1], [6], '100').inv()
        qml.MultiControlledX([2,0,1], [4], '100').inv()
        
        qml.Toffoli(wires = [1, 2, 6]).inv()
        qml.Toffoli(wires = [1, 2, 5]).inv()
        
        qml.Toffoli(wires = [0, 2, 7]).inv()
        qml.Toffoli(wires = [0, 2, 5]).inv()
        
        qml.PauliX(2).inv()
        
        qml.MultiControlledX([2,0,1], [7], '100').inv()
        qml.MultiControlledX([2,0,1], [3], '100').inv()
    
        qml.Toffoli(wires = [1, 2, 8]).inv()
        qml.Toffoli(wires = [1, 2, 3]).inv()
        
        qml.Toffoli(wires = [0, 2, 8]).inv()
        qml.Toffoli(wires = [0, 2, 4]).inv()
    
        qml.Toffoli(wires = [3, 7, 2]).inv()
        qml.Toffoli(wires = [1, 3, 2]).inv()
        qml.Toffoli(wires = [0, 4, 2]).inv()
        
        qml.MultiControlledX([0,1], [2], '00').inv()
    
    if(n == 4):
        qml.SWAP(wires = [4,11]).inv()
        
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [6], '000000').inv()
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [6], '000001').inv()
        
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [4], '000110').inv()
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [4], '000111').inv()
        
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [9], '000000').inv()
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [10], '000100').inv()
        
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [8], '000001').inv()
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [10], '000110').inv()
        
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [8], '000011').inv()
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [9], '000111').inv()
        
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [12], '000000').inv()
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [12], '000100').inv()
        
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [13], '000001').inv()
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [13], '000110').inv()
        
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [14], '000011').inv()
        qml.MultiControlledX([0, 1, 2, 3, 11, 5], [14], '000111').inv()
        
        qml.MultiControlledX([2, 3, 11, 5], [7], '1000').inv()
        qml.MultiControlledX([2, 3, 11, 5], [7], '1001').inv()
        
        qml.MultiControlledX([2, 3, 11, 5], [4], '1110').inv()
        qml.MultiControlledX([2, 3, 11, 5], [4], '1111').inv()
        
        qml.MultiControlledX([2, 3, 11, 5], [9], '1000').inv()
        qml.MultiControlledX([2, 3, 11, 5], [8], '1001').inv()
        
        qml.MultiControlledX([2, 3, 11, 5], [8], '1011').inv()
        qml.MultiControlledX([2, 3, 11, 5], [9], '1111').inv()
        
        qml.MultiControlledX([2, 3, 11, 5], [12], '1000').inv()
        qml.MultiControlledX([2, 3, 11, 5], [12], '1100').inv()
        
        qml.MultiControlledX([2, 3, 11, 5], [13], '1001').inv()
        qml.MultiControlledX([2, 3, 11, 5], [13], '1110').inv()
        
        qml.MultiControlledX([2, 3, 11, 5], [15], '1011').inv()
        qml.MultiControlledX([2, 3, 11, 5], [15], '1111').inv()
        
        qml.MultiControlledX([1, 3, 11, 5], [7], '1000').inv()
        qml.MultiControlledX([1, 3, 11, 5], [6], '1100').inv()
        
        qml.MultiControlledX([1, 3, 11, 5], [7], '1001').inv()
        qml.MultiControlledX([1, 3, 11, 5], [4], '1110').inv()
        
        qml.MultiControlledX([1, 3, 11, 5], [6], '1011').inv()
        qml.MultiControlledX([1, 3, 11, 5], [4], '1111').inv()
        
        qml.MultiControlledX([1, 3, 11, 5], [10], '1000').inv()
        qml.MultiControlledX([1, 3, 11, 5], [8], '1001').inv()
        
        qml.MultiControlledX([1, 3, 11, 5], [8], '1011').inv()
        qml.MultiControlledX([1, 3, 11, 5], [10], '1111').inv()
        
        qml.MultiControlledX([1, 3, 11, 5], [12], '1000').inv()
        qml.MultiControlledX([1, 3, 11, 5], [12], '1100').inv()
        
        qml.MultiControlledX([1, 3, 11, 5], [14], '1001').inv()
        qml.MultiControlledX([1, 3, 11, 5], [14], '1110').inv()
        
        qml.MultiControlledX([1, 3, 11, 5], [15], '1011').inv()
        qml.MultiControlledX([1, 3, 11, 5], [15], '1111').inv()
        
        qml.MultiControlledX([0, 3, 11, 5], [7], '1000').inv()
        qml.MultiControlledX([0, 3, 11, 5], [6], '1100').inv()
        
        qml.MultiControlledX([0, 3, 11, 5], [7], '1001').inv()
        qml.MultiControlledX([0, 3, 11, 5], [6], '1011').inv()
        
        qml.MultiControlledX([0, 3, 11, 5], [10], '1000').inv()
        qml.MultiControlledX([0, 3, 11, 5], [9], '1001').inv()
        
        qml.MultiControlledX([0, 3, 11, 5], [9], '1011').inv()
        qml.MultiControlledX([0, 3, 11, 5], [10], '1111').inv()
        
        qml.MultiControlledX([0, 3, 11, 5], [13], '1000').inv()
        qml.MultiControlledX([0, 3, 11, 5], [13], '1100').inv()
        
        qml.MultiControlledX([0, 3, 11, 5], [14], '1001').inv()
        qml.MultiControlledX([0, 3, 11, 5], [14], '1110').inv()
        
        qml.MultiControlledX([0, 3, 11, 5], [15], '1011').inv()
        qml.MultiControlledX([0, 3, 11, 5], [15], '1111').inv()
        
        qml.Toffoli(wires = [10, 15, 5]).inv()
        qml.Toffoli(wires = [9, 15, 5]).inv()
        qml.Toffoli(wires = [9, 14, 5]).inv()
        qml.Toffoli(wires = [8, 15, 5]).inv()
        qml.Toffoli(wires = [8, 14, 5]).inv()
        qml.Toffoli(wires = [8, 13, 5]).inv()
        
        qml.MultiControlledX([4, 6, 7], [5], '000').inv()
        
        qml.Toffoli(wires = [6, 15, 11]).inv()
        qml.Toffoli(wires = [5, 15, 11]).inv()
        qml.Toffoli(wires = [5, 14, 11]).inv()
        qml.Toffoli(wires = [4, 15, 11]).inv()
        qml.Toffoli(wires = [4, 14, 11]).inv()
        qml.Toffoli(wires = [4, 13, 11]).inv()
        
        qml.MultiControlledX([8,9,10], [11], '000').inv()
        
        qml.Toffoli(wires = [6, 11, 3]).inv()
        qml.Toffoli(wires = [5, 11, 3]).inv()
        qml.Toffoli(wires = [5, 10, 3]).inv()
        qml.Toffoli(wires = [4, 11, 3]).inv()
        qml.Toffoli(wires = [4, 10, 3]).inv()
        qml.Toffoli(wires = [4, 9, 3]).inv()
        
        qml.MultiControlledX([0,1,2], [3], '000').inv()
        
def calculateRequiredAncillaries(n):
    '''
    Given a vertex number, this method returns the number of required
    ancillary qubits to solve a constraint satisfaction problem

    Parameters
    ----------
    n : int
        

    Returns
    -------
    n_ancillary: int

    '''
    if(n <= 2):
        return 1
    return int(np.ceil(np.log(numpy.math.factorial((n - 1)) * (n - 2))) + n - 1)
