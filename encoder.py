#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 12:51:27 2022

@author: Irene Lopez

Encoder from one-hot representation to binary representation (for 2, 3, and 4-graph journeys)

"""

import pennylane as qml
import numpy as np
N = 16
int2bit = lambda x: str(bin(x)[2:].zfill(N))
bit2int = lambda b: int("".join(str(bs) for bs in b), base = 2)

def apply_X(bitstring, i):
    """
    Applies X to qubit `i` (counting from 0) of input bitstring.
    """

    bitstring = list(bitstring)
    
    if bitstring[i] == '0':
        bitstring[i] = '1'
    else:
        bitstring[i] = '0'
        
    return ''.join(bitstring)

def apply_CNOT(bitstring, i_control, i_target):
    """
    Apply CNOT for specified control and target qubits.
    """
    bitstring = list(bitstring)
    
    if bitstring[i_control] == '1':
        bitstring = apply_X(bitstring, i_target)
    
    return ''.join(bitstring)

def apply_Toffoli(bitstring, i_control1, i_control2, i_target):
    """
    Apply Toffoli for specified controls and target.
    """
    bitstring = list(bitstring)
    
    if (bitstring[i_control1] == '1') & (bitstring[i_control2] == '1'):
        bitstring = apply_X(bitstring, i_target)

    return ''.join(bitstring)

def apply_Toffoli_custom(bitstring, i_control1, i_control2, i_control3 , i_target, control_vals):
    """
    Apply Toffoli for specified controls and target.
    """
    bitstring = list(bitstring)

    if (bitstring[i_control1] == control_vals[0]) & (bitstring[i_control2] == control_vals[1]) & (bitstring[i_control3] == control_vals[2]):
        bitstring = apply_X(bitstring, i_target)
        
    return ''.join(bitstring)




def apply_CinvToffoli(bitstring, i_control1, i_control2, i_control3, i_target):
    """
    Apply 'controlled - inverse Toffoli' for specified controls and target.
    """
    bitstring = list(bitstring)
    
    if (bitstring[i_control1] == '1') & (bitstring[i_control2] == '0') & (bitstring[i_control3] == '0'):
        bitstring = apply_X(bitstring, i_target)

    return ''.join(bitstring)


def graph2_encoder(bitstring):
    
    bitstring = apply_CNOT(bitstring, 3, 0)  # first qubit must be zero
    
    bitstring = apply_X(bitstring, 3)
    bitstring = apply_CNOT(bitstring, 3, 1)
    bitstring = apply_CNOT(bitstring, 3, 2) # second and third qubit also to zero
    
    bitstring = apply_X(bitstring, 3) # forth qubit must stay the same
    
    return bitstring

def swap(bitstring, i , j):
    if(i  < j):
        return ''.join(bitstring[:i] + bitstring[j] +  bitstring[i + 1:j] + bitstring[i] + bitstring[j + 1:] )
    if(j < i):
        return ''.join(bitstring[:j] + bitstring[i] +  bitstring[j+ 1:i] + bitstring[j] + bitstring[i + 1:] )
    
# %% 

def apply_Toffoli_customsize(bitstring, i_control1, i_control2, i_control3, i_control4 , i_target, control_vals):
    """
    Apply Toffoli for specified controls and target.
    """
    bitstring = list(bitstring)

    if ( (bitstring[i_control1] == control_vals[0] ) & (bitstring[i_control2] == control_vals[1]) & (bitstring[i_control3] == control_vals[2])  & (bitstring[i_control4] == control_vals[3]) ):
        bitstring = apply_X(bitstring, i_target)
        
    return ''.join(bitstring)

def apply_Toffoli_largersize(bitstring, i_control , i_target, control_vals):
    """
    Apply Toffoli for specified controls and target.
    """
    bitstring = list(bitstring)
    length = len(i_control)
    condition_check_list = [ bitstring[i_control[i]] == control_vals[i]  for i in range(length)]
    
    
    if(all(condition_check_list)):
        bitstring = apply_X(bitstring, i_target)
        
    return ''.join(bitstring)

# %% 

def graph4_encoder(bitstring):
    
    # 6 bit encoding, first 3 shows the position of the 0th vertex.
    # 3: 1 > 2?
    # 4: 1 > 3?
    # 5: 2 > 3?
    # 0    4    8   12  15
    # 0000 0000 0000 0000
    if(bitstring == '0010000101001000'):
        pass
        print('s')
    bitstring =  apply_Toffoli_custom(bitstring, 0, 1, 2, 3, ['0', '0', '0']) # setting the 3rd qubit to 0
    
    bitstring = apply_Toffoli(bitstring, 4, 9, 3)
    bitstring = apply_Toffoli(bitstring, 4, 10, 3)
    bitstring = apply_Toffoli(bitstring, 4, 11, 3)
    bitstring = apply_Toffoli(bitstring, 5, 10, 3)
    bitstring = apply_Toffoli(bitstring, 5, 11, 3)
    bitstring = apply_Toffoli(bitstring, 6, 11, 3)   # 3rd qubit is 1, if 1 > 2
    
    bitstring = apply_Toffoli_custom(bitstring, 8, 9, 10, 11, ['0', '0', '0'])
    
    bitstring = apply_Toffoli(bitstring, 4, 13, 11)
    bitstring = apply_Toffoli(bitstring, 4, 14, 11)
    bitstring = apply_Toffoli(bitstring, 4, 15, 11)
    bitstring = apply_Toffoli(bitstring, 5, 14, 11)
    bitstring = apply_Toffoli(bitstring, 5, 15, 11)
    bitstring = apply_Toffoli(bitstring, 6, 15, 11)   # 11rd qubit is 1, if 1 > 3 
    
    
    bitstring = apply_Toffoli_custom(bitstring, 4, 6, 7, 5, ['0', '0', '0'])
    
    bitstring = apply_Toffoli(bitstring, 8, 13, 5)
    bitstring = apply_Toffoli(bitstring, 8, 14, 5)
    bitstring = apply_Toffoli(bitstring, 8, 15, 5)
    bitstring = apply_Toffoli(bitstring, 9, 14, 5)
    bitstring = apply_Toffoli(bitstring, 9, 15, 5)
    bitstring = apply_Toffoli(bitstring, 10, 15, 5)   # 5th qubit is 1, if 2 > 3 
    
    
    
    # cleaning the last 4 qubits
    print(bitstring)
    bitstring = apply_Toffoli_customsize(bitstring, 0, 3, 11, 5, 15, ['1', '1', '1', '1']) # 0 1 2 3
    bitstring = apply_Toffoli_customsize(bitstring, 0, 3, 11, 5, 15, ['1', '0', '1', '1']) # 0 2 1 3
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 15, ['1', '1', '0', '1']) # false
    bitstring = apply_Toffoli_customsize(bitstring, 0, 3, 11, 5, 14, ['1', '1', '1', '0']) # 0 1 3 2
    bitstring = apply_Toffoli_customsize(bitstring, 0, 3, 11, 5, 14, ['1', '0', '0', '1']) # 0 2 3 1
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 14, ['1', '0', '1', '0']) # false
    bitstring = apply_Toffoli_customsize(bitstring, 0, 3, 11, 5, 13, ['1', '1', '0', '0']) # 0 3 1 2
    bitstring = apply_Toffoli_customsize(bitstring, 0, 3, 11, 5, 13, ['1', '0', '0', '0']) # 0 3 2 1
    
    # cleaning the second last 4 qubits
    
    # 8 9 10 11, 11th already cleaned
    
    bitstring = apply_Toffoli_customsize(bitstring, 0, 3, 11, 5, 10, ['1', '1', '1', '1']) # 0 1 2 3
    bitstring = apply_Toffoli_customsize(bitstring, 0, 3, 11, 5, 9, ['1', '0', '1', '1']) # 0 2 1 3
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 15, ['1', '1', '0', '1']) # false
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 14, ['1', '1', '1', '0']) # 0 1 3 2 already 0 
    bitstring = apply_Toffoli_customsize(bitstring, 0, 3, 11, 5, 9, ['1', '0', '0', '1']) # 0 2 3 1
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 14, ['1', '0', '1', '0']) # false
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 13, ['1', '1', '0', '0']) # 0 3 1 2 already 0
    bitstring = apply_Toffoli_customsize(bitstring, 0, 3, 11, 5, 10, ['1', '0', '0', '0']) # 0 3 2 1
    
    
    # cleaning the third last 4 qubits
    
    # 4 5 6 7, 5th already cleaned
    
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 5, ['1', '1', '1', '1']) # 0 1 2 3 already cleaned
    bitstring = apply_Toffoli_customsize(bitstring, 0, 3, 11, 5, 6, ['1', '0', '1', '1']) # 0 2 1 3
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 15, ['1', '1', '0', '1']) # false
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 5, ['1', '1', '1', '0']) # 0 1 3 2 already cleaned
    bitstring = apply_Toffoli_customsize(bitstring, 0, 3, 11, 5, 7, ['1', '0', '0', '1']) # 0 2 3 1
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 14, ['1', '0', '1', '0']) # false
    bitstring = apply_Toffoli_customsize(bitstring, 0, 3, 11, 5, 6, ['1', '1', '0', '0']) # 0 3 1 2
    bitstring = apply_Toffoli_customsize(bitstring, 0, 3, 11, 5, 7, ['1', '0', '0', '0']) # 0 3 2 1
    
    
    
    
    #, 0 is second, cleaning the last 4 qubits
    
    bitstring = apply_Toffoli_customsize(bitstring, 1, 3, 11, 5, 15, ['1', '1', '1', '1']) # 1 0 2 3
    bitstring = apply_Toffoli_customsize(bitstring, 1, 3, 11, 5, 15, ['1', '0', '1', '1']) # 2 0 1 3
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 15, ['1', '1', '0', '1']) # false
    bitstring = apply_Toffoli_customsize(bitstring, 1, 3, 11, 5, 14, ['1', '1', '1', '0']) # 1 0 3 2
    bitstring = apply_Toffoli_customsize(bitstring, 1, 3, 11, 5, 14, ['1', '0', '0', '1']) # 2 0 3 1
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 14, ['1', '0', '1', '0']) # false
    bitstring = apply_Toffoli_customsize(bitstring, 1, 3, 11, 5, 12, ['1', '1', '0', '0']) # 3 0 1 2
    bitstring = apply_Toffoli_customsize(bitstring, 1, 3, 11, 5, 12, ['1', '0', '0', '0']) # 3 0 2 1
    
    
    # 8 9 10 11 ( 11 already cleaned)
    
    bitstring = apply_Toffoli_customsize(bitstring, 1, 3, 11, 5, 10, ['1', '1', '1', '1']) # 1 0 2 3
    bitstring = apply_Toffoli_customsize(bitstring, 1, 3, 11, 5, 8, ['1', '0', '1', '1']) # 2 0 1 3
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 15, ['1', '1', '0', '1']) # false
    # apply_Toffoli_customsize(bitstring, 1, 3, 4, 5, 14, ['1', '1', '1', '0']) # 1 0 3 2
    bitstring = apply_Toffoli_customsize(bitstring, 1, 3, 11, 5, 8, ['1', '0', '0', '1']) # 2 0 3 1
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 14, ['1', '0', '1', '0']) # false
    # apply_Toffoli_customsize(bitstring, 1, 3, 4, 5, 13, ['1', '1', '0', '0']) # 3 0 1 2
    bitstring = apply_Toffoli_customsize(bitstring, 1, 3, 11, 5, 10, ['1', '0', '0', '0']) # 3 0 2 1
    
    # 4 5 6 7 (5 already cleaned)
    
    bitstring = apply_Toffoli_customsize(bitstring, 1, 3, 11, 5, 4, ['1', '1', '1', '1']) # 1 0 2 3
    bitstring = apply_Toffoli_customsize(bitstring, 1, 3, 11, 5, 6, ['1', '0', '1', '1']) # 2 0 1 3
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 15, ['1', '1', '0', '1']) # false
    bitstring = apply_Toffoli_customsize(bitstring, 1, 3, 11, 5, 4, ['1', '1', '1', '0']) # 1 0 3 2
    bitstring = apply_Toffoli_customsize(bitstring, 1, 3, 11, 5, 7, ['1', '0', '0', '1']) # 2 0 3 1
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 14, ['1', '0', '1', '0']) # false
    bitstring = apply_Toffoli_customsize(bitstring, 1, 3, 11, 5, 6, ['1', '1', '0', '0']) # 3 0 1 2
    bitstring = apply_Toffoli_customsize(bitstring, 1, 3, 11, 5, 7, ['1', '0', '0', '0']) # 3 0 2 1
    
    print(bitstring)
    #, 0 is third,  cleaning the last 4 qubits
    # 12 13 14 15 
    bitstring = apply_Toffoli_customsize(bitstring, 2, 3, 11, 5, 15, ['1', '1', '1', '1']) # 1 2 0 3
    bitstring = apply_Toffoli_customsize(bitstring, 2, 3, 11, 5, 15, ['1', '0', '1', '1']) # 2 1 0 3
    # apply_Toffoli_customsize(bitstring 0, 3, 4, 5, 15, ['1', '1', '0', '1']) # false
    bitstring = apply_Toffoli_customsize(bitstring, 2, 3, 11, 5, 13, ['1', '1', '1', '0']) # 1 3 0 2
    bitstring = apply_Toffoli_customsize(bitstring, 2, 3, 11, 5, 13, ['1', '0', '0', '1']) # 2 3 0 1
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 14, ['1', '0', '1', '0']) # false
    bitstring = apply_Toffoli_customsize(bitstring, 2, 3, 11, 5, 12, ['1', '1', '0', '0']) # 3 1 0 2
    bitstring = apply_Toffoli_customsize(bitstring, 2, 3, 11, 5, 12, ['1', '0', '0', '0']) # 3 2 0 1
    print(bitstring)
    
    # 8 9 10 11 (11 already cleaned)
    bitstring = apply_Toffoli_customsize(bitstring, 2, 3, 11, 5, 9, ['1', '1', '1', '1']) # 1 2 0 3
    bitstring = apply_Toffoli_customsize(bitstring, 2, 3, 11, 5, 8, ['1', '0', '1', '1']) # 2 1 0 3
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 15, ['1', '1', '0', '1']) # false
    # apply_Toffoli_customsize(bitstring, 1, 3, 4, 5, 13, ['1', '1', '1', '0']) # 1 3 0 2
    bitstring = apply_Toffoli_customsize(bitstring, 2, 3, 11, 5, 8, ['1', '0', '0', '1']) # 2 3 0 1
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 14, ['1', '0', '1', '0']) # false
    # apply_Toffoli_customsize(bitstring, 1, 3, 4, 5, 12, ['1', '1', '0', '0']) # 3 1 0 2
    bitstring = apply_Toffoli_customsize(bitstring, 2, 3, 11, 5, 9, ['1', '0', '0', '0']) # 3 2 0 1
    
    print(bitstring)
    
    
    # 4 5 6 7 (5 already cleaned)
    bitstring = apply_Toffoli_customsize(bitstring, 2, 3, 11, 5, 4, ['1', '1', '1', '1']) # 1 2 0 3
    # apply_Toffoli_customsize(bitstring, 2, 3, 4, 5, 8, ['1', '0', '1', '1']) # 2 1 0 3
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 15, ['1', '1', '0', '1']) # false
    bitstring = apply_Toffoli_customsize(bitstring, 1, 3, 11, 5, 4, ['1', '1', '1', '0']) # 1 3 0 2
    bitstring = apply_Toffoli_customsize(bitstring, 2, 3, 11, 5, 7, ['1', '0', '0', '1']) # 2 3 0 1
    # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 14, ['1', '0', '1', '0']) # false
    # apply_Toffoli_customsize(bitstring, 1, 3, 4, 5, 12, ['1', '1', '0', '0']) # 3 1 0 2
    bitstring = apply_Toffoli_customsize(bitstring, 2, 3, 11, 5, 7, ['1', '0', '0', '0']) # 3 2 0 1
    print(bitstring)
    
   #, 0 is last,  cleaning the last 4 qubits
   # This is false, Instead, form a toffoli with the form: 
   # bitstring = apply_Toffoli_custom(bitstring, 0,1,2, 3, 11, 5, 14, ['0', '0', '0', '1', '1', '1']) # 1 2 3 0
   
   # 12 13 14 15 
    
   
    bitstring = apply_Toffoli_largersize(bitstring, [0,1,2,3,11,5], 14,['0', '0', '0', '1', '1', '1']  ) # 1 2 3 0
    bitstring = apply_Toffoli_largersize(bitstring, [0,1,2,3,11,5], 14,['0', '0', '0', '0', '1', '1']  ) # 2 1 3 0
    
    bitstring = apply_Toffoli_largersize(bitstring, [0,1,2,3,11,5], 13,['0', '0', '0', '1', '1', '0']  ) # 1 3 2 0 
    bitstring = apply_Toffoli_largersize(bitstring, [0,1,2,3,11,5], 13,['0', '0', '0', '0', '0', '1']  ) # 2 3 1 0
    
    bitstring = apply_Toffoli_largersize(bitstring, [0,1,2,3,11,5], 12,['0', '0', '0', '1', '0', '0']  ) # 3 1 2 0
    bitstring = apply_Toffoli_largersize(bitstring, [0,1,2,3,11,5], 12,['0', '0', '0', '0', '0', '0']  ) # 3 2 1 0
   
    # bitstring = apply_Toffoli_custom(bitstring, 3, 11, 5, 14, ['1', '1', '1']) # 1 2 3 0
    # bitstring = apply_Toffoli_custom(bitstring, 3, 11, 5, 14, ['0', '1', '1']) # 2 1 3 0
    # # apply_Toffoli_customsize(bitstring 0, 3, 4, 5, 15, ['1', '1', '0', '1']) # false
    # bitstring = apply_Toffoli_custom(bitstring, 3, 11, 5, 13, ['1', '1', '0']) # 1 3 2 0 
    # bitstring = apply_Toffoli_custom(bitstring, 3, 11, 5, 13, ['0', '0', '1']) # 2 3 1 0
    # # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 14, ['1', '0', '1', '0']) # false
    # bitstring = apply_Toffoli_custom(bitstring, 3, 11, 5, 12, ['1', '0', '0']) # 3 1 2 0
    # bitstring = apply_Toffoli_custom(bitstring, 3, 11, 5, 12, ['0', '0', '0']) # 3 2 1 0

    # 8 9 10 11


    bitstring = apply_Toffoli_largersize(bitstring, [0,1,2,3,11,5], 9,['0', '0', '0', '1', '1', '1']  ) # 1 2 3 0
    bitstring = apply_Toffoli_largersize(bitstring, [0,1,2,3,11,5], 8,['0', '0', '0', '0', '1', '1']  ) # 2 1 3 0
    
    bitstring = apply_Toffoli_largersize(bitstring, [0,1,2,3,11,5], 10,['0', '0', '0', '1', '1', '0']  ) # 1 3 2 0 
    bitstring = apply_Toffoli_largersize(bitstring, [0,1,2,3,11,5], 8,['0', '0', '0', '0', '0', '1']  ) # 2 3 1 0
    
    bitstring = apply_Toffoli_largersize(bitstring, [0,1,2,3,11,5], 10,['0', '0', '0', '1', '0', '0']  ) # 3 1 2 0
    bitstring = apply_Toffoli_largersize(bitstring, [0,1,2,3,11,5], 9,['0', '0', '0', '0', '0', '0']  ) # 3 2 1 0

    # bitstring = apply_Toffoli_custom(bitstring, 3, 11, 5, 9, ['1', '1', '1']) # 1 2 3 0
    # bitstring = apply_Toffoli_custom(bitstring, 3, 11, 5, 8, ['0', '1', '1']) # 2 1 3 0
    # # apply_Toffoli_customsize(bitstring 0, 3, 4, 5, 15, ['1', '1', '0', '1']) # false
    # bitstring = apply_Toffoli_custom(bitstring, 3, 11, 5, 10, ['1', '1', '0']) # 1 3 2 0 
    # bitstring = apply_Toffoli_custom(bitstring, 3, 11, 5, 8, ['0', '0', '1']) # 2 3 1 0
    # # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 14, ['1', '0', '1', '0']) # false
    # bitstring = apply_Toffoli_custom(bitstring, 3, 11, 5, 10, ['1', '0', '0']) # 3 1 2 0
    # bitstring = apply_Toffoli_custom(bitstring, 3, 11, 5, 9, ['0', '0', '0']) # 3 2 1 0
    
    
    
    bitstring = apply_Toffoli_largersize(bitstring, [0,1,2,3,11,5], 4,['0', '0', '0', '1', '1', '1']  ) # 1 2 3 0
    
    
    bitstring = apply_Toffoli_largersize(bitstring, [0,1,2,3,11,5], 4,['0', '0', '0', '1', '1', '0']  ) # 1 3 2 0 
    bitstring = apply_Toffoli_largersize(bitstring, [0,1,2,3,11,5], 6,['0', '0', '0', '0', '0', '1']  ) # 2 3 1 0
    
    
    bitstring = apply_Toffoli_largersize(bitstring, [0,1,2,3,11,5], 6,['0', '0', '0', '0', '0', '0']  ) # 3 2 1 0

    
    # # 4 5 6 7 (5 already cleaned)
    # bitstring = apply_Toffoli_custom(bitstring, 3, 11, 5, 4, ['1', '1', '1']) # 1 2 3 0
    # # apply_Toffoli_custom(bitstring, 3, 11, 5, 15, ['0', '1', '1']) # 2 1 3 0
    # # apply_Toffoli_customsize(bitstring 0, 3, 4, 5, 15, ['1', '1', '0', '1']) # false
    # bitstring = apply_Toffoli_custom(bitstring, 3, 11, 5, 4, ['1', '1', '0']) # 1 3 2 0 
    # bitstring = apply_Toffoli_custom(bitstring, 3, 11, 5, 6, ['0', '0', '1']) # 2 3 1 0
    # # apply_Toffoli_customsize(bitstring, 0, 3, 4, 5, 14, ['1', '0', '1', '0']) # false
    # # apply_Toffoli_custom(bitstring, 3, 11, 5, 12, ['1', '0', '0']) # 3 1 2 0
    # bitstring = apply_Toffoli_custom(bitstring, 3, 11, 5, 6, ['0', '0', '0']) # 3 2 1 0

    print(bitstring)
    bitstring = swap(bitstring, 4, 11)
    
    return bitstring



# %% 
def graph3_encoder(bitstring):
    
    bitstring = apply_X(bitstring, 0)
    bitstring = apply_X(bitstring, 1)
    bitstring = apply_Toffoli(bitstring, 0, 1, 2) # third qubit always set to 0
    bitstring = apply_X(bitstring, 0)
    bitstring = apply_X(bitstring, 1)  # first and second qubit always stay the same
    
    
    # last bit is determined by string from qubits 4 to 9
    # if binary number from 4-6 > 7-9, then last bit 1, otherwise it's 0
    # we must consider each case individually
    
    bitstring = apply_Toffoli(bitstring, 0, 4, 2)
    bitstring = apply_Toffoli(bitstring, 1, 3, 2)
    bitstring = apply_Toffoli(bitstring, 3, 7, 2)  # now third qubit tells us if 4-6 > 7-9
    
    
    # all qubits from 4 to 9 must be set to 0
    
    bitstring = apply_Toffoli(bitstring, 0, 2, 4)
    bitstring = apply_Toffoli(bitstring, 0, 2, 8)
    
    
    bitstring = apply_Toffoli(bitstring, 1, 2, 3)
    bitstring = apply_Toffoli(bitstring, 1, 2, 8)
    
    bitstring = apply_CinvToffoli(bitstring, 2, 0, 1, 3)
    bitstring = apply_CinvToffoli(bitstring, 2, 0, 1, 7) 
    
    bitstring = apply_X(bitstring, 2)
    
    bitstring = apply_Toffoli(bitstring, 0, 2, 5)
    bitstring = apply_Toffoli(bitstring, 0, 2, 7)
    
    bitstring = apply_Toffoli(bitstring, 1, 2, 5)
    bitstring = apply_Toffoli(bitstring, 1, 2, 6)
    
    bitstring = apply_CinvToffoli(bitstring, 2, 0, 1, 4)
    bitstring = apply_CinvToffoli(bitstring, 2, 0, 1, 6) 
    
    bitstring = apply_X(bitstring, 2)
    
    return bitstring[-6:] + bitstring[:3] #reorder to put zeros in front






def main():
    
    # for graph with two vertices (only one qubit necessary in binary encoding)
    graph2_dict = {'0110': '0000', '1001': '0001'} 
    
    # test encoder
    print("Testing encoder for graph with 2-vertices: \n")
    for i in graph2_dict:
        print("One-hot encoding:", i)
        print("Binary encoding:", graph2_dict[i])
        print("Encoder output:", graph2_encoder(i))
        print("Does encoder work?:", graph2_encoder(i) == graph2_dict[i])
        print("\n")
    
    
    # for graph with three vertices (only three qubits necessary in binary encoding)z
    graph3_dict = {'001010100': '000000000', 
                   '001100010': '000000001',
                   '010001100': '000000010',
                   '010100001': '000000011',
                   '100001010': '000000100',
                   '100010001': '000000101'} 
    
    # test encoder
    print("Testing encoder for graph with 3-vertices: \n")
    for i in graph3_dict:
        print("One-hot encoding:", i)
        print("Binary encoding:", graph3_dict[i])
        print("Encoder output:", graph3_encoder(i))
        print("Does encoder work?:", graph3_encoder(i) == graph3_dict[i])
        print("\n")
# 
        
    
    # for graph with four vertices (only five qubits necessary in binary encoding)
    

res = [graph3_encoder(int2bit(i)) for i in range(2**N)]
idx = []
for i, r in enumerate(res):
    if(r[:6]=='000000'):
        idx.append(i)
        
        
eo = '0000001001001000'
eventual_output = True
def main2():
    graph4_dict = {'0001001001001000': '0000000000000000',
                   '0001001010000100': '0000000000000001',
                   '0001010000101000': '0000000000000010',
                   '0001010010000010': '0000000000000011',
                   '0001100000100100': '0000000000000100',
                   '0001100001000010': '0000000000000101',
                   '0010000101001000': '0000000000000110',
                   '0010000110000100': '0000000000000111',
                   '0010010000011000': '0000000000001000',
                   '0010010010000001': '0000000000001001',
                   '0010100000010100': '0000000000001010',
                   '0010100001000001': '0000000000001011',
                   '0100000100101000': '0000000000001100',
                   '0100000110000010': '0000000000001101',
                   '0100001000011000': '0000000000001110',
                   '0100001010000001': '0000000000001111',
                   '0100100000010010': '0000000000010000',
                   '0100100000100001': '0000000000010001',
                   '1000000100100100': '0000000000010010',
                   '1000000101000010': '0000000000010011',
                   '1000001000010100': '0000000000010100',
                   '1000001001000001': '0000000000010101',
                   '1000010000010010': '0000000000010110',
                   '1000010000100001': '0000000000010111'}
    
    
    for i in graph4_dict:
        print("One-hot encoding:", i)
        print("Binary encoding:", graph4_dict[i])
        print("Encoder output:", graph4_encoder(i))
        print("Does encoder work?:", graph4_encoder(i) == graph4_dict[i])
        
        print("\n")
    
if __name__ == '__main__':
    
    main2()
    
    
# %%

# for i in range(512):
#     if(graph3_encoder(int2bit(i)) == '000000111'):
#         print(i, int2bit(496))

# Cinv_toff = np.eye(16)
# Cinv_toff[8,8]=0
# Cinv_toff[8,9]=1
# Cinv_toff[9,8]=1
# Cinv_toff[9,9]=0



# N = 9
# int2bit = lambda x: str(bin(x)[2:].zfill(N))
# bit2int = lambda b: int("".join(str(bs) for bs in b), base = 2)

# valid_states_str = ['001010100',
#                     '001100010',
#                     '010001100',
#                     '010100001',
#                     '100001010',
#                     '100010001']

# valid_states_int = [ bit2int(states) for states in valid_states_str]

# valid_states = np.zeros(shape = (len(valid_states_int), 2**N))
# for idx, state in enumerate(valid_states_int):
#     valid_states[idx, state] = 1

# dev = qml.device('default.qubit', wires = 9)

# def runCircuit(psi):
#     res = qCirc(psi)
#     return (-1 * res+1) /2

# @qml.qnode(dev)
# def qCirc(psi):
#     qml.QubitStateVector(psi, wires = range(0,9))
#     qml.PauliX(8)
#     qml.PauliX(7)
#     qml.Toffoli(wires= [8,7,6])
#     qml.PauliX(8)
#     qml.PauliX(7)
    
#     qml.Toffoli(wires= [8,4,6])
#     qml.Toffoli(wires= [7,5,6])
#     qml.Toffoli(wires= [5,1,6])
    
#     qml.Toffoli(wires= [8,6,4])
#     qml.Toffoli(wires= [8,6,0])
    
#     qml.Toffoli(wires= [7,6,5])
#     qml.Toffoli(wires= [7,6,0])
    
#     qml.QubitUnitary(Cinv_toff, wires = [6, 8 , 7 , 5])
#     qml.QubitUnitary(Cinv_toff, wires = [6, 8 , 7 , 1])
    
#     qml.PauliX(6)
    
#     qml.Toffoli(wires= [8,6,3])
#     qml.Toffoli(wires= [8,6,1])
    
#     qml.Toffoli(wires= [7,6,3])
#     qml.Toffoli(wires= [7,6,2])
    
#     qml.QubitUnitary(Cinv_toff, wires = [6, 8 , 7 , 4])
#     qml.QubitUnitary(Cinv_toff, wires = [6, 8 , 7 , 2])
    
#     qml.PauliX(6)
#     return [qml.expval(qml.PauliZ(i)) for i in range(9)]
#     return qml.probs(range(9))