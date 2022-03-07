# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:32:50 2020

@author: burak
"""

import matplotlib.pyplot as plt


# %%

def visualize(out,data,img_shape):
    
    #unnormalizing the output:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))
    # if(padding_op):
    #     print((out[:pad_amount].shape))
    #     unnormed_out = (out[0][:(len(output[0]) - pad_amount)].view(1,-1)  * np.sqrt((data**2).sum().numpy())).view(1,1,training_qubits_size,-1)
    # else:
    #     unnormed_out = (out  * np.sqrt((data**2).sum().numpy())).view(1,1,training_qubits_size,-1)
    
    data = data.view(1,1,img_shape,-1)
    
    axes[0].imshow(data[0].numpy().squeeze(), cmap='gray')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    out = out.view(1,1,img_shape,-1)

    axes[1].imshow(out[0].numpy().squeeze(), cmap='gray')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    plt.show()

# %% Plots of the probability distribution

def visualize_state_vec(output , string,training_qubits_size):
    rep = output.view(1,-1)
    # xPoints = [ str(np.binary_repr(i)) for i in range(0,training_qubits_size**2)]
    xPoints = [ i for i in range(0,2**training_qubits_size )]
    yPoints = [rep[0,i] for i in range(0,2**training_qubits_size )]
    plt.bar(xPoints, yPoints)
    plt.suptitle(string)
    plt.ylim(top = 1) #xmax is your value
    plt.ylim(bottom = 0.00) #xmax is your value
    plt.show()
