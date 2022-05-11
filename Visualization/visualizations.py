# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:32:50 2020

@author: burak
"""

import matplotlib.pyplot as plt
import numpy as np

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

def objective_function_landmark(objective_evals):
    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(objective_evals)
    ax.set_aspect('equal')
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()


def loss_plot(evals, labels, plot_type = 'loss' ):
    
    if(plot_type == 'acc'):
        title = 'Validation Accuracies'
        y_label = 'Accuracy'
    elif(plot_type == 'loss'):
        title = 'Running Average of Losses'
        y_label = 'Loss'
    else:
        raise('Wrong plot type')

    
    fig, ax = plt.subplots()
    plt.title(title, fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    for idx, evals in enumerate(evals):
        ax.plot(evals, label= labels[idx])
        
    # Create a legend for the first line.
    
    # Add the legend manually to the Axes.
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    
    plt.show()
    
def energyLandscape(epsilon, fn_eval):
    mesh = np.meshgrid(epsilon,epsilon)

    fig = plt.figure(figsize=(6,5))
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height]) 

    cp = plt.contourf(epsilon, epsilon, fn_eval )
    plt.colorbar(cp)

    ax.set_title('Expected energy landscape, p=1')
    ax.set_xlabel('gamma')
    ax.set_ylabel('beta')
    plt.show()
    
    
    