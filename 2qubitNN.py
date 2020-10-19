# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:49:07 2020

@author: burak
"""
'''

 + Restructure the NN so that it has 2 outputs, for 2 rotation angles for qubits
 + Restructure the DataLoader, so that we have 4 different targets (0,1,2,3)
 + Restructure Q-circuit, so that it has 2 qubits and 2 H, Ry gates

'''

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


class QuantumCircuit:
    '''
    interface for interactions with qCircuit
    '''
    
    def __init__(self, n_qubits, backend,shots):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = [i for i in range(n_qubits)]
        
        self.theta = qiskit.circuit.Parameter('theta')
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        print(self.theta.parameters)
        self._circuit.measure_all()
        
        self.backend = backend
        self.shots = shots
    def run(self, thetas):
        job = qiskit.execute(self._circuit,
                             self.backend,
                             shots = self.shots,
                             parameter_binds = [{self.theta: theta} for theta in thetas])
        result = job.result().get_counts(self._circuit)
        parameter_binds = [{self.theta: theta} for theta in thetas]
        counts = np.array(list(result.values()))
        states = np.array([i for i in range(2**len(self._circuit.qubits))])
        
        
        dict_qubits = {}
        dict_qubits['00'] = 0
        dict_qubits['01'] = 1
        dict_qubits['10'] = 2
        dict_qubits['11'] = 3
                
        
        
        keys = (list(result.keys()))
        values = (list(result.values()))
        
        
        new_counts  = np.zeros((2**len(self._circuit.qubits)))
        for i in range(len(keys)):
            new_counts[dict_qubits[keys[i]]] = values[i]
        
        probabilities = new_counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([probabilities])
    

    """ Hybrid quantum - classical function definition """
class HybridFunction(Function):  
    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit
        
        
        expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, result)
        
        return result
        
    @staticmethod
    def backward(ctx, grad_output):
        
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        shift_right_first = (input.clone().detach())
        shift_right_second = (input.clone().detach())
        shift_left_first = (input.clone().detach())
        shift_left_second = (input.clone().detach())
        
        
        shift_right_first[0] = (shift_right_first[0][0] + 1) * ctx.shift
        shift_right_second[0][1] = (shift_right_second[0][1] + 1) * ctx.shift
        
        shift_left_first[0][0] = (shift_left_first[0][0] - 1) * ctx.shift
        shift_left_second[0][1] = (shift_left_second[0][1] - 1) * ctx.shift
        
        gradients = []
        
        
        
        expectation_right_first =  ctx.quantum_circuit.run(shift_right_first[0].tolist())
        expectation_right_second = ctx.quantum_circuit.run(shift_right_second[0].tolist())
        expectation_left_first = ctx.quantum_circuit.run(shift_left_first[0].tolist())
        expectation_left_second = ctx.quantum_circuit.run(shift_left_second[0].tolist())

        
        gradient_first = torch.tensor([expectation_right_first]) - torch.tensor([expectation_left_first])
        gradient_second = torch.tensor([expectation_right_second]) - torch.tensor([expectation_left_second])
        
        gradient_first = gradient_first.numpy()
        gradient_second = gradient_second.numpy()

        
        gradients.append(np.absolute(gradient_first).sum())
        gradients.append(np.absolute(gradient_second).sum())
        
        
        gradients = np.array([gradients]).T
        
        return torch.tensor([gradients]).view(-1,2).float(), None, None

class Hybrid(nn.Module):
    # nn.Module has an overloaded __call__ implementation and .forward gets
    #called when Hybrid() is invoked
    """ Hybrid quantum - classical layer definition """
    
    def __init__(self, backend, shots, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(2, backend, shots)
        self.shift = shift
        
    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
# Concentrating on the first 100 samples
n_samples = 100

X_train = datasets.MNIST(root='./data', train=True, download=True,
                         transform=transforms.Compose([transforms.Resize((8,8)),transforms.ToTensor()]))

# Leaving labels 0,1,2,3
n_classes = 4
idx= []
for i in range(n_classes):
    idx = np.append(idx, [np.where(X_train.targets == i)[0][:n_samples]])

X_train.data = X_train.data[idx]
X_train.targets = X_train.targets[idx]

train_loader = torch.utils.data.DataLoader(X_train, batch_size=1, shuffle=True)


n_samples_show = 6

data_iter = iter(train_loader)
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

while n_samples_show > 0:
    images, targets = data_iter.__next__()

    axes[n_samples_show - 1].imshow(images[0].numpy().squeeze(), cmap='gray')
    axes[n_samples_show - 1].set_xticks([])
    axes[n_samples_show - 1].set_yticks([])
    axes[n_samples_show - 1].set_title("Labeled: {}".format(targets.item()))
    
    n_samples_show -= 1
    

  
n_samples = 50

X_test = datasets.MNIST(root='./data', train=False, download=True,
                        transform=transforms.Compose([transforms.Resize((8,8)),transforms.ToTensor()]))

n_classes = 4
idx= []
for i in range(n_classes):
    idx = np.append(idx, [np.where(X_train.targets == i)[0][:n_samples]])

X_test.data = X_test.data[idx]
X_test.targets = X_test.targets[idx]

test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=2)
        self.dropout = nn.Dropout2d()
        # self.fc1 = nn.Linear(256, 64)
        # self.fc2 = nn.Linear(64, 1)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 2)
        self.hybrid = Hybrid(qiskit.Aer.get_backend('qasm_simulator'), 100, np.pi / 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        x = self.hybrid(x)
        #return torch.cat((x, 1 - x), -1)
        return x
model = Net()

optimizer = optim.Adam(model.parameters(), lr=0.001)
#loss_func = nn.NLLLoss()
loss_func = nn.CrossEntropyLoss()
epochs = 20
loss_list = []

model.train()
for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        # Forward pass
        output = model(data).view(1,-1).float()
        # Calculating loss


        loss = loss_func(output, target)
        # Backward pass
        loss.backward()
        # Optimize the weights
        optimizer.step()
        
        total_loss.append(loss.item())
    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
        100. * (epoch + 1) / epochs, loss_list[-1]))


plt.plot(loss_list)
plt.title('Hybrid NN Training Convergence')
plt.xlabel('Training Iterations')
plt.ylabel('Neg Log Likelihood Loss')


with torch.no_grad():
    
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        
        pred = output.argmax(dim=1, keepdim=True) 
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        loss = loss_func(output, target)
        total_loss.append(loss.item())
        
    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / len(test_loader) * 100)
        )
    
    
    
n_samples_show = 6
count = 0
fig, axes = plt.subplots(nrows=1, ncols=n_samples_show, figsize=(10, 3))

model.eval()
with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        if count == n_samples_show:
            break
        output = model(data)
        
        pred = output.argmax(dim=1, keepdim=True) 

        axes[count].imshow(data[0].numpy().squeeze(), cmap='gray')

        axes[count].set_xticks([])
        axes[count].set_yticks([])
        axes[count].set_title('Predicted {}'.format(pred.item()))
        
        count +=1