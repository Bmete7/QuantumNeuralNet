# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:50:09 2020

@author: burak
"""

import torch

import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class IRISDataset(Dataset):
    """IRIS dataset."""

    def __init__(self, traindata,numberOfQubits,  transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.traindata = traindata
        self.traindata  = traindata
        for i in range(numberOfQubits - 2):
            self.traindata  = traindata = torch.cat((self.traindata.clone().detach(), traindata ) ,dim = 1)
            #self.traindata = torch.cat(( torch.cat(  (traindata.clone().detach(),traindata), dim = 1) ,  torch.cat(  (traindata.clone().detach(),traindata),dim = 1 )) , dim = 1)
        # self.traindata  = traindata
        self.transform = transform

    def __len__(self):
        return len(self.traindata)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        
        data = self.traindata[idx], _


        if self.transform:
            data = self.transform(data)

        return data
    
    
def importIRIS():
    dataset = pd.read_csv('../dataset/iris.csv')
    # transform species to numerics
    dataset.loc[dataset.species=='Iris-setosa', 'species'] = 0
    dataset.loc[dataset.species=='Iris-versicolor', 'species'] = 1
    dataset.loc[dataset.species=='Iris-virginica', 'species'] = 2
    
    
    train_X, test_X, train_y, test_y = train_test_split(dataset[dataset.columns[0:4]].values,
                                                        dataset.species.values, test_size=0.8)
    
    # wrap up with Variable in pytorch
    traindata = (torch.Tensor(train_X).float())
    testdata = (torch.Tensor(test_X).float())
    numberOfQubits = 2 
    myData = IRISDataset(traindata,numberOfQubits)
    dataloader = DataLoader(myData)
    return dataloader