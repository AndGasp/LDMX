#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of how to load previously saved parameters of a trained model and evaluate
performance of model on new data
"""

import torch
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from cnn_function import *
from data_format import correct

def round_100(array):
    n = len(array)
    rest = int(n%100)
    return array[:-rest]

#load trained model
model = ModelExtraInput() #create instance of model
model.load_state_dict(torch.load('trained_models/model_multimass.pt')) #load previously saved parameters


m_file_tab = ['dataset/data_formatted_xyz_m_1_noback.npy','dataset/data_formatted_xyz_m_5_noback.npy','dataset/data_formatted_xyz_m_10_noback.npy',
'dataset/data_formatted_xyz_m_50_noback.npy','dataset/data_formatted_xyz_m_100_noback.npy','dataset/data_formatted_xyz_m_500_noback.npy',
'dataset/data_formatted_xyz_m_1000_noback.npy'] #list of datasets for which want to evaluate model

mass_tab = [1,5,10,50,100,500,1000,0]
batch_size = 100
num_epochs = 5
learning_rate = 0.0005
thresh = 0.83 #threshold that was found when evaluating ROC curve of network

#if havent noted network, evaluate test dataset on network again and compute threshold
"""
#find optimal threshold
file = 'dataset/data_formatted_xyz_allm.npy' #dataset used for training / testing
data = np.load(file,allow_pickle=True, encoding='latin1')

dim = len(data['image'][0,:,:,0])
depth = len(data['image'][0,0,0,:])

np.random.shuffle(data)
n_event = len(data) #number of events

all_transforms = transforms.Compose([transforms.ToTensor()])
event_test = my_Dataset(npy_array=data,transform=all_transforms) #create test dataset
test_loader = DataLoader(dataset=event_test, batch_size=batch_size, shuffle=True)

nam = 'All masses' #name to title figures

outs,labels,im_tab,id_tab,m_tab = test_any_model(model,test_loader,nam,batch_size,dim,depth=3) #evaluate model

area, thresh = roc_curve(labels,outs[:,1],num_epochs,learning_rate,batch_size,nam) #compute are under ROC and optimal threshold

accur,accept,sign_frac, back_sup = accuracy(labels,outs[:,1],thresh) #test accuracy of predictions on dataset, given a certain threshold

print('m = {},optimal threshold = {:.2f}, accuracy of {:.4f}, Signal acceptance of {:.4f}%, fraction of events above cut that are signal {:.4f}'.format(m_tab[0],thresh,accur,accept,sign_frac))
"""

for j in range(len(m_file_tab)): #evaluate ntrained network on other data

    data = np.load(m_file_tab[j],allow_pickle=True, encoding='latin1')[-400000:] #load dataset

    n_event = len(data) #number of events
    dim = len(data['image'][0,:,:,0])
    depth = len(data['image'][0,0,0,:])

    #normalize
    for i in range(n_event):
        data['image'][i,:,:,:] = uni_norm(data['image'][i,:,:,:])

    data = correct(data) #patch

    nam = 'mass = {}'.format(mass_tab[j]) #description to ass as title to figures

    #shuffle events
    np.random.shuffle(data)

    all_transforms = transforms.Compose([transforms.ToTensor()]) #extra transf. to apply to images

    event_test = my_Dataset(npy_array=data,transform=all_transforms) #create test dataset

    test_loader = DataLoader(dataset=event_test, batch_size=batch_size, shuffle=True) #create iterator

    outs,labels,im_tab,id_tab,m_tab = test_any_model(model,test_loader,nam,batch_size,dim,depth=3) #evaluate model

    accur,accept,sign_frac,back_sup = accuracy(labels,outs[:,1],thresh) #test accuracy of predictions on dataset, given a certain threshold

    print('m = {},optimal threshold = {:.2f}, accuracy of {:.4f}, Signal acceptance of {:.4f}%, fraction of events above cut that are signal {:.4f}, background sup. of {:.4f}'.format(m_tab[0],thresh,accur,accept,sign_frac,back_sup))

#Example to visualize events having a certain characteristic
#want to visualize signal events who got a low score (misclassified)
# picks num random events to visualize amongst events satisfying chosen critera
#ind_misc = np.where(outs[:,1]<0.5)[0]
#find_events(id_tab[ind_misc,:],im_tab[ind_misc,:,:,:],outs[ind_misc,1],num=20)