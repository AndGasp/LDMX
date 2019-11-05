#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example code of how to create dataset, train model and evaluate it.
"""

import torch
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from cnn_function import *
from data_format import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# Hyperparameters
num_epochs = 5
batch_size = 100
learning_rate = 0.0005

#create images out of list of hits if not already done, for desired signal and background events
#formatted_array = merge_arrays([1,100],200000,400000,'xyz',20)

#save dataset to reuse
#name_save = 'dataset/data_formatted_xyz_2ms.npy'
#np.save(name_save,formatted_array)

#import data to use as dataset for training and testing 
data = np.load('dataset/data_formatted_xyz_allm.npy',allow_pickle=True, encoding='latin1')
name = 'simple 2 layer CNN with extra mass input, RBG-XYZ projections \n20x20, not normalized' #description to ass as title to figures

n_event = len(data) #number of events
print('total number of events:{}'.format(n_event))
dim = len(data['image'][0,:,:,0])
depth = len(data['image'][0,0,0,:])

#shuffle events
np.random.shuffle(data)

#normalize each event with respect to the max of the event
for i in range(n_event):
	data['image'][i,:,:,:] = uni_norm(data['image'][i,:,:,:])
data = correct(data)

#split into training and test datasets
f_train = 0.7 #fraction of train events
n_train = int(f_train*n_event)
n_test = int((1-f_train)*n_event)

data_train = data[:n_train]
data_test = data[-n_test:]

#Compose extra transformation to apply to data, minimally need to convert image to tensor
all_transforms = transforms.Compose([transforms.ToTensor()])

#Initiate training and test datasets
event_train = my_Dataset(npy_array=data_train,transform=all_transforms) #create training dataset
event_test = my_Dataset(npy_array=data_test,transform=all_transforms) #create test dataset

#Create event iterators
train_loader_1 = DataLoader(dataset=event_train, batch_size=batch_size, shuffle=True) #create iterator for training and testinsg
test_loader_1 = DataLoader(dataset=event_test, batch_size=batch_size, shuffle=True)


model = ModelExtraInput() #create instance of model
#If want to train without using mass information, use ConvNet()

#run training and testing
outs , labels , images , ids, masses = train_test_cnn_multiple_m(model,train_loader_1,test_loader_1,num_epochs,learning_rate,batch_size,name,dim,depth=3)
#if using ConvNet(), need to use train_test_cnn_single_m ,  same arguments


#Create histogram and ROC curves for he output on this testing dataset
area, thresh = roc_curve(labels,outs[:,1],num_epochs,learning_rate,batch_size,name)

#computes fraction of correctly classified events, fraction of signal passing cuts
#and fraction of events passing cuts that are actually signal
accur,accept,sign_frac,back_sup = accuracy(labels,outs[:,1],thresh)


print('optimal threshold = {:.4f}, {:.4f} misclassified events ({:.4f}%), accuracy of {:.4f}, Signal acceptance of {:.4f}%, background sup. of {:.4f}'.format(thresh,n_false,accur,accept,sign_frac,back_sup))

#save parameters of trained model
torch.save(model.state_dict(), 'trained_models/model_multimass.pt')

"""
#Show some misclassified events
find_events(ids[ind_false,:],images[ind_false,:,:,:],outs[ind_false,1])
"""