from __future__ import print_function, division
import torch
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from cnn_function import *


#load trained model
model = ModelExtraInput()
model.load_state_dict(torch.load('trained_model/model_multimass.pt'))


m_file_tab = ['dataset/data_formatted_xyz_m_1.npy','dataset/data_formatted_xyz_m_10.npy',
    'dataset/data_formatted_xyz_m_100.npy','dataset/data_formatted_xyz_m_1000.npy']
mass_tab = [1,10,100,1000]
batch_size = 100


for j in range(len(m_file_tab)):
    data = np.load(m_file_tab[j],allow_pickle=True, encoding='latin1')
    nam = 'mass = {}'.format(mass_tab[j]) #description to ass as title to figures
    n_event = len(data) #number of events
    print('total number of events:{}'.format(n_event))
    #shuffle events
    np.random.shuffle(data)

    data_max,data_mean,data_std = mean_per_channel(data['image'])  #in case want to normalize data
    dim = len(data['image'][0,:,:,0])
    depth = len(data['image'][0,0,0,:])

    all_transforms = transforms.Compose([transforms.ToTensor()])

    event_test = my_Dataset(npy_array=data,transform=all_transforms) #create test dataset

    test_loader = DataLoader(dataset=event_test, batch_size=batch_size, shuffle=True)

    test_any_model(model,test_loader,nam,batch_size,dim,depth=3)