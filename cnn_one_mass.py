
from __future__ import print_function, division
import torch
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from cnn_function import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# Hyperparameters
num_epochs = 1
batch_size = 100
learning_rate = 0.001

#import data to use as dataset for training and testing (array already signal and background together, see data_format.py)
data = np.load('data_formatted_xyz_m_100.npy',allow_pickle=True, encoding='latin1')
name = '100 MeV simple 2 layer CNN, RBG-XYZ projections \n20x20, not normalized' #description to ass as title to figures

n_event = len(data) #number of events
print('total number of events:{}'.format(n_event))

#shuffle events
np.random.shuffle(data)

#split into training and test datasets
f_train = 0.7 #fraction of train events
n_train = int(0.05*n_event)
n_test = int(0.05*n_event)

data_train = data[:n_train]
data_test = data[-n_test:]

data_max,data_mean, data_std = mean_per_channel(data['image']) #in case want to normalize date
dim = len(data['image'][0,:,:,0])
depth = len(data['image'][0,0,0,:])

all_transforms = transforms.Compose([transforms.ToTensor(),uni_norm(data_max)])

event_train = my_Dataset(npy_array=data_train,transform=all_transforms) #create training dataset
event_test = my_Dataset(npy_array=data_test,transform=all_transforms) #create test dataset

train_loader_1 = DataLoader(dataset=event_train, batch_size=batch_size, shuffle=True) #create iterator for training and testinsg
test_loader_1 = DataLoader(dataset=event_test, batch_size=batch_size, shuffle=True)


model = ConvNet() #create instance of Ctorch.NN model

#run training and testing
outs , labels , images , ids = train_test_cnn_singlem(model,train_loader_1,test_loader_1,num_epochs,learning_rate,batch_size,name,dim,depth=3)

#Create histogram and ROC curves for he output on this testing dataset
area, thresh = roc_curve(labels,outs[:,1],num_epochs,learning_rate,batch_size,name)

classification = (outs[:,1]>thresh) #final classification from this network
ind_false = np.where((classification+labels)==1)[0] #indexes of misclassified events
misc_sig = np.where((classification+1)*labels==1)[0] #missed signal

n_false = len(ind_false) #number of misc. events.
n_sig = np.sum(labels) #actual number of signal events
sig_false = len(misc_sig) #number of misc. sig events


print('optimal threshold = {:.2f}, {} misclassified events ({:.2f}%), Signal acceptance of {}%'.format(thresh,n_false,n_false/n_test*100,(n_sig-sig_false)/n_sig*100))


#Show some misclassified events
find_events(ids[ind_false,:],images[ind_false,:,:,:],outs[ind_false,1])




# Save the model and plot
#torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt') 