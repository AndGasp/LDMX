#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset definitions, model definitions, function to run, test and evaluate models, 
to produce ROC curves and compute accuracy of models.

"""
import torch
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_format import flat_npy,plot_scatter


class my_Dataset(Dataset):
    """
    Used to define training and testing datasets
    takes in a numpy structured array containing processed events (images already created from data)
    Array need to contain the fields 'image' ((dim,dim,depth) arrays of floats), 'label' (0 or 1),
    'num' (int) and 'mass',(float) for every event

    Can also accept defined torchvision transforms to apply the data before training (minimally need to
    convert data to pytorch tensor)
    
    len defined to get number of events

    getitem defined to use dataset for iterators over events (Dataloader)
    """

    def __init__(self, npy_array, transform=None):

        self.event_frame = npy_array['image'] #image to train network on (input)
        self.label_frame = torch.from_numpy(np.array(npy_array['label'])) #label/target
        self.id_frame =np.array([npy_array['sim'],npy_array['num']]) #id to find original event
        array_m = np.zeros((len(npy_array),1))
        array_m[:,0]=npy_array['mass']
        self.mass = torch.from_numpy(array_m).double() #A' mass for every event
        self.transform = transform

    def __len__(self):
        return len(self.event_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        event = self.event_frame[idx,:,:,:]

        label = self.label_frame[idx]

        if self.transform:
            event = self.transform(event)

        return (event,label,self.id_frame[:,idx],self.mass[idx,0])



class ConvNet(torch.torch.nn.Module):
    """
    CNN with two conv. layers and 2 fully connected layers to train for single mass point
    Architecture not optimized!
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(5 * 5 * 64, 500)
        self.fc2 = torch.nn.Linear(500, 2)


    def forward(self, x):
    	out = self.layer1(x)
    	out = self.layer2(out)
    	out = out.reshape(out.size(0), -1)
    	out = self.drop_out(out)
    	out = F.relu(self.fc1(out))
    	out = self.fc2(out)
    	return out

class ModelExtraInput(torch.torch.nn.Module):
    """
    Same simple architecture as ConvNet, but take extra input as an argument (A' mass!), 
    adds it the the 2nd FCC layer and has two extra FCC layers to compute an output as 
    a function of mass
    """
    def __init__(self):
        super(ModelExtraInput, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(5 * 5 * 64, 20)
        
        self.fc2 = torch.nn.Linear(20 + 1, 60)
        self.fc3 = torch.nn.Linear(60, 2)
        
    def forward(self, x, m):

        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        x1 = out.float()
        x2 = torch.empty([len(x),1])
        x2[:,0] = m.float()
        
        out = torch.cat((x1, x2.float()), dim=1)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
        


def train_test_cnn_singlem(model,train_loader,test_loader,n_epochs,learn_rate,batch_size,nam,dim,depth=3):
    """
    takes in CNN model who accepts a single mass, the prepared iterators for training and test events,
    the number or epochs for training, the learning rate (constant for now, can be changed), the batch size for
    training, a string to title plots (nam), and the dimensions of the images used for training

    Trains the model on the training data, evaluates the accuracy on the test data at every epoch and print results

    plots a histogram of the output of the network on the test dataset for signal and for background

    returns array with output for each test event, array with label of the event (0 for back, 1 for sig), 
    array with images and array with tags to find original events, in the order in which they were fed to the network
    """
    # Loss and optimizer
    criterion =  torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)


    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(n_epochs):
        for i, (images,labels,ids,masses) in enumerate(train_loader):
            # Run the forward pass
            outputs = model(images) #predictions
            labels = labels.long()

            loss = criterion(outputs, labels) #compute loss between output vs actual label
            loss_list.append(loss.item()) #add loss for this epoch to loss list

            # Backprop and perform Adam optimisation
            optimizer.zero_grad() #all gradients equal to 0
            loss.backward() #backpropagation
            optimizer.step() 

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, n_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))

        model.eval()

        n_test = len(test_loader)*batch_size
     
        label_tab= np.zeros(n_test) #arrays to contain output information
        mod_tab = np.zeros((n_test,2))
        im_tab = np.zeros((n_test,dim,dim,depth))
        id_tab = np.zeros((n_test,2))


        with torch.no_grad(): #without modifying any parameters
            correct = 0
            total = 0
            for i,(images, labels,ids,masses) in enumerate(test_loader):

                outputs = model(images) #evaluate output on test data
                mod_outputs = F.softmax(outputs) #convert to probability
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()


                #insert outputs in arrays to plot result of training
                label_tab[i*batch_size:(i+1)*batch_size] = labels.data.numpy()
                mod_tab[i*batch_size:(i+1)*batch_size,:] = mod_outputs.data.numpy()
                im_tab[i*batch_size:(i+1)*batch_size,:,:,:] = np.swapaxes(np.swapaxes(images.numpy(),1,2),2,3)
                id_tab[i*batch_size:(i+1)*batch_size,:] = ids

                i+=1

            print('Test Accuracy of the model on the test events: {} %'.format((correct / total) * 100))

    ind_1 = np.where(label_tab==1) #events where target was signal
    ind_0 = np.where(label_tab==0) #events where target was background
    prob_sig = mod_tab[:,1] #probability of being signal
    prob_1 = prob_sig[ind_1] #outputs for signal events
    prob_0 = prob_sig[ind_0] #outputs for background events

    #produce histogram
    
    bins = np.logspace(-5,0, 50)

    #Show histogram of distribution of outputs 
    plt.hist(prob_1, bins, alpha=0.5, label='Signal')
    plt.hist(prob_0, bins, alpha=0.5, label='Background')
    plt.gca().set_xscale("log")
    plt.yscale('log')
    plt.xlabel('Output probability of signal')
    plt.text(0.001,200,'Efficiency = {:.2f}\nm_A = {} MeV\n Batch size = {}\n# Epochs = {}\nLearning rate = {}'.format((correct / total) * 100,100,batch_size,n_epochs,learn_rate))
    plt.legend(loc='upper right')
    plt.title(nam)
    plt.show()

    return mod_tab,label_tab,im_tab,id_tab


def train_test_cnn_multiple_m(model,train_loader,test_loader,n_epochs,learn_rate,batch_size,nam,dim,depth=3):
    """
    takes in CNN model who accepts mass as a extra input, the prepared iterators for training and test events,
    the number or epochs for training, the learning rate (constant for now, can be changed), the batch size for
    training, a string to title plots (nam), and the dimensions of the images used for training

    Trains the model on the training data, evaluates the accuracy on the test data at every epoch and print results

    plots a histogram of the output of the network on the test dataset for signal and for background

    returns array with output for each test event, array with label of the event (0 for back, 1 for sig), 
    array with images and array with tags to find original events, in the order in which they were fed to the network
    """
    # Loss and optimizer
    criterion =  torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)


    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    for epoch in range(n_epochs):
        for i, (images,labels,ids,masses) in enumerate(train_loader):
            # Run the forward pass
            outputs = model(images,masses/1000) #predictions
            labels = labels.long()

            loss = criterion(outputs, labels) #compute loss between output vs actual label
            loss_list.append(loss.item()) #add loss for this epoch to loss list

            # Backprop and perform Adam optimisation
            optimizer.zero_grad() #all gradients equal to 0
            loss.backward() #backpropagation
            optimizer.step() 

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, n_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))

        model.eval()

        n_test = len(test_loader)*batch_size
     
        label_tab= np.zeros(n_test)
        mod_tab = np.zeros((n_test,2))
        im_tab = np.zeros((n_test,dim,dim,depth))
        id_tab = np.zeros((n_test,2))
        m_tab = np.zeros((n_test))


        with torch.no_grad(): #without modifying any parameter
            correct = 0
            total = 0
            for i,(images, labels,ids,masses) in enumerate(test_loader):

                outputs = model(images,masses/1000)
                mod_outputs = F.softmax(outputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()


                #insert outputs in arrays to plot result of training
             
                label_tab[i*batch_size:(i+1)*batch_size] = labels.data.numpy()
                mod_tab[i*batch_size:(i+1)*batch_size,:] = mod_outputs.data.numpy()
                im_tab[i*batch_size:(i+1)*batch_size,:,:,:] = np.swapaxes(np.swapaxes(images.numpy(),1,2),2,3)
                id_tab[i*batch_size:(i+1)*batch_size,:] = ids
                m_tab[i*batch_size:(i+1)*batch_size] = masses.data.numpy()
                i+=1


            print('Test Accuracy of the model on the test events: {} %'.format((correct / total) * 100))

    ind_1 = np.where(label_tab==1) #events where target was signal
    ind_0 = np.where(label_tab==0) #events where target was background
    prob_sig = mod_tab[:,1] #probability of being signal
    prob_1 = prob_sig[ind_1] #outputs for signal events
    prob_0 = prob_sig[ind_0] #outputs for background events

    #produce histogram
    
    bins = np.logspace(-5,0, 50)

    #Show histogram of distribution of outputs 
    plt.hist(prob_1, bins, alpha=0.5, label='Signal')
    plt.hist(prob_0, bins, alpha=0.5, label='Background')
    plt.gca().set_xscale("log")
    plt.yscale('log')
    plt.xlabel('Output probability of signal')
    plt.text(0.001,200,'Efficiency = {:.2f}\nm_A = {} MeV\n Batch size = {}\n# Epochs = {}\nLearning rate = {}'.format((correct / total) * 100,100,batch_size,n_epochs,learn_rate))
    plt.legend(loc='upper right')
    plt.title(nam)
    plt.show()

    return mod_tab,label_tab,im_tab,id_tab, m_tab

def test_any_model(model,test_loader,nam,batch_size,dim,depth=3):
    """
    takes in CNN model who accepts mass as a extra input, the prepared iterators for training and test events,
    the number or epochs for training, the learning rate (constant for now, can be changed), the batch size for
    training, a string to title plots (nam), and the dimensions of the images used for training

    Trains the model on the training data, evaluates the accuracy on the test data at every epoch and print results

    plots a histogram of the output of the network on the test dataset for signal and for background

    returns array with output for each test event, array with label of the event (0 for back, 1 for sig), 
    array with images and array with tags to find original events, in the order in which they were fed to the network
    """
    # Loss and optimizer
  
    model.eval()

    n_test = len(test_loader)*batch_size
     
    label_tab= np.zeros(n_test)
    mod_tab = np.zeros((n_test,2))
    im_tab = np.zeros((n_test,dim,dim,depth))
    id_tab = np.zeros((n_test,2))
    m_tab = np.zeros((n_test))


    with torch.no_grad(): #without modifying any parameter
        correct = 0
        total = 0
        for i,(images, labels,ids,masses) in enumerate(test_loader):

            outputs = model(images,masses/1000)
            mod_outputs = F.softmax(outputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()


            #insert outputs in arrays to plot result of training
             
            label_tab[i*batch_size:(i+1)*batch_size] = labels.data.numpy()
            mod_tab[i*batch_size:(i+1)*batch_size,:] = mod_outputs.data.numpy()
            im_tab[i*batch_size:(i+1)*batch_size,:,:,:] = np.swapaxes(np.swapaxes(images.numpy(),1,2),2,3)
            id_tab[i*batch_size:(i+1)*batch_size,:] = ids
            m_tab[i*batch_size:(i+1)*batch_size] = masses.data.numpy()
            i+=1


        print('Test Accuracy of the model on the test events: {} %'.format((correct / total) * 100))

    ind_1 = np.where(label_tab==1) #events where target was signal
    ind_0 = np.where(label_tab==0) #events where target was background
    prob_sig = mod_tab[:,1] #probability of being signal
    prob_1 = prob_sig[ind_1] #outputs for signal events
    prob_0 = prob_sig[ind_0] #outputs for background events

    #produce histogram
    
    bins = np.logspace(-5,0, 50)

    #Show histogram of distribution of outputs 
    plt.hist(prob_1, bins, alpha=0.5, label='Signal')
    plt.hist(prob_0, bins, alpha=0.5, label='Background')
    plt.gca().set_xscale("log")
    plt.yscale('log')
    plt.xlabel('Output probability of signal')
    plt.text(0.001,200,'Efficiency = {:.2f}\nm_A = {} MeV'.format((correct / total) * 100,m_tab[0]))
    plt.legend(loc='upper right')
    plt.title(nam)
    plt.show()

    return mod_tab,label_tab,im_tab,id_tab,m_tab


def roc_curve(labels,output,batch_size,n_epochs,learning_rate,nam):
    """
    takes in array of labels (targets) and correspinding array of network output
    (probability of an event being signal) for test events.
    Produces ROC curve and computes area under
    Returns area under ROC and threshold for signal minimizing the number of misclassified events
    """
    n_p = 500 #number of points to plot ROC curve

    thresh_tab = np.flip(np.linspace(0,1,n_p))

    tvp = np.zeros(n_p) #to contains true positive rate
    tfp = np.zeros(n_p) #to contain false positive rate
    misc = np.zeros(n_p) #to contain number of misclassified events
    t_p_frac = np.zeros(n_p) #to contain number of misclassified events passing the cut


    for i,t in enumerate(thresh_tab):
        positives = (output>t)

        vp_ind = np.where(positives*labels>0)[0] #events where true positive
        n_vp = len(vp_ind) #number of true positives

        fp_ind = np.where(positives*(labels+1)==1)[0] #where false positives
        n_fp = len(fp_ind)

        vn_ind = np.where((positives+labels)==0)[0] #where true negatives
        n_vn = len(vn_ind)

        fn_ind = np.where((positives+1)*labels==1)[0] #where false negatives
        n_fn = len(fn_ind)

        tvp[i] = n_vp/(n_vp+n_fn)

        tfp[i] = n_fp/(n_fp+n_vn)

        misc[i] = n_fp + n_fn
        if (n_fp + n_vp)!=0:
        	t_p_frac[i] = n_vp/(n_fp + n_vp)
        else:
        	t_p_frac[i] = 0

    area_under = np.trapz(tvp,tfp) #computes area under curve using trap. rule

    #define suggested threshold where it minimises misclassified events (change?)
    threshold = thresh_tab[np.argmin(misc)]

    print('oprimal threshold = {:.3f}'.format(threshold))
    print('AU ROC = {:.4f}'.format(area_under))

    #plot ROC curve
    plt.plot(tfp,tvp,'k-')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(nam)
    plt.text(0.2,0.8,'AU ROC = {:.2f}\nOpt. thresh. = {:.2f}\nBatch size = {}\n# Epochs = {}\nLearning rate = {}'.format(area_under,threshold,batch_size,n_epochs,learning_rate))
    plt.show()

    return area_under, threshold

def accuracy(labels, outs,thresh):
    """
    Function returning accuracy over all events, acceptance of signal, and portion of events passing cut being signal
    """
    classification = (outs>thresh) #final classification from this network, 0 if background, 1 if signal
    n_tot = len(classification)

    n_false = len(np.where((classification+labels)==1)[0]) #number of misc. events.
    n_sig = np.sum(labels) #actual number of signal events
    sig_false = len(np.where((classification+1)*labels==1)[0]) #number of misc. sig events
    sig_true = n_sig - sig_false #number of properly classied signal events
    back_false = n_false - sig_false # number of background events passing cuts

    accur = np.sum(classification==labels)/n_tot #fraction of events classified correctly
    accept = sig_true/n_sig
    sign_frac = sig_true/(back_false+sig_true)
    back_sup = back_false/(n_tot-n_sig)

    return accur,accept,sign_frac,back_sup

path = 'data/ecal/'
tags = np.array([0,1,2,3,4,5,6,7,1000,10000,100000,1000000]) #tags used to identify events
files = ['background/background_0.npy','background/background_1.npy','background/background_2.npy','background/background_3.npy',
    'background/background_4.npy','background/background_6.npy','background/background_7.npy','background/background_8.npy',
    'signal/signal_1.npy','signal/signal_5.npy','signal/signal_10.npy','signal/signal_100.npy','signal/signal_1000.npy'] #correspinding file where the original event was located
names = ['back. 0','back. 1','back. 2','back. 3','back. 4','back. 6','back. 7','back. 8','Signal m=1',
    'Signal m=5','Signal m=10','Signal m=100','Signal m=1000']


def find_events(tag_tab,im_tab,score_tab,num=2):
    """
    Function to visualize misclassified events
    Takes in array of tags of events to visualize, with corresponding array of processed images
    and array of output scores
    num number of events to visualize for every 'kind' of event

    This is very unefficient, find better way of doing this
    """

    diff_tag = []
    #create list of files necessary to access all events
    for i in range(len(tag_tab)):
        t = tag_tab[i,0]
        if t not in diff_tag:
            diff_tag.append(t)

    diff_tag = np.array(diff_tag)


    for j in range(len(diff_tag)): #loop over tags
        ac_tag = diff_tag[j]
        index = np.argmin(np.abs(tags-ac_tag))
        data_original = flat_npy(files[index]) #read file corresponding to this tag

        ind_tag = np.where(tag_tab[:,0]==ac_tag)[0] #index of events originating from this file
        np.random.shuffle(ind_tag) #shuffle
        ind_good =ind_tag[:num] #pick num number of events to visualize

        num_event = tag_tab[ind_good,1] #event numbers of these events within file

        for k in range(len(ind_good)): #loop over events in this file

            ind_tab = np.where(data_original[:,1]==num_event[k])[0] #lines in file corresponding to this event

            data_actual_event = data_original[ind_tab,3:] #assemble hits corresponding to event

            data_comp = im_tab[ind_good[k],:,:,:] #find processed image corresponding to this event

            #plot event
            plot_scatter(data_actual_event,data_comp,3,title=names[index]+'\noutput={}'.format(score_tab[ind_good[k]]))



def uni_norm(array):
    """
    normalize events between 0 and 1, with respect to maximimum energy for ALL events
    i.e. keeps information on relative intensity between events

    takes in events and maximum intensity value needed for normalisation
    """
    max_val = np.amax(array+0.1)
  
    return (array+0.1)/max_val



def mean_per_channel(image_array):
    """
    Mean and std for each channel of pictures for dataset normalisation (bot used anymore, makes
    performance worse) to feed in as argument to transforms.Normalize
    """

    num_channel = len(image_array[0,0,0,:])
    mean_tab = []
    std_tab= []
    for c in range(num_channel):
        mean_tab.append(np.mean(image_array[:,:,:,c]))
        std_tab.append(np.std(image_array[:,:,:,c]))

    max_v = np.amax(image_array)

    return max_v,tuple(mean_tab),tuple(std_tab)
