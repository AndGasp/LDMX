# LDMX
ML to optimize background suppression

Train CNN on simpulation output to distinguish background from signal using Ecal data


- root_2_python.py to convert ROOT files to npy arrays
- data_format.py contains functions used to format data for training
- cnn_function.py contains functions to define transformations/datasets/models, train and test models, ROC curves etc.
- cnn_multiple_m.py example of code to train models and evaluate them
- test_masses.py example of code to load trained models and evaluate them on new data / visualize events


Note:

--ROOT files containing all hits can be found at: /nfs/slac/g/ldmx/users/lene/hits
--Corresponding already converted npy files can be found at: /afs/slac.stanford.edu/u/re/agaspert/ldmx/data
--A few already prepared datasets (datasets of 200 000 images for each
A' mass and one dataset of 400 000 images for background can be found at: /afs/slac.stanford.edu/u/re/agaspert/ldmx/dataset


1. Convert ROOT file to numpy array

ROOT file:
- contain events which have passed a trigger requirement of less than 1.5 GeV of energy deposited in the first 20 layers of the Ecal
- generated with a 4.0 GeV beam energy, using the "v9" detector geometry and should be the same samples (as per 15 Oct 2019) that are used for our PN rejection paper, in production: 
---> signal from  nfs/slac/g/ldmx/data/mc/v9/signal_bdt_training/
---> background from  nfs/slac/g/ldmx/data/mc/v9/4gev_1e_ecal_pn/4gev_1e_ecal_pn_00_3.21e13 (or additional similar subdirs)

.root file structure:
- a tree called ecalHits or hcalHits
- containing 
--- eventNumber: one per event
--- x, y, z [mm]: each with one unsorted array per event
--- E [MeV]: one unsorted array per event

Run root_2_python.py to convert ROOT tree to numpy array
needs ROOT, numpy and root_numpy library

Converts in to a numpy array of the shape (n_hits , 6)
[:,0] => event number
[:, 1] => hit number within event
[:, 2] => x position in mm
[:, 3] => y position in mm
[:, 4] => z position in mm
[:, 5] => E position in MeV

2. Format data before training
 
 #THERE IS AN ERROR THAT AFFECTS A FEW HUNDRED EVENTS THAT IS PATCHED, but needs to be resolved
 use funtion merge_arrays in data_format.py:
 
 merge_arrays(m_list,n_s_event,n_b_event,format_type,dim,depth=3,log=False)
 
 takes in arguments:
 List of A' masses you want to include as signal for training dataset
 the number of signal event wanted for each mass
 the number of background event wanted
 format_type : 'xyz' RGB image with x,y and z projections of events
               'z_split' split events in the z direction 
 dim : dimension of images
 depth : only if using 'z_split', can chose to split z axis in depth seperations
 log : only if using 'z_split'. True if want to split z axis in log-spaced bins
 
 loops over all the hits for every events and format it as an 'image' to train network on
 returns structured array with fields
 
 array['sim'] containing a number to indentify the file from witch the event came (number for background identifying the number of the file from witch it came, mass in keV for signal)
 array['num'] containing number of the event within file
 array['label'] 0 for background or 1 for signal
 array['mass'] containing the mass (actual mass for signal or random mass amongst signal masses for background)
 array['image'] containing the dim x dim x depth image
 
 3.  Train network
 
 Aleady existant code to start from:
 - cnn_multiple_masses.py to train network on dataset with multiple masses and use mass info
 
 - load wanted npy array with formatted data
 - define number of epochs, batch_size, learning_rate
 - randomize data, separate in training and test datasets, apply transofrmations (norm.), create iterators to train network
 - 2 different networks :
    ConvNet: 2 conv. layers and 2. fully connected layers
    ModelExtraInput: same as ConvNet but takes in mass as a parameter and as extra FCC to use mass as training parameter
    
 - run already written function to train networks:
  train_test_cnn_singlem to use ConvNet
  train_test_cnn_multiple_m to use ModelExtraInput
  
  takes in model, training data iterator, test data iterator, number of epochs, learning rate, batch size 
  name (string to identify figures), dimension of pictures and depth
  
  trains model for the given number of epochs, evaluate it on test data at every epoch (to check for overtraining)
  plots histogram of output of the network for signal and background
  returns array with output of the network on test events, array with labels, array with images, array with id number and array with masses
  
  save parameters of trained model to .pt file to load trained model for later use
  
  4. Evaluate model performance
 
  run roc_curve to plot roc curve, compute area under and compute threshold that minimises number of misclassified events
  run accuracy on the output of a model and the labels, for a given threshold, to quickly compute the accuracy, the signal 
  acceptance, the background suppression and the percentage of events above cuts that are signal
  
  5. Test trained model with data not in original train/test dataset
  
  Example code to load trained model and evaluate on new data in test_masses.py
