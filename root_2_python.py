# Simple code to read ROOT tree with events and store in np array using the root_numpy python library
# Need ROOT, numpy and root_numpy to run

import ROOT as root
from root_numpy import root2array
from sys import exit

try:
    import numpy as np
except:
    print("Failed to import numpy.")
    exit()


#Method 1 with direct use of numpy
#change this with names of wanted ROOT files to be converted and corresponding target names of numpy files
fname_tab = ['ecalHits_signal_mA5MeV.root','ecalHits_signal_mA50MeV.root','ecalHits_signal_mA500MeV.root',
	'ntuple_ecal_hits_1.8e8EOT_5.root','ntuple_ecal_hits_1.8e8EOT_6.root','ntuple_ecal_hits_1.8e8EOT_7.root',
	'ntuple_ecal_hits_1.8e8EOT_8.root','ntuple_ecal_hits_1.8e8EOT_9.root','ntuple_ecal_hits_1.8e8EOT_10.root',
	'ntuple_ecal_hits_1.8e8EOT_11.root','ntuple_ecal_hits_1.8e8EOT_12.root','ntuple_ecal_hits_1.8e8EOT_13.root',
	'ntuple_ecal_hits_1.8e8EOT_14.root','ntuple_ecal_hits_1.8e8EOT_15.root']

target_name_tab = ['signal_5.npy','signal_50.npy','signal_500.npy','background_5.npy','background_6.npy',
	'background_7.npy','background_8.npy','background_9.npy','background_10.npy','background_11.npy',
	'background_12.npy','background_13.npy','background_14.npy','background_15.npy']

file_placement = 'data/'
i=0
for file in fname_tab:
	array = root2array(file)
	np.save(file_placement+target_name_tab[i] , array)
	i+=1


#Method which does not require the use of root_numpy library