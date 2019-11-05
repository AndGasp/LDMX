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
fname_tab = ['ntuple_ecal_hits_1.8e8EOT_1.root','ntuple_ecal_hits_1.8e8EOT_2.root',
	'ntuple_ecal_hits_1.8e8EOT_3.root','ntuple_ecal_hits_1.8e8EOT_4.root','ntuple_ecal_hits_1.8e8EOT_5.root','ntuple_ecal_hits_1.8e8EOT_6.root',
	'ntuple_ecal_hits_1.8e8EOT_7.root','ntuple_ecal_hits_1.8e8EOT_8.root','ntuple_ecal_hits_1.8e8EOT_9.root','ntuple_ecal_hits_1.8e8EOT_10.root',
	'ntuple_ecal_hits_1.8e8EOT_11.root','ntuple_ecal_hits_1.8e8EOT_12.root','ntuple_ecal_hits_1.8e8EOT_13.root','ntuple_ecal_hits_1.8e8EOT_14.root',
	'ntuple_hcal_hits_1.8e8EOT_0.root','ntuple_hcal_hits_1.8e8EOT_1.root','ntuple_hcal_hits_1.8e8EOT_2.root',
	'ntuple_hcal_hits_1.8e8EOT_3.root','ntuple_hcal_hits_1.8e8EOT_4.root','hcalHits_signal_mA1MeV.root','hcalHits_signal_mA5MeV.root',
	'hcalHits_signal_mA10MeV.root','hcalHits_signal_mA50MeV.root','hcalHits_signal_mA100MeV.root','hcalHits_signal_mA500MeV.root','hcalHits_signal_mA1000MeV.root']

target_name_tab = ['background_0.npy','background_1.npy','background_2.npy','background_3.npy',
	'background_4.npy','background_5.npy','background_6.npy','background_7.npy',
	'background_8.npy','background_9.npy','background_10.npy','background_11.npy',
	'background_12.npy','background_13.npy','background_14.npy','hcal_background_0.npy',
	'hcal_background_1.npy','hcal_background_2.npy','hcal_background_3.npy','hcal_background_4.npy',
	'hcal_signal_m_1.npy','hcal_signal_m_5.npy','hcal_signal_m_10.npy','hcal_signal_m_50.npy',
	'hcal_signal_m_100.npy','hcal_signal_m_500.npy','hcal_signal_m_1000.npy']

file_placement = 'data/'
i=0
for file in fname_tab:
	array = root2array(file)
	np.save(file_placement+target_name_tab[i] , array)
	i+=1


#Method which does not require the use of root_numpy library? (does not exist for windowns and not on SLAC server)