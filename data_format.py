#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
format data before training
Andrea Gaspert 10/2019 agaspert@stanford.edu
"""

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

#maxima of x,y and z positions in Ecal as defined in outputs of sim., to define lattice params
x_min, x_max = -200,200
y_min, y_max = -200,200
z_min, z_max = -200,200

m_tab = np.array([1,5,10,50,100,500,1000])
path='data/ecal/'
sig_tab = ['signal/signal_1.npy','signal/signal_5.npy','signal/signal_10.npy','signal/signal_50.npy','signal/signal_100.npy','signal/signal_500.npy','signal/signal_1000.npy']
back_tab = ['background/background_0.npy','background/background_1.npy','background/background_2.npy','background/background_3.npy','background/background_4.npy',
	'background/background_5.npy','background/background_6.npy','background/background_7.npy','background/background_8.npy','background/background_9.npy',
	'background/background_10.npy','background/background_11.npy','background/background_12.npy','background/background_13.npy','background/background_14.npy']


def total_hits(npy_array):
	"""
	Counts total number of event and Ecal hits in npy array converted from ROOT
	"""
	n_event = np.size(npy_array)

	n_tot = 0

	for j in range(n_event): 
		n_tot+=npy_array[j][1]

	return n_event, n_tot


def flat_npy(file_nam,tag=0,m=0):
	"""
	Function to convert output of ROOT conversion function into a (n_event x 6) array. 
	(0 = number to trace original sim output file (number for background, mass for signal),
	1 = event number within that file, 2 = mass (0 for background), 3 = x position (mm), 4 = y position (mm), 5 = z position (mm), 6 = Energy)
	"""
	npy_array = np.load(path+file_nam,allow_pickle=True, encoding='latin1')

	n_e , n_t = total_hits(npy_array)

	data_flat = np.zeros((n_t,7))

	hits=0

	k = 0 

	for j in range(n_e):
		hits = npy_array[j][1]
		data_flat[k:k+hits,0] = tag #number of back. file for background and mass in eV for signal
		data_flat[k:k+hits,1] = np.ones(hits, dtype=int)*(j+1) #event number in original output file
		data_flat[k:k+hits,2] = m #mass, equals 0 for background for now
		data_flat[k:k+hits,3] = npy_array[j][2]
		data_flat[k:k+hits,4] = npy_array[j][3]
		data_flat[k:k+hits,5] = npy_array[j][4]
		data_flat[k:k+hits,6] = npy_array[j][5]
		k+=hits


	return data_flat




def plot_scatter(arr1,arr2,depth,title=' '):
	#function to plot original hits in detector vs what is used for training
	#scatter plot of original data

	indexes = np.argsort(arr1[:,-1])

	n_hits = len(arr1[:,1])
	arr1 = arr1[indexes]

	fig = plt.figure()
	ax = fig.add_subplot(111,projection = '3d')

	e = arr1[:,-1]

	e_max = np.amax(e)
	m_size = e/e_max*20
	m_size = m_size.astype(int)


	ax.scatter(arr1[:,2], arr1[:,0], arr1[:,1], s=m_size, c=e,vmin=0,vmax=e_max,cmap='jet')

	#plt.colorbar()

	ax.set_xlabel('Z')
	ax.set_ylabel('X')
	ax.set_zlabel('Y')
	plt.title(title)
	plt.show()


	rows = int(depth/2) +1
	fig, axs = plt.subplots(rows, 2)

	k=0

	for i in range(rows):
		for j in range(2):
			axs[i, j].matshow(arr2[:,:,k])
			axs[i, j].set_title('Channel {}'.format(k))
			k+=1
			if k>=depth:
				break
	plt.show()

	# Hide x labels and tick labels for top plots and y ticks for right plots.
	for ax in axs.flat:
		ax.label_outer()




def im_split_z(data,dim,n_tot,depth,log=False):
	"""
	function accepts list of hits and returns images for training for every EVENT (not hit)
	#if log=true, the z axis will be separated logarithmically (more info at the end of event)
	dim = dimension of image wanted for training (i.e. will produce a dim x dim output array)
	depth = number of channel for image. Z axis will be split into depth parts and each part will be 
	summed and added to a channel
	"""
	n_h = len(data[:,0]) #number of hits

	if log==False: #define lattice to insert energies into to creta images
		x_tab = np.linspace(x_min,x_max,dim)
		y_tab = np.linspace(y_min,y_max,dim)
		z_tab = np.linspace(z_min,z_max,depth)

	else:
		x_tab = np.linspace(x_min,x_max,dim)
		y_tab = np.linspace(y_min,y_max,dim)
		z_tab = np.logspace(z_min,z_max,depth)		


	k=1 #read lines
	h=0 #count events
	n_event = 1 #event counter within 1 file

	im_tab = np.zeros((dim,dim,depth)) #array to contain all constructed images
	merged_array = np.zeros(n_tot, dtype=[('sim','u1'),('num','u1'),('mass', 'f4'),('label','u1'),('image','f4',(dim,dim,depth))])


	#first hit
	x_coord = np.argmin(np.abs(x_tab-data[0,3]))
	y_coord = np.argmin(np.abs(y_tab-data[0,4]))
	z_coord = np.argmin(np.abs(z_tab-data[0,5]))

	im_tab[x_coord,y_coord,z_coord] += data[0,6]

	n_hit = 1 #hit counter within one event

	while h<n_tot: #loop over events

		while k<n_h and data[k,1].astype(int) == n_event: #loop over hits within event

			x_coord = np.argmin(np.abs(x_tab-data[k,3]))
			y_coord = np.argmin(np.abs(y_tab-data[k,4]))
			z_coord = np.argmin(np.abs(z_tab-data[k,5]))

			im_tab[x_coord,y_coord,z_coord] += data[k,6]


			n_hit += 1 #next hit within event
			k+=1 #next overall hit

			if k>= n_h:
				break

		#fill in array with information on event
		merged_array['sim'][h]= data[k-1,0]
		merged_array['num'][h] = data[k-1,1]
		merged_array['mass'][h]= data[k-1,2]
		merged_array['label'][h] = (data[k-1,0]>100) #1 for signal, 0 for back.
		merged_array['image'][h] = im_tab

		#show plot of actual event vs "compressed event"
		"""
		if h<5:
			data_actual = data[k-n_hit:k,3:]
			data_comp = im_tab
			plot_scatter(data_actual,data_comp,depth)
		"""
		n_hit = 0 #restart counter on hits within events
		h+=1 #next event
		if k<n_h:
			n_event = data[k,1].astype(int)

		if h % 10000 == 0:
			print('{}% completed'.format(h/n_tot*100))

	return merged_array


def im_xyz(data,dim,n_tot,depth=3,log=False):
	#function accepting list of hits and returning images for training for every EVENT (not hit)
	n_h = len(data[:,0]) #number of hits

	x_tab = np.linspace(x_min,x_max,dim) #definition of lattice points for creation of images
	y_tab = np.linspace(y_min,y_max,dim)
	z_tab = np.linspace(z_min,z_max,dim)

	k=1 #count overall hits
	h=0 #count events
	n_event = 1 #count over events within one file

	im_tab = np.zeros((dim,dim,3)) #array to contain constructed image
	merged_array = np.zeros(n_tot, dtype=[('sim','int32'),('num','int32'),('mass', 'f4'),('label','int32'),('image','f4',(dim,dim,depth))])

	#first hit
	x_coord = np.argmin(np.abs(x_tab-data[0,3])) #coordinates of pixel where energy for this hit will be added
	y_coord = np.argmin(np.abs(y_tab-data[0,4]))
	z_coord = np.argmin(np.abs(z_tab-data[0,5]))

	im_tab[x_coord,y_coord,0] += data[0,6] #add energy to proper pixels in each channel
	im_tab[y_coord,z_coord,1] += data[0,6]
	im_tab[z_coord,x_coord,2] += data[0,6]
	
	n_hit = 1 #hits within one event

	while h<n_tot: #loop over events

		im_tab = np.zeros((dim,dim,3))

		while k<n_h and data[k,1].astype(int) == n_event: #loop over hits within event

			x_coord = np.argmin(np.abs(x_tab-data[k,3])) #coordinates of pixel where energy for this hit will be added
			y_coord = np.argmin(np.abs(y_tab-data[k,4]))
			z_coord = np.argmin(np.abs(z_tab-data[k,5]))

			im_tab[x_coord,y_coord,0] += data[k,6] #add energy to proper pixels
			im_tab[y_coord,z_coord,1] += data[k,6]
			im_tab[z_coord,x_coord,2] += data[k,6]


			n_hit += 1 #counter of hits within event

			k+=1 #counter of overall hits

			if k>= n_h:
				break

		#fill in array with information on event
		merged_array['sim'][h]= data[k-1,0]
		merged_array['num'][h] = data[k-1,1]
		merged_array['mass'][h]= data[k-1,2]
		#print(data[k-1,0])
		merged_array['label'][h] = (data[k-1,0]>100) #1 for signal, 0 for back.
		merged_array['image'][h] = im_tab

		#show plot of actual event vs "compressed event"
		"""
		if h<3 or h>16:
			data_actual = data[k-n_hit:k,3:]
			data_comp = im_tab
			plot_scatter(data_actual,data_comp,3)
		"""

		n_hit = 0 #restart counter over hits in an event
		h+=1 #next event
		if k<n_h:
			n_event = data[k,1].astype(int)

		if h % 10000 == 0:
			print('{}% completed'.format(h/n_tot*100))

	return merged_array

def merge_arrays(m_list,n_s_event,n_b_event,format_type,dim,depth=3,log=False):
	"""
	Accepts list with ints of masses (MeV) for wich to include data, the number of events to include for each mass (n_s_event)
	and the number of background event to include.

	Available masses: 500 000 events for 1, 10, 100, 1000, 150 000 events for 5, 50, 500
	Around 1 000 000 events available for background (more available, but not converted from root tree to python-usable array yet)

	format_type either 'z_split' or 'xyz' (z_split splits z axis into depth image channels and xyz fills RGB channels with
	x, y and z projections of events)
	dim size of wanted image for training, depth number of channel (3 by default, can only be changed if using z_split)
	log = False by default, use log=True if using z_split and want to split the z axis logarithmically to fill channels
	Merges all arrays containing wanted signal and background to train a network together
	into a structured array with dtype:
	dtype=[('num','u1'),('mass', 'u1'),('label','u1'),('image','f4'),(dim,dim,depth))]
	num being the event number, mass the mass of the A' (for background events, will randomly select
	A' mass witin same distribution as signal), label 1 for signal and 0 for background and image contains 
	the dim x dim x depth image used as input for training

	"""
	if format_type == 'z_split':
		fun = im_split_z
	if format_type == 'xyz':
		fun = im_xyz

	i=0
	#flatten all necessary signal data arrays and concatenate them
	if n_s_event != 0:
		for m in m_list: 
			ind = np.where((m_tab-m)==0)[0][0]

			if ind==None:
				print('Mass not available!')
			else:
				if i==0:
					data_s = flat_npy(sig_tab[ind],m*1e3,m)
					index = np.where(data_s[:,1] == float(n_s_event))[0][-1]
					data_s = data_s[:index,:]
				else:
					new_data_s = flat_npy(sig_tab[ind],m*1e3,m) # array to contain all hits flattened out
					index = np.where(new_data_s[:,1] == float(n_s_event))[0][-1]
					new_data_s = new_data_s[:index,:]

					data_s = np.concatenate((data_s,new_data_s))
			i+=1

	#flatten all necessary background data arrays to get to desired number of events and conc. them
	i = 1
	if n_b_event != 0:
		data_b = flat_npy(back_tab[0],0,0)
		n_b = data_b[-1,1] #number of events in first dataset
		num = n_b_event - n_b #number of events still needed to get to desired amount

		if num<0:
			index = np.where(data_b[:,1] == float(n_b_event))[0][-1]
			data_b = data_b[:index,:]

		while num>0:
			new_data_b = flat_npy(back_tab[i],i,0)

			n_b = new_data_b[-1,1]

			if (num-n_b)<0:
				index = np.where(new_data_b[:,1] == float(num))[0][-1]
				new_data_b = new_data_b[:index,:]

			num -= n_b

			data_b = np.concatenate((data_b,new_data_b))

			i+=1

		#add mass information to background events
		n_m = len(m_list) #number of masses in signal data
		m_ind = np.random.randint(0,n_m,size=len(data_b)) 
		m_list = np.array(m_list)
		data_b[:,2] = m_list[m_ind]

	#concatenate signal and background
	if n_b_event != 0 :
		data = np.concatenate((data_s,data_b))
	else:
		data = data_s
	n_tot = len(m_list)*n_s_event + n_b_event

	#create numpy array for all wanted events and fill it with info
	merged_array_1= fun(data,dim,n_tot,depth,log)

	merged_array_final = correct(merged_array_1)


	return merged_array_final

def correct(array_data):
	"""
	Patch to correct problem in event generation, some events are randomly empty.
	Until problem found
	"""
	wrong = []
	for i in range(len(array_data)):
		if np.sum(array_data['image'][i,:,:,:]) == float(1200):
			wrong.append(i)
	array_data_good = np.delete(array_data,np.array(wrong))
	n = len(array_data_good)
	rest = int(n%100)
	if rest != 0:
		array_data_good = array_data_good[:-rest]
	return array_data_good



file_placement = 'dataset/'

#Already existant datasets
"""
formatted_array = merge_arrays([100],200000,200000,'xyz',20)

name_save = 'data_formatted_xyz_m_100.npy'
np.save(file_placement+name_save,formatted_array)

formatted_array = merge_arrays([1],200000,200000,'xyz',20)

name_save = 'data_formatted_xyz_m_1.npy'
np.save(file_placement+name_save,formatted_array)

formatted_array = merge_arrays([10],200000,200000,'xyz',20)

name_save = 'data_formatted_xyz_m_10.npy'
np.save(file_placement+name_save,formatted_array)

formatted_array = merge_arrays([1000],200000,200000,'xyz',20)

name_save = 'data_formatted_xyz_m_1000.npy'
np.save(file_placement+name_save,formatted_array)


formatted_array = merge_arrays([1,10,100,1000],150000,400000,'xyz',20)

name_save = 'data_formatted_xyz_allm.npy'
np.save(file_placement+name_save,formatted_array)

formatted_array = merge_arrays([100],200000,200000,'z_split',20,depth=3,log=False)

name_save = 'data_formatted_z_m_100_d3.npy'
np.save(file_placement+name_save,formatted_array)

formatted_array = merge_arrays([100],200000,200000,'z_split',20,depth=3,log=True)

name_save = 'data_formatted_z_m_100_d3_log.npy'
np.save(file_placement+name_save,formatted_array)

formatted_array = merge_arrays([100],200000,200000,'z_split',20,depth=5,log=False)

name_save = 'data_formatted_z_m_100_d5.npy'
np.save(file_placement+name_save,formatted_array)

formatted_array = merge_arrays([100],200000,200000,'z_split',20,depth=5,log=True)

name_save = 'data_formatted_z_m_100_d5_log.npy'
np.save(file_placement+name_save,formatted_array)


formatted_array = merge_arrays([1,10,100,1000],200000,400000,'xyz',20)

name_save = 'data_formatted_xyz_allm_large.npy'
np.save(file_placement+name_save,formatted_array)

formatted_array = merge_arrays([1],100001,0,'xyz',20)

name_save = 'sample_data_formatted_xyz_m_1_noback.npy'
np.save(file_placement+name_save,formatted_array)

formatted_array = merge_arrays([5],100001,0,'xyz',20)

name_save = 'sample_data_formatted_xyz_m_5_noback.npy'
np.save(file_placement+name_save,formatted_array)

formatted_array = merge_arrays([10],100001,0,'xyz',20)

name_save = 'sample_data_formatted_xyz_m_10_noback.npy'
np.save(file_placement+name_save,formatted_array)

formatted_array = merge_arrays([50],100001,0,'xyz',20)

name_save = 'sample_data_formatted_xyz_m_50_noback.npy'
np.save(file_placement+name_save,formatted_array)

formatted_array = merge_arrays([100],100001,0,'xyz',20)

name_save = 'sample_data_formatted_xyz_m_100_noback.npy'
np.save(file_placement+name_save,formatted_array)

formatted_array = merge_arrays([500],100001,0,'xyz',20)

name_save = 'sample_data_formatted_xyz_m_500_noback.npy'
np.save(file_placement+name_save,formatted_array)

formatted_array = merge_arrays([1000],100001,0,'xyz',20)

name_save = 'sample_data_formatted_xyz_m_1000_noback.npy'
np.save(file_placement+name_save,formatted_array)
"""