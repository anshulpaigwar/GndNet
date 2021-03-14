from __future__ import print_function, division
import os
import sys
sys.path.append("..") # Adds higher directory to python modules path.

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import yaml

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipdb as pdb








class kitti_gnd(Dataset):
	def __init__(self, data_dir, train = True, skip_frames = 1):
		self.train = train

		if self.train:
			self.train_data = []
			self.train_labels = []
			print('loading training data ')
			seq_folders = os.listdir(data_dir +"training/")
			for seq_num in seq_folders:
				seq_path = data_dir +"training/"+ "seq_"+ "%03d" % int(seq_num.split("_")[1])
				files_in_seq = os.listdir(seq_path + '/reduced_velo/')

				for data_num in range(0, len(files_in_seq),skip_frames): # too much of dataset we skipping files
					data_path = seq_path + '/reduced_velo/' + "%06d.npy" % data_num
					point_set = np.load(data_path) #(N,3) point set
					self.train_data.append(point_set)

					label_path = seq_path + '/gnd_labels/' + "%06d.npy" % data_num
					label = np.load(label_path) # (W x L)
					self.train_labels.append(label)

		else:
			self.valid_data = []
			self.valid_labels = []
			print('loading validation data ')
			seq_folders = os.listdir(data_dir +"validation/")
			for seq_num in seq_folders:
				seq_path = data_dir +"validation/"+ "seq_"+ "%03d" % int(seq_num.split("_")[1])
				files_in_seq = os.listdir(seq_path + '/reduced_velo/')

				for data_num in range(0, len(files_in_seq),skip_frames): # too much of dataset we skipping files
					data_path = seq_path + '/reduced_velo/' + "%06d.npy" % data_num
					point_set = np.load(data_path) #(N,3) point set
					self.valid_data.append(point_set)

					label_path = seq_path + '/gnd_labels/' + "%06d.npy" % data_num
					label = np.load(label_path) # (W x L)
					self.valid_labels.append(label)


	def __getitem__(self, index):
		if self.train:
			return self.train_data[index], self.train_labels[index]
		else:
			return self.valid_data[index], self.valid_labels[index]


	def __len__(self):
		if self.train:
			return len(self.train_data)
		else:
			return len(self.valid_data)


# class kitti_gnd(Dataset):
# 	def __init__(self, data_dir, train = True):
# 		self.train = train

# 		if self.train:
# 			self.train_data = []
# 			self.train_labels = []
# 			print('loading training data ')
# 			seq_folders = os.listdir(data_dir +"training/")
# 			for seq_num in seq_folders:
# 				seq_path = data_dir +"training/"+ "seq_"+ "%03d" % int(seq_num.split("_")[1])
# 				files_in_seq = os.listdir(seq_path + '/reduced_velo/')

# 				for data_num in range(0, len(files_in_seq)):
# 					data_path = seq_path + '/reduced_velo/' + "%06d.npy" % data_num
# 					self.train_data.append(data_path)

# 					label_path = seq_path + '/gnd_labels/' + "%06d.npy" % data_num
# 					self.train_labels.append(label_path)

# 		else:
# 			self.valid_data = []
# 			self.valid_labels = []
# 			print('loading validation data ')
# 			seq_folders = os.listdir(data_dir +"validation/")
# 			for seq_num in seq_folders:
# 				seq_path = data_dir +"validation/"+ "seq_"+ "%03d" % int(seq_num.split("_")[1])
# 				files_in_seq = os.listdir(seq_path + '/reduced_velo/')

# 				for data_num in range(0, len(files_in_seq)):
# 					data_path = seq_path + '/reduced_velo/' + "%06d.npy" % data_num
# 					self.valid_data.append(data_path)

# 					label_path = seq_path + '/gnd_labels/' + "%06d.npy" % data_num
# 					self.valid_labels.append(label_path)


# 	def __getitem__(self, index):
# 		if self.train:
# 			data = np.load(self.train_data[index])#(N,4) point set
# 			label = np.load(self.train_labels[index])# (W x L)
# 			return data, label
# 		else:
# 			data = np.load(self.valid_data[index])#(N,4) point set
# 			label = np.load(self.valid_labels[index])# (W x L)
# 			return data, label


# 	def __len__(self):
# 		if self.train:
# 			return len(self.train_data)
# 		else:
# 			return len(self.valid_data)




def get_valid_loader(data_dir, batch = 4, skip = 1):

	use_cuda = torch.cuda.is_available()
	if use_cuda:
		print("using cuda")
		num_workers = 1
		pin_memory = True
	else:
		num_workers = 4
		pin_memory = True


	valid_loader = DataLoader(kitti_gnd(data_dir,train = False, skip_frames = skip),
					batch_size= batch, num_workers=num_workers, pin_memory=pin_memory,shuffle=True,drop_last=True)

	print("Valid Data size ",len(valid_loader)*batch)

	return valid_loader





def get_train_loader(data_dir, batch = 4, skip = 1):

	use_cuda = torch.cuda.is_available()
	if use_cuda:
		print("using cuda")
		num_workers = 1
		pin_memory = True
	else:
		num_workers = 4
		pin_memory = True

	train_loader = DataLoader(kitti_gnd(data_dir,train = True, skip_frames = skip),
					batch_size= batch, num_workers=num_workers, pin_memory=pin_memory,shuffle=True,drop_last=True)

	print("Train Data size ",len(train_loader)*batch)

	return train_loader




if __name__ == '__main__':

	with open('config/config_kittiSem.yaml') as f:
		config_dict = yaml.load(f, Loader=yaml.FullLoader)

	class ConfigClass:
		def __init__(self, **entries):
			self.__dict__.update(entries)

	cfg = ConfigClass(**config_dict) # convert python dict to class for ease of use
	

	# Ros Includes
	import rospy
	from sensor_msgs.msg import PointCloud2
	import std_msgs.msg
	from visualization_msgs.msg import Marker
	import ros_numpy
	from utils.ros_utils import np2ros_pub, gnd_marker_pub
	
	rospy.init_node('gnd_data_provider', anonymous=True)
	pcl_pub = rospy.Publisher("/kitti/reduced_velo", PointCloud2, queue_size=10)
	marker_pub = rospy.Publisher("/kitti/gnd_marker", Marker, queue_size=10)
	fig = plt.figure()
	data_dir = '/home/anshul/es3cap/my_codes/GndNet/data/'
	train_loader, valid_loader =  get_data_loaders(data_dir)
	
	for batch_idx, (data, labels) in enumerate(valid_loader):
		B = data.shape[0] # Batch size
		N = data.shape[1] # Num of points in PointCloud
		print(N)
		data = data.float()
		labels = labels.float()

		for i in range(B):
			pdb.set_trace()
			np2ros_pub(data[i].numpy(),pcl_pub)
			gnd_marker_pub(labels[i].numpy(),marker_pub, cfg, color = "red")
			# # visualize_gnd_3D(gnd_label, fig)
			# visualize_2D(labels[i],data[i],fig)

