from __future__ import print_function, division
import os
import sys
sys.path.append("..") # Adds higher directory to python modules path.



import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from core.point_cloud.point_cloud_ops import points_to_voxel


# Ros Includes
import rospy

from sensor_msgs.msg import PointCloud2
import std_msgs.msg
from visualization_msgs.msg import Marker
import ros_numpy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



import ipdb as pdb

plt.ion()
grid_size = [0, -30, -1, 60, 30, 3] #xyz
length = grid_size[3] - grid_size[0]
width = grid_size[4] - grid_size[1]

voxel_size = [1,1,4]
max_points = 100 # per voxel
max_voxels = 3600

rospy.init_node('gnd_data_provider', anonymous=True)
pcl_pub = rospy.Publisher("/reduced_velo", PointCloud2, queue_size=10)



def np2ros_pub(points):
	npoints = points.shape[0] # Num of points in PointCloud
	points_arr = np.zeros((npoints,), dtype=[
										('x', np.float32),
										('y', np.float32),
										('z', np.float32),
										('r', np.uint8),
										('g', np.uint8),
										('b', np.uint8)])
	points = np.transpose(points)
	points_arr['x'] = points[0]
	points_arr['y'] = points[1]
	points_arr['z'] = points[2]
	points_arr['r'] = 255
	points_arr['g'] = 255
	points_arr['b'] = 255

	cloud_msg = ros_numpy.msgify(PointCloud2, points_arr,stamp =rospy.Time.now(), frame_id = "/kitti/base_link")
	rospy.loginfo("happily publishing sample pointcloud.. !")
	pcl_pub.publish(cloud_msg)
	# rospy.sleep(0.1)




def visualize_gnd(gnd_label , fig):
	fig.clf()
	sub_plt = fig.add_subplot(111, projection='3d')
	sub_plt.set_xlabel('$X$', fontsize=20)
	sub_plt.set_ylabel('$Y$')
	X = np.arange(0, length, 1)
	Y = np.arange(0, width, 1)
	X, Y = np.meshgrid(X, Y)  # `plot_surface` expects `x` and `y` data to be 2D
	sub_plt.plot_surface(X, Y, gnd_label.numpy())
	sub_plt.set_zlim(-10, 10)
	plt.draw()
	plt.pause(0.01)






class kitti_gnd(Dataset):
	def __init__(self, data_dir, train = True):
		self.train = train

		if self.train:
			self.train_data = []
			self.train_labels = []
			print('loading training data ')
			seq_folders = os.listdir(data_dir +"training/")
			for seq_num in range(0, len(seq_folders)):
				seq_path = data_dir +"training/"+ "seq_"+ "%03d" % seq_num
				files_in_seq = os.listdir(seq_path + '/reduced_velo/')

				for data_num in range(0, len(files_in_seq)):
					data_dic = {}
					data_path = seq_path + '/reduced_velo/' + "%06d.npy" % data_num
					data = np.load(data_path) #(N,3) point set
					data = points_to_voxel(data, voxel_size, grid_size, max_points, True, max_voxels)
					data_dic['voxels'] = data[0]
					data_dic['coors'] = data[1]
					data_dic['num_points'] = data[2]
					self.train_data.append(data_dic)

					label_path = seq_path + '/gnd_labels/' + "%06d.npy" % data_num
					label = np.load(label_path) # (W x L)
					self.train_labels.append(label)

		else:
			self.valid_data = []
			self.valid_labels = []
			print('loading validation data ')
			seq_folders = os.listdir(data_dir +"validation/")
			for seq_num in range(0, len(seq_folders)):
				seq_path = data_dir +"validation/"+ "seq_"+ "%03d" % seq_num
				files_in_seq = os.listdir(seq_path + '/reduced_velo/')

				for data_num in range(0, len(files_in_seq)):
					data_dic = {}
					data_path = seq_path + '/reduced_velo/' + "%06d.npy" % data_num
					data = np.load(data_path) #(N,3) point set
					data = points_to_voxel(data, voxel_size, grid_size, max_points, True, max_voxels)
					data_dic['voxels'] = data[0]
					data_dic['coors'] = data[1]
					data_dic['num_points'] = data[2]
					self.valid_data.append(data_dic)

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






def get_data_loaders(data_dir, batch = 32):

	use_cuda = torch.cuda.is_available()
	if use_cuda:
		print("using cuda")
		num_workers = 1
		pin_memory = True
	else:
		num_workers = 4
		pin_memory = True


	train_loader = DataLoader(kitti_gnd(data_dir,train = True),
					batch_size= batch, num_workers=num_workers, pin_memory=pin_memory,shuffle=True,drop_last=True)

	valid_loader = DataLoader(kitti_gnd(data_dir,train = False),
					batch_size= batch, num_workers=num_workers, pin_memory=pin_memory,shuffle=True,drop_last=True)

	print("Train Data size ",len(train_loader)*batch)
	print("Valid Data size ",len(valid_loader)*batch)

	return train_loader, valid_loader






if __name__ == '__main__':
	fig = plt.figure()
	data_dir = '/home/anshul/es3cap/dataset/kitti_gnd_dataset/'
	train_loader, valid_loader =  get_data_loaders(data_dir)
	
	for batch_idx, (data, labels) in enumerate(valid_loader):
		pdb.set_trace()
		B = data.shape[0] # Batch size
		N = data.shape[1] # Num of points in PointCloud
		print(N)
		data = data.float()
		labels = labels.float()

		for i in range(B):
			pdb.set_trace()
			np2ros_pub(data[i])
			visualize_gnd(labels[i], fig)















