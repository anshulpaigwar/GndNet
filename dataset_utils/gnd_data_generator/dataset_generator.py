import glob
import os
import sys
# sys.path.append("..") # Adds higher directory to python modules path.

import yaml
import ipdb as pdb
import time
import numpy as np
import rospy
from visualization_msgs.msg import MarkerArray, Marker
from message_filters import TimeSynchronizer, Subscriber,ApproximateTimeSynchronizer
import message_filters
from sensor_msgs.msg import PointCloud2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import ros_numpy
import numba
from numba import jit,types
from functools import reduce
from scipy.spatial import Delaunay

import torch
from torch.utils.data.sampler import SubsetRandomSampler



plt.ion()

# with open('config/config.yaml') as f:
with open('config/config_kittiSem.yaml') as f:
	config_dict = yaml.load(f)

class ConfigClass:
	def __init__(self, **entries):
		self.__dict__.update(entries)

cfg = ConfigClass(**config_dict) # convert python dict to class for ease of use



# resolution of ground estimator grid is 1m x 1m 
visualize = False


pc_range = cfg.pc_range[:2] + cfg.pc_range[3:5] # select minmax xy
grid_size = cfg.grid_range
length = int(grid_size[2] - grid_size[0]) # x direction
width = int(grid_size[3] - grid_size[1])	# y direction
out_dir = '/home/anshul/es3cap/my_codes/GndNet/data/training/seq_006'


# @jit(nopython=True)
def in_hull(p, hull):
	if not isinstance(hull,Delaunay):
		hull = Delaunay(hull)
	return hull.find_simplex(p)>=0

def extract_pc_in_box3d(pc, box3d):
	''' pc: (N,3), box3d: (8,3) '''
	box3d_roi_inds = in_hull(pc[:,0:3], box3d)
	return pc[box3d_roi_inds,:], box3d_roi_inds


# @jit(nopython=True)
def extract_pc_in_box2d(pc, box2d):
	''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
	box2d_corners = np.zeros((4,2))
	box2d_corners[0,:] = [box2d[0],box2d[1]] 
	box2d_corners[1,:] = [box2d[2],box2d[1]] 
	box2d_corners[2,:] = [box2d[2],box2d[3]] 
	box2d_corners[3,:] = [box2d[0],box2d[3]] 
	box2d_roi_inds = in_hull(pc[:,0:2], box2d_corners)
	return pc[box2d_roi_inds,:]



# @jit(nopython=True)
def random_sample_torch(cloud, N):
	if(cloud.size > 0):
		cloud = torch.from_numpy(np.asarray(cloud)).float().cuda()

		points_count = cloud.shape[0]
		# pdb.set_trace()
		# print("indices", len(ind))
		if(points_count > 1):
			prob = torch.randperm(points_count) # sampling without replacement
			if(points_count > N):
				idx = prob[:N]
				sampled_cloud = cloud[idx]
				# print(len(crop))
			else:
				r = int(N/points_count)
				cloud = cloud.repeat(r+1,1)
				sampled_cloud = cloud[:N]

		else:
			sampled_cloud = torch.ones(N,3).cuda()
	else:
		sampled_cloud = torch.ones(N,3).cuda()
	return sampled_cloud.cpu().numpy()





# @jit(nopython=True)
def random_sample_numpy(cloud, N):
	if(cloud.size > 0):
		points_count = cloud.shape[0]
		if(points_count > 1):
			idx = np.random.choice(points_count,N) # sample with replacement
			sampled_cloud = cloud[idx]
		else:
			sampled_cloud = np.ones((N,3))
	else:
		sampled_cloud = np.ones((N,3))
	return sampled_cloud

@jit(nopython=True)
def shift_cloud_func(cloud, height):
	cloud += np.array([0,0,height,0], dtype=np.float32)
	return cloud

# @jit(nopython=True)
def process_cloud(cloud_msg):
	# Convert Ros pointcloud2 msg to numpy array
	pc = ros_numpy.numpify(cloud_msg)
	points=np.zeros((pc.shape[0],4))
	points[:,0]=pc['x']
	points[:,1]=pc['y']
	points[:,2]=pc['z']
	points[:,3]=pc['intensity']
	cloud = np.array(points, dtype=np.float32)
	# cloud = cloud[:, :3]  # exclude luminance
	# pdb.set_trace()

	cloud = extract_pc_in_box2d(cloud, pc_range)

	# random sample point cloud to specified number of points
	cloud = random_sample_numpy(cloud, N = cfg.num_points)

	if cfg.shift_cloud:
		cloud  = shift_cloud_func(cloud, cfg.lidar_height)

	# print(cloud.shape)
	return cloud


def process_label(ground_msg):
	label = np.zeros((width, length))
	for pt in ground_msg.points:
		# print(pt.y, pt.x)
		# print(int(pt.y - grid_size[1]), int(pt.x - grid_size[0]))
		label[ int(pt.y - grid_size[1]), int(pt.x - grid_size[0])] = pt.z
	return label


# @jit(nopython=True)
def recorder(cloud, gnd_label,num):
	velo_path = out_dir + "/reduced_velo/" + "%06d" % num
	label_path = out_dir + "/gnd_labels/" + "%06d" % num
	np.save(velo_path,cloud)
	np.save(label_path,gnd_label)




class listener(object):
	def __init__(self):

		# self.point_cloud_sub = message_filters.Subscriber("/kitti/classified_cloud", PointCloud2)
		self.point_cloud_sub = message_filters.Subscriber("/kitti/raw/pointcloud", PointCloud2)
		self.ground_marker_sub = message_filters.Subscriber('/kitti/ground_marker', Marker)

		ts = ApproximateTimeSynchronizer([self.point_cloud_sub, self.ground_marker_sub],10, 0.1, allow_headerless=True)
		ts.registerCallback(self.callback)
		self.hf = plt.figure()
		self.count = 0

	def callback(self,cloud_msg, ground_msg):
		start_time = time.time()
		gnd_label = process_label(ground_msg)
		# label_time = time.time()

		#@imp: We are subscribing classified cloud which is wrt base_link so no need to shift the point cloud.in z direction
		cloud = process_cloud(cloud_msg)
		# cloud_time = time.time()

		# print("label_process: ", label_time- start_time)
		# print("cloud_process: ", cloud_time- label_time)

		recorder(cloud, gnd_label, self.count)
		end_time = time.time()
		print("total_process: ", end_time- start_time, self.count)
		self.count += 1


		if visualize:

			self.hf.clear()
			cs = plt.imshow(gnd_label.T, interpolation='nearest')
			cbar = self.hf.colorbar(cs)


			self.ha = self.hf.add_subplot(111, projection='3d')
			self.ha.set_xlabel('$X$', fontsize=20)
			self.ha.set_ylabel('$Y$')
			X = np.arange(0, length, 1)
			Y = np.arange(0, width, 1)
			X, Y = np.meshgrid(X, Y)  # `plot_surface` expects `x` and `y` data to be 2D
			# R = np.sqrt(X**2 + Y**2)
			self.ha.plot_surface(Y, X, gnd_label)
			self.ha.set_zlim(-10, 10)
			plt.draw()
			plt.pause(0.01)
			self.hf.clf()
		


	



if __name__ == '__main__':
	rospy.init_node('ground_estimation_dataset', anonymous=True)
	obj = listener()
	rospy.spin()
	# plt.show(block=True)





	# cloud += np.array([0,0,lidar_height], dtype=np.float32) # shift the pointcloud as you want it wrt base_link
	# print(cloud.shape)
	# cloud = np.array([p for p in cloud if p[0] > abs(p[1])])
	# cloud = np.array([p for p in cloud if p[0] > 0])

	# # Use PCL to clip the pointcloud
	# region = (grid_size[0],grid_size[1],grid_size[2],0,grid_size[3],grid_size[4],grid_size[5],0)
	# #(xmin, ymin, zmin, smin, xmax, ymax, zmax, smax)
	# pcl_cloud = pcl.PointCloud()
	# pcl_cloud.from_array(cloud)
	# clipper = pcl_cloud.make_cropbox()
	# clipper.set_MinMax(*region)
	# out_cloud = clipper.filter()

	# # if(out_cloud.size > 15000):
	# leaf_size = 0.05
	# vox = out_cloud.make_voxel_grid_filter()
	# vox.set_leaf_size(leaf_size, leaf_size, leaf_size)
	# out_cloud = vox.filter()

