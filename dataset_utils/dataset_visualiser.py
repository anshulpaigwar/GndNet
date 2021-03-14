import glob
import os
import sys
sys.path.append("..") # Adds higher directory to python modules path.

import ipdb as pdb
import time
import numpy as np
import rospy
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import ros_numpy
from utils.point_cloud_ops_test import points_to_voxel


plt.ion()

pc_range= [0.6, -30, -1, 60.6, 30, 3] # cmcdot grid origin is at base_link not the velodyne so have to crop points from 0.6
voxel_size = [1, 1, 4]
max_points_voxel = 100
max_voxels= 3600

# resolution of ground estimator grid is 1m x 1m 
# grid_size = [0, -30, 60, 30]
grid_size = [-50, -50, 50, 50]
length = int(grid_size[2] - grid_size[0]) # x direction
width = int(grid_size[3] - grid_size[1])	# y direction


rospy.init_node('gnd_dataset_visualiser', anonymous=True)


def gnd_marker_pub(gnd_label, marker_pub):
	gnd_marker = Marker()
	gnd_marker.header.frame_id = "/kitti/base_link"
	gnd_marker.header.stamp = rospy.Time.now()
	gnd_marker.type = gnd_marker.LINE_LIST
	gnd_marker.action = gnd_marker.ADD
	gnd_marker.scale.x = 0.05
	gnd_marker.scale.y = 0.05
	gnd_marker.scale.z = 0.05
	gnd_marker.color.a = 1.0
	gnd_marker.color.r = 0.0
	gnd_marker.color.g = 1.0
	gnd_marker.color.b = 0.0
	gnd_marker.points = []

# gnd_labels are arranged in reverse order
	for j in range(gnd_label.shape[0]):
		for i in range(gnd_label.shape[1]):
			pt1 = Point()
			pt1.x = i + grid_size[0]
			pt1.y = j + grid_size[1]
			pt1.z = gnd_label[j,i]

			if j>0 :
				pt2 = Point()
				pt2.x = i + grid_size[0]
				pt2.y = j-1 +grid_size[1]
				pt2.z = gnd_label[j-1, i]
				gnd_marker.points.append(pt1)
				gnd_marker.points.append(pt2)

			if i>0 :
				pt2 = Point()
				pt2.x = i -1 + grid_size[0]
				pt2.y = j + grid_size[1]
				pt2.z = gnd_label[j, i-1]
				gnd_marker.points.append(pt1)
				gnd_marker.points.append(pt2)

			if j < width-1 :
				pt2 = Point()
				pt2.x = i + grid_size[0]
				pt2.y = j + 1 + grid_size[1]
				pt2.z = gnd_label[j+1, i]
				gnd_marker.points.append(pt1)
				gnd_marker.points.append(pt2)

			if i < length-1 :
				pt2 = Point()
				pt2.x = i + 1 + grid_size[0]
				pt2.y = j + grid_size[1]
				pt2.z = gnd_label[j, i+1]
				gnd_marker.points.append(pt1)
				gnd_marker.points.append(pt2)

	marker_pub.publish(gnd_marker)












def np2ros_pub(points, pcl_pub):
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
	# rospy.loginfo("happily publishing sample pointcloud.. !")
	pcl_pub.publish(cloud_msg)
	# rospy.sleep(0.1)




def visualize_gnd_3D(gnd_label , fig):
	fig.clf()
	sub_plt = fig.add_subplot(111, projection='3d')
	sub_plt.set_xlabel('$X$', fontsize=20)
	sub_plt.set_ylabel('$Y$')
	X = np.arange(0, length, 1)
	Y = np.arange(0, width, 1)
	X, Y = np.meshgrid(X,Y)  # `plot_surface` expects `x` and `y` data to be 2D # gnd_labels are arrranged in reverse order YX
	sub_plt.plot_surface(Y, X, gnd_label)
	sub_plt.set_zlim(-10, 10)
	plt.draw()
	plt.pause(0.01)




def visualize_2D(gnd_label, points ,fig):
	fig.clf()
	fig.add_subplot(1, 2, 1)
	plt.imshow(gnd_label, interpolation='nearest')
	pc_img = points_to_voxel(points, voxel_size, pc_range, max_points_voxel, True, max_voxels)
	fig.add_subplot(1, 2, 2)
	plt.imshow(pc_img[0], interpolation='nearest')
	plt.show()
	plt.pause(0.01)







def visualize_data(data_dir):
	pcl_pub = rospy.Publisher("/kitti/reduced_velo", PointCloud2, queue_size=10)
	marker_pub = rospy.Publisher("/kitti/ground_marker", Marker, queue_size=10)
	# markerArray = MarkerArray()
	fig = plt.figure()
	files = os.listdir(data_dir + "/reduced_velo/")
	print(len(files))
	for num in range(len(files)):
		cloud_path = data_dir + "/reduced_velo/"+ "%06d.npy" % num
		points = np.load(cloud_path)
		labels_path = data_dir + "/gnd_labels/" + "%06d.npy" % num
		gnd_label = np.load(labels_path)

		np2ros_pub(points,pcl_pub)
		gnd_marker_pub(gnd_label,marker_pub)

		# visualize_gnd_3D(gnd_label, fig)


		# visualize_2D(gnd_label,points,fig)


		pdb.set_trace()




if __name__ == '__main__':
	data_dir = '/home/anshul/es3cap/my_codes/GndNet/data/training/seq_006'
	visualize_data(data_dir)

