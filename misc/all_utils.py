
"""
Author: Anshul Paigwar
email: p.anshul6@gmail.com
"""



from __future__ import print_function
from __future__ import division

import argparse
import os
import sys
sys.path.append("..") # Adds higher directory to python modules path.

import shutil
import time
import yaml
import torch
from torchvision import datasets, transforms
# from utils.point_cloud_ops_test import points_to_voxel

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Ros Includes
import rospy
# from tf.transformations import euler_from_quaternion, quaternion_from_euler
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
from visualization_msgs.msg import Marker,MarkerArray
from geometry_msgs.msg import Point

import ros_numpy
import shapely.geometry
from scipy.spatial import Delaunay

import numba
from numba import jit,types










def gnd_marker_pub(gnd_label, marker_pub, cfg, color = "red"):
    length = int(cfg.grid_range[2] - cfg.grid_range[0]) # x direction
    width = int(cfg.grid_range[3] - cfg.grid_range[1])    # y direction
    gnd_marker = Marker()
    gnd_marker.header.frame_id = "/kitti/base_link"
    gnd_marker.header.stamp = rospy.Time.now()
    gnd_marker.type = gnd_marker.LINE_LIST
    gnd_marker.action = gnd_marker.ADD
    gnd_marker.scale.x = 0.05
    gnd_marker.scale.y = 0.05
    gnd_marker.scale.z = 0.05
    if(color == "red"):
        gnd_marker.color.a = 1.0
        gnd_marker.color.r = 1.0
        gnd_marker.color.g = 0.0
        gnd_marker.color.b = 0.0
    if(color == "green"):
        gnd_marker.color.a = 1.0
        gnd_marker.color.r = 0.0
        gnd_marker.color.g = 1.0
        gnd_marker.color.b = 0.0
    gnd_marker.points = []

# gnd_labels are arranged in reverse order
    for j in range(gnd_label.shape[0]):
        for i in range(gnd_label.shape[1]):
            pt1 = Point()
            pt1.x = i + cfg.grid_range[0]
            pt1.y = j + cfg.grid_range[1]
            pt1.z = gnd_label[j,i]

            if j>0 :
                pt2 = Point()
                pt2.x = i + cfg.grid_range[0]
                pt2.y = j-1 +cfg.grid_range[1]
                pt2.z = gnd_label[j-1, i]
                gnd_marker.points.append(pt1)
                gnd_marker.points.append(pt2)

            if i>0 :
                pt2 = Point()
                pt2.x = i -1 + cfg.grid_range[0]
                pt2.y = j + cfg.grid_range[1]
                pt2.z = gnd_label[j, i-1]
                gnd_marker.points.append(pt1)
                gnd_marker.points.append(pt2)

            if j < width-1 :
                pt2 = Point()
                pt2.x = i + cfg.grid_range[0]
                pt2.y = j + 1 + cfg.grid_range[1]
                pt2.z = gnd_label[j+1, i]
                gnd_marker.points.append(pt1)
                gnd_marker.points.append(pt2)

            if i < length-1 :
                pt2 = Point()
                pt2.x = i + 1 + cfg.grid_range[0]
                pt2.y = j + cfg.grid_range[1]
                pt2.z = gnd_label[j, i+1]
                gnd_marker.points.append(pt1)
                gnd_marker.points.append(pt2)

    marker_pub.publish(gnd_marker)












def np2ros_pub(points, pcl_pub, timestamp = None):
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

    if timestamp == None:
        timestamp = rospy.Time.now()
    cloud_msg = ros_numpy.msgify(PointCloud2, points_arr,stamp =timestamp, frame_id = "/kitti/base_link")
    # rospy.loginfo("happily publishing sample pointcloud.. !")
    pcl_pub.publish(cloud_msg)
    # rospy.sleep(0.1)





def rgb_to_float(color):
    """ Converts an RGB list to the packed float format used by PCL
    
        From the PCL docs:
        "Due to historical reasons (PCL was first developed as a ROS package),
         the RGB information is packed into an integer and casted to a float"
    
        Args:
            color (list): 3-element list of integers [0-255,0-255,0-255]
            
        Returns:
            float_rgb: RGB value packed as a float
    """
    hex_r = (0xff & color[0]) << 16
    hex_g = (0xff & color[1]) << 8
    hex_b = (0xff & color[2])

    hex_rgb = hex_r | hex_g | hex_b

    float_rgb = struct.unpack('f', struct.pack('i', hex_rgb))[0]

    return float_rgb









def np2ros_pub_2(points, pcl_pub, timestamp, color):
    npoints = points.shape[0] # Num of points in PointCloud
    points_arr = np.zeros((npoints,), dtype=[
                                        ('x', np.float32),
                                        ('y', np.float32),
                                        ('z', np.float32),
                                        ('intensity', np.float32)])
    points = np.transpose(points)
    points_arr['x'] = points[0]
    points_arr['y'] = points[1]
    points_arr['z'] = points[2]

    # float_rgb = rgb_to_float(color)
    points_arr['intensity'] = color
    # points_arr['g'] = 255
    # points_arr['b'] = 255

    if timestamp == None:
        timestamp = rospy.Time.now()
    cloud_msg = ros_numpy.msgify(PointCloud2, points_arr,stamp =timestamp, frame_id = "/kitti/base_link")
    # rospy.loginfo("happily publishing sample pointcloud.. !")
    pcl_pub.publish(cloud_msg)







def visualize_gnd_3D(gnd_label , fig, cfg):
    length = int(cfg.grid_range[2] - cfg.grid_range[0]) # x direction
    width = int(cfg.grid_range[3] - cfg.grid_range[1])    # y direction
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




# def visualize_2D(gnd_label, points ,fig, cfg):
#     fig.clf()
#     fig.add_subplot(1, 2, 1)
#     plt.imshow(gnd_label, interpolation='nearest')
#     pc_img = points_to_voxel(points, cfg.voxel_size, cfg.pc_range, cfg.max_points_voxel, True, cfg.max_voxels)
#     fig.add_subplot(1, 2, 2)
#     plt.imshow(pc_img[0], interpolation='nearest')
#     plt.show()
#     plt.pause(0.01)






################################ Dataset generation util ###################################



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



# @jit(nopython=True)
def crop_cloud_msg(cloud_msg, cfg, shift_cloud = False, sample_cloud = True):
    # Convert Ros pointcloud2 msg to numpy array
    pc = ros_numpy.numpify(cloud_msg)
    points=np.zeros((pc.shape[0],4))
    points[:,0]=pc['x']
    points[:,1]=pc['y']
    points[:,2]=pc['z']
    points[:,3]=pc['intensity']
    cloud = np.array(points, dtype=np.float32)

    if shift_cloud:
        cloud += np.array([0,0,cfg.lidar_height,0], dtype=np.float32)


    pc_range = [cfg.pc_range[0],cfg.pc_range[1],cfg.pc_range[3],cfg.pc_range[4]]
    cloud = extract_pc_in_box2d(cloud, pc_range)

    if sample_cloud:
        # random sample point cloud to specified number of points
        cloud = random_sample_numpy(cloud, N = 50000)

    return cloud

#################################################################################################

@jit(nopython=True)
def shift_cloud_func(cloud, height):
    cloud += np.array([0,0,height,0], dtype=np.float32)
    return cloud

def cloud_msg_to_numpy(cloud_msg, cfg, shift_cloud = False):
    # Convert Ros pointcloud2 msg to numpy array
    pc = ros_numpy.numpify(cloud_msg)
    points=np.zeros((pc.shape[0],4))
    points[:,0]=pc['x']
    points[:,1]=pc['y']
    points[:,2]=pc['z']
    points[:,3]=pc['intensity']
    cloud = np.array(points, dtype=np.float32)

    if shift_cloud:
        cloud  = shift_cloud_func(cloud, cfg.lidar_height)
    return cloud


@jit(nopython=True)
def segment_cloud(points, grid_size, voxel_size, elevation_map, threshold = 0.2):
    lidar_data = points[:, :2] # neglecting the z co-ordinate
    height_data = points[:, 2] #+ 1.732
    rgb = np.zeros(points.shape[0])
    # pdb.set_trace()
    lidar_data -= np.array([grid_size[0], grid_size[1]])
    lidar_data = lidar_data /voxel_size # multiplying by the resolution
    lidar_data = np.floor(lidar_data)
    lidar_data = lidar_data.astype(np.int32)
    N = lidar_data.shape[0] # Total number of points
    for i in range(N):
        x = lidar_data[i,0]
        y = lidar_data[i,1]
        z = height_data[i]
        if (0 < x < elevation_map.shape[0]) and (0 < y < elevation_map.shape[1]):
            if z > elevation_map[x,y] + threshold:
                rgb[i] = 1 # is obs
            else:
                rgb[i] = 0 # is gnd
        else:
            rgb[i] = -1 # outside range
    return rgb



@jit(nopython=True)
def lidar_to_img(points, grid_size, voxel_size, fill):
    # pdb.set_trace()
    lidar_data = points[:, :2] # neglecting the z co-ordinate
    height_data = points[:, 2] + 1.732
    # pdb.set_trace()
    lidar_data -= np.array([grid_size[0], grid_size[1]])
    lidar_data = lidar_data /voxel_size # multiplying by the resolution
    lidar_data = np.floor(lidar_data)
    lidar_data = lidar_data.astype(np.int32)
    # lidar_data = np.reshape(lidar_data, (-1, 2))
    voxelmap_shape = (grid_size[2:]-grid_size[:2])/voxel_size
    lidar_img = np.zeros((int(voxelmap_shape[0]),int(voxelmap_shape[1])))
    N = lidar_data.shape[0]
    for i in range(N):
        if(height_data[i] < 10):
            if (0 < lidar_data[i,0] < lidar_img.shape[0]) and (0 < lidar_data[i,1] < lidar_img.shape[1]):
                lidar_img[lidar_data[i,0],lidar_data[i,1]] = fill
    return lidar_img




@jit(nopython=True)
def lidar_to_heightmap(points, grid_size, voxel_size, max_points):
    lidar_data = points[:, :2] # neglecting the z co-ordinate
    height_data = points[:, 2]
    # pdb.set_trace()
    lidar_data -= np.array([grid_size[0], grid_size[1]])
    lidar_data = lidar_data /voxel_size # multiplying by the resolution
    lidar_data = np.floor(lidar_data)
    lidar_data = lidar_data.astype(np.int32)
    # lidar_data = np.reshape(lidar_data, (-1, 2))
    heightmap_shape = (grid_size[2:]-grid_size[:2])/voxel_size
    heightmap = np.zeros((int(heightmap_shape[0]),int(heightmap_shape[1]), max_points))
    num_points = np.ones((int(heightmap_shape[0]),int(heightmap_shape[1])), dtype = np.int32) # num of points in each cell # np.ones just to avoid division by zero
    N = lidar_data.shape[0] # Total number of points
    for i in range(N):
        x = lidar_data[i,0]
        y = lidar_data[i,1]
        z = height_data[i]
        if(z < 10):
            if (0 < x < heightmap.shape[0]) and (0 < y < heightmap.shape[1]):
                k = num_points[x,y] # current num of points in a cell
                if k-1 <= max_points:
                    heightmap[x,y,k-1] = z
                    num_points[x,y] += 1
    return heightmap.sum(axis = 2)/num_points
