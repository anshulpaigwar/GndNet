import glob
import os
import sys
sys.path.append("../..") # Adds higher directory to python modules path.
import ipdb as pdb
import time
import numpy as np
from numpy.linalg import inv
import rospy
from visualization_msgs.msg import MarkerArray, Marker
from message_filters import TimeSynchronizer, Subscriber,ApproximateTimeSynchronizer
import message_filters
from sensor_msgs.msg import PointCloud2
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_from_matrix
import tf

from geometry_msgs.msg import Point

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import ros_numpy
import numba
from numba import jit,types
from functools import reduce
from scipy.spatial import Delaunay
from skimage.morphology import reconstruction
import torch
from torch.utils.data.sampler import SubsetRandomSampler
# from utils.utils import np2ros_pub

plt.ion()
fig = plt.figure()




# # resolution of ground estimator grid is 1m x 1m 
# visualize = False
# pc_range = [0.6, -30, 60.6, 30]
# grid_size = [0, -30, 60, 30]
grid_size = [-50, -50, 50, 50]
length = int(grid_size[2] - grid_size[0]) # x direction
width = int(grid_size[3] - grid_size[1])    # y direction
# lidar_height = 1.732
# out_dir = '/home/anshul/es3cap/dataset/kitti_gnd_dataset/training/seq_039'





def np2ros_pub(points, pcl_pub, timestamp = None):
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
    points_arr['intensity'] = points[3]
    # points_arr['g'] = 255
    # points_arr['b'] = 255

    if timestamp == None:
        timestamp = rospy.Time.now()
    cloud_msg = ros_numpy.msgify(PointCloud2, points_arr,stamp =timestamp, frame_id = "/kitti/velo_link")
    # rospy.loginfo("happily publishing sample pointcloud.. !")
    pcl_pub.publish(cloud_msg)




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

















def parse_calibration(filename):
  """ read calibration file with given filename
      Returns
      -------
      dict
          Calibration matrices as 4x4 numpy arrays.
  """
  calib = {}

  calib_file = open(filename)
  for line in calib_file:
    key, content = line.strip().split(":")
    values = [float(v) for v in content.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    calib[key] = pose

  calib_file.close()

  return calib




def parse_poses(filename, calibration):
  """ read poses file with per-scan poses from given filename
      Returns
      -------
      list
          list of poses as 4x4 numpy arrays.
  """
  file = open(filename)

  poses = []

  Tr = calibration["Tr"]
  Tr_inv = inv(Tr)

  for line in file:
    values = [float(v) for v in line.strip().split()]

    pose = np.zeros((4, 4))
    pose[0, 0:4] = values[0:4]
    pose[1, 0:4] = values[4:8]
    pose[2, 0:4] = values[8:12]
    pose[3, 3] = 1.0

    poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

  return poses


def broadcast_TF(pose,timestamp):
    quat = quaternion_from_matrix(pose)
    br = tf.TransformBroadcaster()
    br.sendTransform((pose[0,3], pose[1,3], pose[2,3]), quat,
                     timestamp,
                     "/kitti/zoe_odom_origin",
                     "/kitti/world")

    # br.sendTransform((1.5, 0, 1.732), (0,0,0,1),
    br.sendTransform((1.5, 0, 2.1), (0,0,0,1),
                     timestamp,
                     "/kitti/velo_link",
                     "/kitti/zoe_odom_origin")

    br.sendTransform((3.334, 0, 0.34), (0,0,0,1),
                     timestamp,
                     "/kitti/base_link",
                     "/kitti/zoe_odom_origin")



# @jit(nopython=True)
def segment_cloud(cloud, gnd_labels):
    index = np.isin(cloud[:,4], gnd_labels)
    gnd = cloud[index]
    obs = cloud[np.invert(index)]
    return gnd, obs


@jit(nopython=True)
def lidar_to_img(points, grid_size, voxel_size, fill):
    # pdb.set_trace()
    lidar_data = points[:, :2] # neglecting the z co-ordinate
    height_data = points[:, 2] # neglecting the z co-ordinate
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
        if(height_data[i] < 0):
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
        if(z < 2):
            if (0 < x < heightmap.shape[0]) and (0 < y < heightmap.shape[1]):
                k = num_points[x,y] # current num of points in a cell
                if k-1 <= max_points:
                    heightmap[x,y,k-1] = z
                    num_points[x,y] += 1
    return heightmap.sum(axis = 2)/num_points


# @jit(nopython=True)
def normalise_heightmap(heightmap, h_range):
    heightmap = heightmap.clip(min=h_range[0], max = h_range[1])
    heightmap -= h_range[0]
    heightmap = heightmap/(h_range[1] - h_range[0])
    heightmap *= 255
    heightmap = np.fabs(heightmap)
    heightmap = heightmap.astype(np.uint8)
    return heightmap


def process_cloud(cloud):
    # start = time.time()
    # remove all non ground points; gnd labels = [40, 44, 48, 49]
    # gnd, obs = segment_cloud(cloud,[40, 44, 48, 49])
    gnd, obs = segment_cloud(cloud,[40, 44, 48, 49,60,72])
    visualize = True

    if visualize:
        voxel_size = 1
        grid_size = np.array([-50, -50, 50, 50])
        # gnd_img = lidar_to_img(gnd, grid_size, voxel_size, fill = 2)
        # obs_img = lidar_to_img(obs, grid_size, voxel_size, fill = 1)
        # fig.add_subplot(1, 4, 1)
        # plt.imshow(gnd_img, interpolation='nearest')
        # kernel = np.ones((3,3),np.uint8)
        # gnd_img = cv2.dilate(gnd_img,kernel,iterations = 1)
        # fig.add_subplot(1, 4, 2)
        # plt.imshow(gnd_img, interpolation='nearest')
        # # obs_img = cv2.dilate(obs_img,kernel,iterations = 1)
        # final_img = gnd_img - obs_img
        # final_img = final_img.clip(min=0)
        # fig.add_subplot(1, 4, 3)
        # plt.imshow(obs_img, interpolation='nearest')
        # # dim = (grid_size[3]-grid_size[1], grid_size[2]-grid_size[0])
        # # resized_final_img = cv2.resize(final_img, dim, interpolation = cv2.INTER_NEAREST) 
        # fig.add_subplot(1, 4, 4)
        # plt.imshow(final_img, interpolation='nearest')
        # # plt.imshow(dilation, interpolation='nearest')
        # plt.show()




        gnd_img = lidar_to_img(np.copy(gnd), grid_size, voxel_size, fill = 2)
        gnd_heightmap = lidar_to_heightmap(np.copy(gnd), grid_size, voxel_size, max_points = 100)

        fig.add_subplot(1, 4, 1)
        plt.imshow(gnd_img, interpolation='nearest')
        fig.add_subplot(1, 4, 2)
        plt.imshow(gnd_heightmap, interpolation='nearest')
        kernel = np.ones((3,3),np.uint8)
        gnd_img_dil = cv2.dilate(gnd_img,kernel,iterations = 2)
        mask = gnd_img_dil - gnd_img
        mask = mask.clip(min=0)
        fig.add_subplot(1, 4, 3)
        plt.imshow(mask, interpolation='nearest')
        gnd_heightmap = normalise_heightmap(gnd_heightmap, h_range = [-4,2]) # heights are wrt velodyne not baselink
        final_img = cv2.inpaint(gnd_heightmap,mask.astype(np.uint8),3,cv2.INPAINT_TELEA)
        fig.add_subplot(1, 4, 4)
        plt.imshow(final_img, interpolation='nearest')
        # plt.imshow(dilation, interpolation='nearest')
        plt.show()



    return gnd, final_img



def kitti_semantic_data_generate(data_dir):
    rospy.init_node('gnd_data_provider', anonymous=True)
    pcl_pub = rospy.Publisher("/kitti/velo/pointcloud", PointCloud2, queue_size=10)
    pcl_pub2 = rospy.Publisher("/kitti/raw/pointcloud", PointCloud2, queue_size=10)
    marker_pub = rospy.Publisher("/kitti/ground_marker", Marker, queue_size=10)
    velodyne_dir = data_dir + "velodyne/"
    label_dir = data_dir + 'labels/'
    frames = os.listdir(velodyne_dir)
    calibration = parse_calibration(os.path.join(data_dir, "calib.txt"))
    poses = parse_poses(os.path.join(data_dir, "poses.txt"), calibration)
    rate = rospy.Rate(2) # 10hz
    for f in range(len(frames)):
        points_path = os.path.join(velodyne_dir, "%06d.bin" % f)
        points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)

        label_path = os.path.join(label_dir, "%06d.label" % f)
        label = np.fromfile(label_path, dtype=np.uint32)
        label = label.reshape((-1))
        if label.shape[0] == points.shape[0]:
            sem_label = label & 0xFFFF  # semantic label in lower half
        label = np.expand_dims(label, axis = 1)
        points = np.concatenate((points,label), axis = 1)
        cloud, gnd_label = process_cloud(points)
        
        timestamp = rospy.Time.now()
        broadcast_TF(poses[f],timestamp)
        np2ros_pub(cloud,pcl_pub,timestamp)
        np2ros_pub(points,pcl_pub2,timestamp)
        gnd_marker_pub(gnd_label,marker_pub)
        # print(points.shape)
        pdb.set_trace()
        # rate.sleep()




if __name__ == '__main__':
    data_dir = "/home/anshul/es3cap/dataset/kitti_semantic/dataset/sequences/00/"

    kitti_semantic_data_generate(data_dir)




# # @jit(nopython=True)
# def in_hull(p, hull):
#     if not isinstance(hull,Delaunay):
#         hull = Delaunay(hull)
#     return hull.find_simplex(p)>=0

# def extract_pc_in_box3d(pc, box3d):
#     ''' pc: (N,3), box3d: (8,3) '''
#     box3d_roi_inds = in_hull(pc[:,0:3], box3d)
#     return pc[box3d_roi_inds,:], box3d_roi_inds


# # @jit(nopython=True)
# def extract_pc_in_box2d(pc, box2d):
#     ''' pc: (N,2), box2d: (xmin,ymin,xmax,ymax) '''
#     box2d_corners = np.zeros((4,2))
#     box2d_corners[0,:] = [box2d[0],box2d[1]] 
#     box2d_corners[1,:] = [box2d[2],box2d[1]] 
#     box2d_corners[2,:] = [box2d[2],box2d[3]] 
#     box2d_corners[3,:] = [box2d[0],box2d[3]] 
#     box2d_roi_inds = in_hull(pc[:,0:2], box2d_corners)
#     return pc[box2d_roi_inds,:]

