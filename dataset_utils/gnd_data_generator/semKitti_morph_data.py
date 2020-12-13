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
from scipy.spatial.transform import Rotation as R
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import ros_numpy
import numba
from numba import jit,types
from functools import reduce
from scipy.spatial import Delaunay
from scipy import signal
from skimage.morphology import reconstruction
from skimage.restoration import inpaint
from skimage.morphology import erosion, dilation
import torch
from torch.utils.data.sampler import SubsetRandomSampler
# from utils.utils import np2ros_pub

plt.ion()





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




# def np2ros_pub_2(points, pcl_pub, timestamp, color):
#     npoints = points.shape[0] # Num of points in PointCloud
#     points_arr = np.zeros((npoints,), dtype=[
#                                         ('x', np.float32),
#                                         ('y', np.float32),
#                                         ('z', np.float32),
#                                         ('r', np.uint8),
#                                         ('g', np.uint8),
#                                         ('b', np.uint8)])
#     points = np.transpose(points)
#     points_arr['x'] = points[0]
#     points_arr['y'] = points[1]
#     points_arr['z'] = points[2]
#     points_arr['r'] = color[0] * 255
#     points_arr['g'] = color[1] * 255
#     points_arr['b'] = color[2] * 255

#     if timestamp == None:
#         timestamp = rospy.Time.now()
#     cloud_msg = ros_numpy.msgify(PointCloud2, points_arr,stamp =timestamp, frame_id = "/kitti/base_link")
#     # rospy.loginfo("happily publishing sample pointcloud.. !")
#     pcl_pub.publish(cloud_msg)
#     # rospy.sleep(0.1)

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
    points_arr['intensity'] = color[0]
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
    # br.sendTransform((1.5, 0, 2.1), (0,0,0,1),
    br.sendTransform((-0.27, 0, 1.73), (0,0,0,1),
                     timestamp,
                     "/kitti/velo_link",
                     "/kitti/zoe_odom_origin")

    # br.sendTransform((3.334, 0, 0.34), (0,0,0,1),
    br.sendTransform((2.48, 0, 0), (0,0,0,1),
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
    height_data = points[:, 2] + 1.732
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





def rotate_cloud(cloud, theta):
    # xyz = cloud[:,:3]
    # xyz = np.concatenate((xyz,np.array([1]), axis = 1))
    r = R.from_euler('zyx', theta, degrees=True)
    # r = r.as_matrix()
    r = r.as_dcm()
    cloud[:,:3] = np.dot(cloud[:,:3], r.T)
    return cloud




@jit(nopython=True)
def semantically_segment_cloud(points, grid_size, voxel_size, elevation_map, threshold = 0.2):
    lidar_data = points[:, :2] # neglecting the z co-ordinate
    height_data = points[:, 2] + 1.732
    rgb = np.zeros((points.shape[0],3))
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
                rgb[i,0] = 1
                # rgb[i,1] = 1
            else:
                rgb[i,0] = 0
        else:
            rgb[i,0] = -1
    return rgb




fig = plt.figure()
def process_cloud(cloud):
    # start = time.time()
    # remove all non ground points; gnd labels = [40, 44, 48, 49]
    # gnd, obs = segment_cloud(cloud,[40, 44, 48, 49])
    gnd, obs = segment_cloud(cloud,[40, 44, 48, 49,60,72])
    visualize = True

    if visualize:
        fig.clear()
        voxel_size = 1
        grid_size = np.array([-50, -50, 50, 50])
        gnd_img = lidar_to_img(np.copy(gnd), grid_size, voxel_size, fill = 1)
        gnd_heightmap = lidar_to_heightmap(np.copy(gnd), grid_size, voxel_size, max_points = 100)

        fig.add_subplot(2, 3, 1)
        plt.imshow(gnd_img, interpolation='nearest')

        kernel = np.ones((5,5),np.uint8)
        gnd_img_dil = cv2.dilate(gnd_img,kernel,iterations = 2)

        fig.add_subplot(2, 3, 2)
        plt.imshow(gnd_img_dil, interpolation='nearest')

        mask = gnd_img_dil - gnd_img

        fig.add_subplot(2, 3, 3)
        plt.imshow(mask, interpolation='nearest')
        

        fig.add_subplot(2, 3, 4)
        cs = plt.imshow(gnd_heightmap, interpolation='nearest')
        cbar = fig.colorbar(cs)

        image_result = inpaint.inpaint_biharmonic(gnd_heightmap, mask)
        
        fig.add_subplot(2, 3, 5)
        cs = plt.imshow(image_result, interpolation='nearest')
        cbar = fig.colorbar(cs)

        image_result = signal.convolve2d(image_result, kernel, boundary='wrap', mode='same')/kernel.sum()
        # image_result = inpaint.inpaint_biharmonic(image_result, mask)
        # image_result = cv2.dilate(image_result,kernel,iterations = 1)

        # kernel = np.array([[0,1,0],
        #                    [1,0,1],
        #                    [0,1,0]])
        # kernel = np.ones((7,7),np.uint8)
        # kernel[3,3] = 0
        # ind = mask == 1

        # for i in range(10):
        #     conv_out = signal.convolve2d(gnd_heightmap, kernel, boundary='wrap', mode='same')/kernel.sum()
        #     gnd_heightmap[ind] = conv_out[ind]

        fig.add_subplot(2, 3, 6)
        cs = plt.imshow(image_result, interpolation='nearest')
        cbar = fig.colorbar(cs)
        plt.show()
        # cbar.remove()

        seg = semantically_segment_cloud(cloud.copy(), grid_size, voxel_size, image_result)
        return gnd, image_result.T, seg

    return gnd


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
    angle = 0
    increase = True
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

        if angle > 5.0:
            increase = False
        elif angle < -5.0:
            increase = True


        if increase:
            angle +=0.1
        else:
            angle -=0.1


        points  = rotate_cloud(points, theta = [0,5,angle]) #zyx

        # cloud, gnd_label, seg = process_cloud(points)
        cloud = process_cloud(points)

        # points += np.array([0,0,1.732,0,0], dtype=np.float32)
        # points[0,2] += 1.732
        
        timestamp = rospy.Time.now()
        broadcast_TF(poses[f],timestamp)
        np2ros_pub(cloud, pcl_pub, timestamp)
        np2ros_pub_2(points, pcl_pub2,timestamp, seg.T)
        
        # gnd_marker_pub(gnd_label,marker_pub)
        # print(points.shape)
        pdb.set_trace()
        # rate.sleep()




if __name__ == '__main__':
    data_dir = "/home/anshul/es3cap/dataset/kitti_semantic/dataset/sequences/06/"

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

