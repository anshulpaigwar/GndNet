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






def rotate_cloud(cloud, theta):
    # xyz = cloud[:,:3]
    # xyz = np.concatenate((xyz,np.array([1]), axis = 1))
    r = R.from_euler('zyx', theta, degrees=True)
    # r = r.as_matrix()
    r = r.as_dcm()
    cloud[:,:3] = np.dot(cloud[:,:3], r.T)
    return cloud



def parse_semanticKitti(data_dir):
    rospy.init_node('gnd_data_provider', anonymous=True)
    pcl_pub = rospy.Publisher("/kitti/velo/pointcloud", PointCloud2, queue_size=10)
    pcl_pub2 = rospy.Publisher("/kitti/raw/pointcloud", PointCloud2, queue_size=10)

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


        ##### Data Augmentation:
        if angle > 5.0:
            increase = False
        elif angle < -5.0:
            increase = True


        if increase:
            angle +=0.1
        else:
            angle -=0.1

        points  = rotate_cloud(points, theta = [0,angle,angle]) #zyx

        gnd_points, obs_points = segment_cloud(points.copy(),[40, 44, 48, 49,60,72])
        
        timestamp = rospy.Time.now()
        broadcast_TF(poses[f],timestamp)
        np2ros_pub(gnd_points, pcl_pub, timestamp)
        np2ros_pub(points, pcl_pub2,timestamp)
        
        rate.sleep()




if __name__ == '__main__':
    data_dir = "/home/anshul/es3cap/dataset/kitti_semantic/dataset/sequences/06/"

    parse_semanticKitti(data_dir)




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

