#!/usr/bin/env python

"""
Author: Anshul Paigwar
email: p.anshul6@gmail.com
"""



import argparse
import os
import shutil
import yaml
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# from modules import gnd_est_Loss
from model import GroundEstimatorNet
from utils.point_cloud_ops import points_to_voxel
from utils.utils import cloud_msg_to_numpy, segment_cloud
from utils.ros_utils import np2ros_pub_2, gnd_marker_pub
import ipdb as pdb

# Ros Includes
import rospy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker

use_cuda = torch.cuda.is_available()

if use_cuda:
    print('setting gpu on gpu_id: 0') #TODO: find the actual gpu id being used



rospy.init_node('gnd_data_provider', anonymous=True)
pcl_pub = rospy.Publisher("/kitti/gndnet_segcloud", PointCloud2, queue_size=10)
marker_pub_1 = rospy.Publisher("/kitti/ground_marker", Marker, queue_size=10)
marker_pub_2 = rospy.Publisher("/kitti/gnd_marker_pred", Marker, queue_size=10)


#############################################xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx#######################################


parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-v', '--visualize', dest='visualize', action='store_true', help='visualize model on validation set')
parser.add_argument('--config', default='config/config_kittiSem.yaml', type=str, metavar='PATH', help='path to config file (default: none)')
args = parser.parse_args()


if os.path.isfile(args.config):
    print("using config file:", args.config)
    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    class ConfigClass:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    cfg = ConfigClass(**config_dict) # convert python dict to class for ease of use

else:
    print("=> no config file found at '{}'".format(args.config))

cfg.batch_size = 1
#############################################xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx#######################################


model = GroundEstimatorNet(cfg).cuda()


if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
else:
    raise Exception('please specify checkpoint to load')



# switch to evaluate mode
model.eval()


def callback(cloud_msg):

    start_time = time.time()
    # cloud = process_cloud(cloud_msg, cfg, shift_cloud = True, sample_cloud = False)
    cloud = cloud_msg_to_numpy(cloud_msg, cfg, shift_cloud = True)

    # np_conversion = time.time()
    # print("np_conversion_time: ", np_conversion- start_time)

    voxels, coors, num_points = points_to_voxel(cloud, cfg.voxel_size, cfg.pc_range, cfg.max_points_voxel, True, cfg.max_voxels)
    voxels = torch.from_numpy(voxels).float().cuda()
    coors = torch.from_numpy(coors)
    coors = F.pad(coors, (1,0), 'constant', 0).float().cuda()
    num_points = torch.from_numpy(num_points).float().cuda()

    # cloud_process = time.time()
    # print("cloud_process: ", cloud_process - np_conversion)



    with torch.no_grad():
            output = model(voxels, coors, num_points)
            # model_time = time.time()
            # print("model_time: ", model_time - cloud_process)

    pred_GndSeg = segment_cloud(cloud.copy(),np.asarray(cfg.grid_range), cfg.voxel_size[0], elevation_map = output.cpu().numpy().T, threshold = 0.2)
    # seg_time = time.time()
    # print("seg_time: ", seg_time - model_time )
    # print("total_time: ", seg_time - np_conversion)
    # print()
    # pdb.set_trace()
    gnd_marker_pub(output.cpu().numpy(),marker_pub_2, cfg, color = "red")
    np2ros_pub_2(cloud, pcl_pub, None, pred_GndSeg)
    # vis_time = time.time()
    # print("vis_time: ", vis_time - model_time)



def listener():
    rospy.Subscriber("/kitti/velo/pointcloud", PointCloud2, callback, queue_size = 1)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()







if __name__ == '__main__':
    listener()

