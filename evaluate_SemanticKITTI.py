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
from modules.loss_func import MaskedHuberLoss
from dataset_utils.dataset_provider import get_train_loader, get_valid_loader
from utils.point_cloud_ops import points_to_voxel
from utils.utils import np2ros_pub_2, gnd_marker_pub, segment_cloud, lidar_to_img, lidar_to_heightmap
import ipdb as pdb
import matplotlib.pyplot as plt
# Ros Includes
import rospy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
import numba
from numba import jit,types



use_cuda = torch.cuda.is_available()

if use_cuda:
    print('setting gpu on gpu_id: 0') #TODO: find the actual gpu id being used



rospy.init_node('gnd_data_provider', anonymous=True)
pcl_pub = rospy.Publisher("/kitti/velo/pointcloud", PointCloud2, queue_size=10)
# marker_pub_1 = rospy.Publisher("/kitti/ground_marker", Marker, queue_size=10)
marker_pub_2 = rospy.Publisher("/kitti/gnd_marker_pred", Marker, queue_size=10)

# rospy.init_node('gnd_data_provider', anonymous=True)
# pcl_pub = rospy.Publisher("/kitti/velo/pointcloud", PointCloud2, queue_size=10)
# pcl_pub2 = rospy.Publisher("/kitti/raw/pointcloud", PointCloud2, queue_size=10)
# marker_pub = rospy.Publisher("/kitti/ground_marker", Marker, queue_size=10)

#############################################xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx#######################################


parser = argparse.ArgumentParser()
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--config', default='config/config_kittiSem.yaml', type=str, metavar='PATH', help='path to config file (default: none)')
parser.add_argument('-v', '--visualize', dest='visualize', action='store_true', help='visualize model on validation set')
parser.add_argument('-s', '--save_checkpoints', dest='save_checkpoints', action='store_true',help='evaluate model on validation set')
parser.add_argument('--start_epoch', default=0, type=int, help='epoch number to start from')
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

print("setting batch_size to 1")
cfg.batch_size = 1

model = GroundEstimatorNet(cfg).cuda()
optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=0.0005)




def get_GndSeg(sem_label, GndClasses):
    index = np.isin(sem_label, GndClasses)
    GndSeg = np.ones(sem_label.shape)
    GndSeg[index] = 0
    index = np.isin(sem_label, [0,1])
    GndSeg[index] = -1
    return GndSeg



@jit(nopython=True)
def remove_outliers(pred_GndSeg, GndSeg): # removes the points outside grid and unlabled points
    index = pred_GndSeg >= 0
    pred_GndSeg = pred_GndSeg[index]
    GndSeg = GndSeg[index]

    index = GndSeg >=0
    pred_GndSeg = pred_GndSeg[index]
    GndSeg = GndSeg[index]
    return 1-pred_GndSeg, 1-GndSeg




@jit(nopython=True)
def _shift_cloud(cloud, height):
    cloud += np.array([0,0,height,0], dtype=np.float32)
    return cloud





def get_target_gnd(cloud, sem_label):
    if cloud.shape[0] != sem_label.shape[0]:
        raise Exception('Points and label MisMatch')

    index = np.isin(sem_label, [40, 44, 48, 49,60,72])
    gnd = cloud[index]
    gnd_mask = lidar_to_img(np.copy(gnd), np.asarray(cfg.grid_range), cfg.voxel_size[0], fill = 1)
    gnd_heightmap = lidar_to_heightmap(np.copy(gnd), np.asarray(cfg.grid_range), cfg.voxel_size[0], max_points = 100)
    return  gnd_heightmap, gnd_mask




def InferGround(cloud):

    cloud = _shift_cloud(cloud[:,:4], cfg.lidar_height)

    voxels, coors, num_points = points_to_voxel(cloud, cfg.voxel_size, cfg.pc_range, cfg.max_points_voxel, True, cfg.max_voxels)
    voxels = torch.from_numpy(voxels).float().cuda()
    coors = torch.from_numpy(coors)
    coors = F.pad(coors, (1,0), 'constant', 0).float().cuda()
    num_points = torch.from_numpy(num_points).float().cuda()
    with torch.no_grad():
            output = model(voxels, coors, num_points)
    return output


# visualize = False
visualize = True
# if visualize:
#     plt.ion()
#     fig = plt.figure()

def evaluate_SemanticKITTI(data_dir):

    velodyne_dir = data_dir + "velodyne/"
    label_dir = data_dir + 'labels/'
    frames = os.listdir(velodyne_dir)
    # calibration = parse_calibration(os.path.join(data_dir, "calib.txt"))
    # poses = parse_poses(os.path.join(data_dir, "poses.txt"), calibration)
    # rate = rospy.Rate(2) # 10hz
    # angle = 0
    # increase = True
    iou_score = 0
    mse_score = 0
    prec_score = 0
    recall_score = 0
    print(len(frames))
    for f in range(len(frames)):
        points_path = os.path.join(velodyne_dir, "%06d.bin" % f)
        points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)

        label_path = os.path.join(label_dir, "%06d.label" % f)
        sem_label = np.fromfile(label_path, dtype=np.uint32)
        sem_label = sem_label.reshape((-1))

        pred_gnd = InferGround(points)
        pred_gnd = pred_gnd.cpu().numpy()
        # TODO: Remove the points which are very below the ground
        pred_GndSeg = segment_cloud(points.copy(),np.asarray(cfg.grid_range), cfg.voxel_size[0], elevation_map = pred_gnd.T, threshold = 0.2)
        GndSeg = get_GndSeg(sem_label, GndClasses = [40, 44, 48, 49,60,72])
        
        if visualize:
            np2ros_pub_2(points, pcl_pub, None, pred_GndSeg)
            gnd_marker_pub(pred_gnd, marker_pub_2, cfg, color = "red")

        pred_GndSeg, GndSeg = remove_outliers(pred_GndSeg, GndSeg)
        intersection = np.logical_and(GndSeg, pred_GndSeg)
        union = np.logical_or(GndSeg, pred_GndSeg)
        iou = np.sum(intersection) / np.sum(union)
        prec = np.sum(intersection)/pred_GndSeg.sum()
        recall = np.sum(intersection)/GndSeg.sum()
        # tn =   np.count_nonzero(pred_GndSeg==0) - np.sum(intersection)
        # iou = np.count_nonzero(pred_GndSeg==0)/(np.count_nonzero(GndSeg==0)+tn)
        iou_score += iou
        prec_score += prec
        recall_score += recall



        target_gnd, gnd_mask = get_target_gnd(points, sem_label)
        # if visualize: 
            # fig.clear()
            # fig.add_subplot(1, 3, 1)
            # plt.imshow(gnd_mask, interpolation='nearest')
            # fig.add_subplot(1, 3, 2)
            # cs = plt.imshow(target_gnd*gnd_mask, interpolation='nearest')
            # cbar = fig.colorbar(cs)
            # fig.add_subplot(1, 3, 3)
            # cs = plt.imshow(pred_gnd.T*gnd_mask, interpolation='nearest')
            # cbar = fig.colorbar(cs)
            # plt.show()

        mse = (np.square(target_gnd - pred_gnd.T)*gnd_mask).sum()
        mse = mse/gnd_mask.sum()
        mse_score += mse

        print(f, iou, mse, prec, recall)
        pdb.set_trace()

    iou_score = iou_score/len(frames)
    mse_score = mse_score/len(frames)
    recall_score = recall_score/len(frames)
    prec_score = prec_score/len(frames)
    print(iou_score, mse_score, prec_score, recall_score)







def main():
    # rospy.init_node('pcl2_pub_example', anonymous=True)
    global args
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['lowest_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        raise Exception('please specify checkpoint to load')

    data_dir = "/home/anshul/es3cap/semkitti_gndnet/kitti_semantic/dataset/sequences/07/"
    evaluate_SemanticKITTI(data_dir)



if __name__ == '__main__':
    main()
