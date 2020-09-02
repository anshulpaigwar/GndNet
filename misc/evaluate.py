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
from utils.utils import save_checkpoint, AverageMeter,np2ros_pub_2, gnd_marker_pub, visualize_2D, segment_cloud,np2ros_pub
import ipdb as pdb
import matplotlib.pyplot as plt
# Ros Includes
import rospy
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker

use_cuda = torch.cuda.is_available()

if use_cuda:
    print('setting gpu on gpu_id: 0') #TODO: find the actual gpu id being used



rospy.init_node('gnd_data_provider', anonymous=True)
pcl_pub = rospy.Publisher("/kitti/velo/pointcloud", PointCloud2, queue_size=10)
marker_pub_1 = rospy.Publisher("/kitti/ground_marker", Marker, queue_size=10)
marker_pub_2 = rospy.Publisher("/kitti/gnd_marker_pred", Marker, queue_size=10)


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


#############################################xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx#######################################




# train_loader =  get_train_loader(cfg.data_dir, cfg.batch_size)
valid_loader =  get_valid_loader(cfg.data_dir, cfg.batch_size, skip = 5)

model = GroundEstimatorNet(cfg).cuda()
optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=0.0005)
lossHuber = nn.SmoothL1Loss().cuda()
# masked_huber_loss = MaskedHuberLoss().cuda()


def evaluate():

    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    # switch to evaluate mode
    # model.eval()
    # if args.evaluate:
    #     model.train()
    with torch.no_grad():
        start = time.time()
        
        for batch_idx, (data, labels) in enumerate(valid_loader):

            data_time.update(time.time() - start) # measure data loading time
            B = data.shape[0] # Batch size
            N = data.shape[1] # Num of points in PointCloud

            voxels = []; coors = []; num_points = []
            data = data.numpy()
            for i in range(B):
                fig = plt.figure()
                v, c, n = points_to_voxel(data[i], cfg.voxel_size, cfg.pc_range, cfg.max_points_voxel, True, cfg.max_voxels)
                # mask = np.zeros((100,100))
                # indi = c[:,1:]
                # mask[tuple(indi.T)] = 1
                # fig.clf()
                # fig.add_subplot(1, 2, 1)
                # plt.imshow(mask, interpolation='nearest')
                # fig.add_subplot(1, 2, 2)
                # plt.imshow(labels[i], interpolation='nearest')
                # plt.show()
                # # visualize_2D(mask, data[i], fig, cfg)
                # pdb.set_trace()
                
                c = torch.from_numpy(c)
                c = F.pad(c, (1,0), 'constant', i)
                voxels.append(torch.from_numpy(v))
                coors.append(c)
                num_points.append(torch.from_numpy(n))

            voxels = torch.cat(voxels).float().cuda()
            coors = torch.cat(coors).float().cuda()
            num_points = torch.cat(num_points).float().cuda()
            labels = labels.float().cuda()

            optimizer.zero_grad()
            output = model(voxels, coors, num_points)
            # pdb.set_trace()

            # loss = masked_huber_loss(output, labels, mask)
            loss = lossHuber(output, labels)

            losses.update(loss.item(), B)

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            if args.visualize:
                for j in range(B):
                    pdb.set_trace()
                    out = output[j].cpu().numpy()
                    seg = segment_cloud(data[j].copy(),np.asarray(cfg.grid_range), cfg.voxel_size[0], elevation_map = out.T, threshold = 0.3)
                    np2ros_pub_2(data[j],pcl_pub, None, seg)
                    # np2ros_pub(data[j],pcl_pub)
                    # gnd_marker_pub(labels[j].cpu().numpy(),marker_pub_1, cfg, color = "red")
                    gnd_marker_pub(out,marker_pub_2, cfg, color = "red")




            if batch_idx % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       batch_idx, len(valid_loader), batch_time=batch_time, loss=losses))

    return losses.avg





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


    evaluate()



if __name__ == '__main__':
    main()
