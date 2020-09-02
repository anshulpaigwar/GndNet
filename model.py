#!/usr/bin/env python
import math
import numpy as np
import ipdb as pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.pointpillars import PillarFeatureNet, PointPillarsScatter
from modules.segnet import segnetGndEst



class GroundEstimatorNet(nn.Module):

    def __init__(self, cfg):
        super(GroundEstimatorNet, self).__init__()
        # voxel feature extractor
        self.cfg = cfg
        self.voxel_feature_extractor = PillarFeatureNet( num_input_features = cfg.input_features,
                use_norm = cfg.use_norm,
                num_filters=cfg.vfe_filters,
                with_distance=cfg.with_distance,
                voxel_size=cfg.voxel_size,
                pc_range=cfg.pc_range)

        grid_size = (np.asarray(cfg.pc_range[3:]) - np.asarray(cfg.pc_range[:3])) / np.asarray(cfg.voxel_size)
        grid_size = np.round(grid_size).astype(np.int64)
        dense_shape = [1] + grid_size[::-1].tolist() + [cfg.vfe_filters[-1]] #grid_size[::-1] reverses the index from xyz to zyx

        # Middle feature extractor
        self.middle_feature_extractor = PointPillarsScatter(output_shape = dense_shape,
                                        num_input_features = cfg.vfe_filters[-1])

        self.encoder_decoder = segnetGndEst(in_channels=64, is_unpooling=True)

    def forward(self, voxels, coors, num_points):
        # pdb.set_trace()
        voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
        spatial_features = self.middle_feature_extractor(voxel_features, coors, self.cfg.batch_size)
        gnd_pred = self.encoder_decoder(spatial_features)

        return torch.squeeze(gnd_pred)
