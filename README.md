# GndNet: Fast Ground plane Estimation and Point Cloud Segmentation for Autonomous Vehicles.
---
Authors: Anshul Paigwar, Ozgur Erkent, David Sierra Gonzalez, Christian Laugier

Teaser Img

## Introduction

This repository is code release for our paper accepted in International conference on Robotic Systems,IROS 2020. [presentation](https://sites.google.com/view/wad2019/overview).
In this work, we study 3D object detection directly from point clouds obtained
from 3D LiDARS.

## Abstract

Ground plane estimation and ground point seg-mentation is a crucial precursor for many applications in robotics and intelligent vehicles like navigable space detection and occupancy grid generation, 3D object detection, point cloud matching for localization and registration for mapping. In this paper, we present GndNet, a novel end-to-end approach that estimates the ground plane elevation information in a grid-based representation and segments the ground points simultaneously in real-time. GndNet uses PointNet and Pillar Feature Encoding network to extract features and regresses ground height for each cell of the grid. We augment the SemanticKITTI dataset to train our network. We demonstrate qualitative and quantitative evaluation of our results for ground elevation estimation and semantic segmentation of point cloud. GndNet establishes a new state-of-the-art, achieves a run-time of 55Hz for ground plane estimation and ground point segmentation. 
<img src="https://github.com/anshulpaigwar/GndNet/blob/master/doc/GndNet_architecture_final.png" alt="drawing" width="800"/>

## Installation

We have tested the algorithm on the system with Ubuntu 18.04, 12 GB RAM and NVIDIA GTX-1080.

### Dependencies
```
Python 3.6
CUDA (tested on 10.1)
PyTorch (tested on 1.4)
scipy
ipdb
argparse
numba
```
### Visualization
For visualisation of the ground estimation, semantic segmentation of pointcloud, and easy integration with our real system we use Robot Operating System (ROS):
```
ROS
ros_numpy
```
## Data Preparation

* We train our model on agumented SematicKITTI dataset [link](http://www.semantic-kitti.org/).
* We subdivide object classes in SematicKITTI dataset in two categories 
	1. Ground(road, sidewalk, parking, other-ground, vegetation, terrain)
	2. Non ground(all other)
* To prepare our ground elevation dataset we take only ground points and use CRF-based surface fitting method described in [1].
* Ground labels are generated in a form of 2D grid with cell resolution 1m x 1m. Values of each cell represent the local elevation of ground.
* We store ground labels and raw point cloud (both ground and non ground points) to train our network.
* We provide a sample dataset in this repository, full dataset can be made available on request.

<!-- Data augumentation Img

Download the KITTI 3D object detection dataset from [link](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
  Your data directory should look like:
```
|--data_object_velodyne
    |--training
        |--calib
        |--label_2
        |--velodyne
```
To generate the augumented dataset for the training and validation of Attentional PointNet
 use the code in folder kitti_custom:

 ```
 python kitti_lidarImg_data_generator.py
 ```
It will generate the dataset in following format:
```
|--attentional_pointnet_data
    |--validation
    |--training
        |--heightmap_crop
            |--0000.png
            |--
        |--labels
            |--0000.txt
            |--
        |--velodyne_crop
            |--0000.npy
            |--
```
Form each cropped region of 12m x 12m we have point cloud data, heightmap and a label file.
 Each label file contains three instances. These instances could be a car or non-car.
 depending upon number of cars in the scene. Instances are defined as

```
Float x,y,z
Float theta
Int theta_binned
Float H, W, L
Bool category car/ non-car
```
For non-car category we keep a fixed x,y,z which is outside of 12m x 12m region. -->

## Training

To train the model update the data directory path in the config file: config_kittiSem.yaml
```
python main.py -s
```
 It take around 6 hours for the network to converge and model parameters would be stored
 in checkpoint.pth.tar file. A pretrained model is provided in trained_models folder it can be used to 
 evaluate a sequence in SemanticKITTI dataset.

```
python evaluate_SemanticKITTI.py --resume checkpoint.pth.tar --data_dir /home/.../kitti_semantic/dataset/sequences/07/
```

## Using pre-trained model
Download SemanticKITTI dataset from their website [link](http://www.semantic-kitti.org/). To visualize the output we use ROS and rviz. The predicted class (ground or non-ground) of the points in the point cloud is substituted in the intensity field of sensor_msgs.pointcloud. In the rviz use intensity as color transformer to visualize segmented pointcloud. For the visualisation of Ground elevation we use ros line marker. 

```
roscore
rviz
python evaluate_SemanticKITTI.py --resume trained_models/checkpoint.pth.tar -v -gnd --data_dir /home/.../SemanticKITTI/dataset/sequences/00/
```
Note: The current version of the code for visualisation is written in python which can be very slow specifically generation of ros marker.
To only visualize segmentation output without ground elevation remove the `-gnd` flag.

## Results



## TODO
* Current dataloader loads the entire dataset in to the RAM first, this reduces training time but can be hogging for system with low RAM.
* Speed up visualisation of ground elevation. Write C++ code for ros marker 
* Create generalised ground elevation dataset with corespondence to SemanticKitti


## Citation

If you find this project useful in your research, please consider cite:
```
@inproceedings{paigwar2020gndnet,
  title={GndNet: Fast Ground Plane Estimation and Point Cloud Segmentation for Autonomous Vehicles},
  author={Paigwar, Anshul and Erkent, {\"O}zg{\"u}r and Gonz{\'a}lez, David Sierra and Laugier, Christian},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020}
}
```

## Contribution

We Welcome you in contributing to this repo, and feel free to contact us for any potential bugs and issues.


## References
---
[1] L. Rummelhard, A. Paigwar, A. Nègre and C. Laugier, "Ground estimation and point cloud segmentation using SpatioTemporal Conditional Random Field," 2017 IEEE Intelligent Vehicles Symposium (IV), Los Angeles, CA, 2017, pp. 1105-1110, doi: 10.1109/IVS.2017.7995861.

[2] Behley, J., Garbade, M., Milioto, A., Quenzel, J., Behnke, S., Stachniss, C., & Gall, J. (2019). SemanticKITTI: A dataset for semantic scene understanding of lidar sequences. In Proceedings of the IEEE International Conference on Computer Vision (pp. 9297-9307).
