### This code is proprietary ###

[kitti_semantic_dataset] ----> semKitti_parser.py ---->  agumented raw point cloud + agumented segmented gnd points

agumented segmented gnd points ----> CRF-gnd-estimator.py ----> smooth gnd elevation labels 

agumented raw point cloud + smooth gnd elevation labels ----> dataset_generator.py ----> [reduced/cropped agumented raw point cloud + smooth gnd elevation labels]   
