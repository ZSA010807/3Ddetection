import torch
import numpy as np
from os import path as osp
import os
from mmengine import  load
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes, Box3DMode
from mmdet3d.structures import CameraInstance3DBoxes
from mmdet3d.visualization.test import pre_3dbox, gt_3dbox
from mmdet3d.visualization.read_nusc import pred, gt_boxes, box_pred_gt
# /home/work/Downloads/BEVFormer/data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151604547893.pcd.bin
# /home/work/Downloads/waymo/kitti_format/training/velodyne/1001190.bin
# infos = load('/home/work/fsdownload/nusc_boxes.pkl')
p_gt = '/home/work/Downloads/waymo/kitti_format/waymo_infos_val.pkl'
p_gt2 = '/home/work/Downloads/pklfile/gt_annos_car.pkl'
p_pre = '/home/work/Downloads/pklfile/nusc_waymo_car_75m_3x.pkl'
source = '/home/work/Downloads/waymo/kitti_format/training/velodyne/1003120.bin'
gt = load(p_gt)
pre = load(p_pre)
pre_box_da = []
gt_box = []
j = 0
for index, info in enumerate(gt['data_list']):
    if info['sample_idx'] == 1003120:
        j = index
        lidar2cam = info['images']['CAM_FRONT']['lidar2cam']
        for i in range(len(info['instances'])):
            if info['instances'][i]['bbox_label_3d'] == 0:
                gt_box.append(info['instances'][i]['bbox_3d'])

pre_box = np.concatenate((pre[j]['location'], pre[j]['dimensions'], pre[j]['rotation_y'].reshape(-1, 1)), axis=1)
gt_box = np.array(gt_box, dtype=np.float32)
pre_box = np.array(pre_box, dtype=np.float32)
pre_box_da = np.array(pre_box_da, dtype=np.float32)
points = np.fromfile(source, dtype=np.float32)
points = points.reshape(-1, 6)
points = points[:, :4]
visualizer = Det3DLocalVisualizer()
# set point cloud in visualizer
visualizer.set_points(points)
bboxes_3d = CameraInstance3DBoxes(
    torch.tensor(pre_box), origin=(0.5, 1.0, 0.5)).convert_to(Box3DMode.LIDAR, np.linalg.inv(lidar2cam))
bboxes_3d_da = CameraInstance3DBoxes(
    torch.tensor(pre_box_da), origin=(0.5, 1.0, 0.5))
gt_bboxes_3d = LiDARInstance3DBoxes(
    torch.tensor(gt_box), origin=(0.5, 0.5, 0))
# Draw 3D bboxe
visualizer.draw_bboxes_3d(bboxes_3d, bboxes_3d_da, gt_bboxes_3d)
visualizer.show(save_path='/home/work/Downloads/pc_vis/1009132.png')

# points = np.fromfile('/home/work/Downloads/waymo/kitti_format/training/velodyne/1006073.bin', dtype=np.float32)
# points = points.reshape(-1, 6)
# # min = np.min(points[:, 3])
# # max = np.max((points[:, 3]))
# # nor = (points[:, 3] - min) / (max - min)
# # points[:, 3] = nor
# # p_min = np.min(points[:, 3])
# # p_max = np.max((points[:, 3]))
# points = points[:, :4]
# # points[:, [0, 1]] = points[:, [1, 0]]
# # points[:, 1] = -points[:, 1]
# # points[:, :3] += np.array([0,0,1.8])
# visualizer = Det3DLocalVisualizer()
# # set point cloud in visualizer
# visualizer.set_points(points)
# p_boxes= pre_3dbox()
# gt_boxes = gt_3dbox()
# p_box_2 = []
# bboxes_3d = LiDARInstance3DBoxes(
#     torch.tensor(p_boxes))
# gt_bboxes_3d = LiDARInstance3DBoxes(
#     torch.tensor(gt_boxes))
# p_boxes_2 = LiDARInstance3DBoxes(torch.tensor(p_box_2))
# # Draw 3D bboxe
# visualizer.draw_bboxes_3d(bboxes_3d, p_boxes_2, gt_bboxes_3d)
# visualizer.show(save_path='/home/work/Downloads/waymo/1006073.png')