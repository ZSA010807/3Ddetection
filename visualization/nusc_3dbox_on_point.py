import torch
import numpy as np
from os import path as osp
import os
from mmengine import  load
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.visualization.test import pre_3dbox, gt_3dbox
from mmdet3d.visualization.read_nusc import pred, gt_boxes, box_pred_gt
# /home/work/Downloads/BEVFormer/data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151604547893.pcd.bin
# /home/work/Downloads/waymo/kitti_format/training/velodyne/1001190.bin
# infos = load('/home/work/fsdownload/nusc_boxes.pkl')
path = '/home/work/fsdownload'
source = '/home/work/fsdownload/vis_lidarfiles'
infos = box_pred_gt()
for info in infos:
    lidarpath = info['lidarpath']
    file_name = os.path.basename(lidarpath)
    source_path = osp.join(source, file_name)
    token = info['sample_token']
    pred_box = info['pred_box']
    # pred_box_da = info['pred_box_da']
    pred_box_da = []
    gt_box = info['gt_box']
    points = np.fromfile(source_path, dtype=np.float32)
    points = points.reshape(-1, 5)
    points = points[:, :4]
    visualizer = Det3DLocalVisualizer()
    # set point cloud in visualizer
    visualizer.set_points(points)
    bboxes_3d = LiDARInstance3DBoxes(
        torch.tensor(pred_box), origin=(0.5, 0.5, 0.5))
    bboxes_3d_da = LiDARInstance3DBoxes(
        torch.tensor(pred_box_da), origin=(0.5, 0.5, 0.5))
    gt_bboxes_3d = LiDARInstance3DBoxes(
        torch.tensor(gt_box), origin=(0.5, 0.5, 0.5))
    # Draw 3D bboxe
    visualizer.draw_bboxes_3d(bboxes_3d,bboxes_3d_da, gt_bboxes_3d)
    img_path = osp.join(path, '{}.pang'.format(token))
    visualizer.show(save_path=img_path)

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