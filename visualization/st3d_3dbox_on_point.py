import mmengine
import torch
import numpy as np
from os import path as osp
import os
from mmengine import  load
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.visualization.read_nusc import pred, gt_boxes, box_pred_gt
import pickle as pkl
# /home/work/Downloads/BEVFormer/data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151604547893.pcd.bin
# /home/work/Downloads/waymo/kitti_format/training/velodyne/1001190.bin
# infos = load('/home/work/fsdownload/nusc_boxes.pkl')
def generate_boxes(info_path):
    info_gt = load(info_path)
    vis_info = {
        'lidar_path': [],
        'token': [],
        'gt_boxes': [],
        'gt_names': [],
        'num_car': []
    }
    for i in range(0, len(info_gt['infos']), 10):
        vis_info['lidar_path'].append(info_gt['infos'][i]['lidar_path'])
        vis_info['token'].append(info_gt['infos'][i]['token'])
        boxes = []
        vis_info['gt_names'].append('car')
        for j in range(len(info_gt['infos'][i]['gt_names'])):
            if info_gt['infos'][i]['gt_names'][j] == 'car':
                boxes.append(info_gt['infos'][i]['gt_boxes'][j])
        vis_info['gt_boxes'].append(boxes)
        vis_info['num_car'].append(len(boxes))
    mmengine.dump(vis_info, '/home/work/Downloads/car.pkl')
    print('ok')
def generate_ps_boxes(label_path, gt_car_path):
    infos = pkl.load(open(gt_car_path, 'rb'))
    labels = pkl.load(open(label_path, 'rb'))
    label_info = {
        'file_name': [],
        'gt_boxes': [],
        'cls_scores': [],
        'iou_scores': [],
        'num_car': []
    }
    t = 0
    for path in infos['lidar_path']:
        # 提取文件名（包括扩展名）
        file_name_with_ext = os.path.basename(path)
        # 去除扩展名 .bin
        file_name = os.path.splitext(file_name_with_ext)[0]
        if file_name in labels:
                label_info['file_name'].append(file_name)
                gtboxes = labels[file_name]['gt_boxes'][:, :7]
                gtboxes[:, 0:3] -= np.array([0, 0, 1.8])
                label_info['gt_boxes'].append(gtboxes)
                if labels[file_name]['cls_scores'] is not None:
                    label_info['cls_scores'].append(labels[file_name]['cls_scores'])
                else:
                    label_info['cls_scores'].append(labels[file_name]['gt_boxes'][:, 8])
                if labels[file_name]['iou_scores'] is not None:
                    label_info['iou_scores'].append(labels[file_name]['iou_scores'])
                else:
                    label_info['iou_scores'].append(labels[file_name]['gt_boxes'][:, 8])
                label_info['num_car'].append(len(gtboxes))
                print(label_info['num_car'])
    mmengine.dump(label_info, '/home/work/Downloads/ps_labels_dts_cls.pkl')
    print(len(infos['lidar_path']),len(label_info['gt_boxes']))
def remove_boxes(gt_car_path, ps_car_path):
    vis_info = {
        'lidar_path': [],
        'token': [],
        'gt_boxes': [],
        'gt_names': [],
        'num_car': []
    }
    infos = pkl.load(open(gt_car_path, 'rb'))
    labels = pkl.load(open(ps_car_path, 'rb'))
    for i in range(len(infos['lidar_path'])):
        # 提取文件名（包括扩展名）
        file_name_with_ext = os.path.basename(infos['lidar_path'][i])
        # 去除扩展名 .bin
        file_name = os.path.splitext(file_name_with_ext)[0]
        if file_name in labels['file_name']:
            vis_info['lidar_path'].append(file_name_with_ext)
            vis_info['token'].append(infos['token'][i])
            vis_info['gt_boxes'].append(infos['gt_boxes'][i])
            vis_info['gt_names'].append(infos['gt_names'][i])
            vis_info['num_car'].append(infos['num_car'][i])
            print(vis_info['num_car'])
    mmengine.dump(vis_info, '/home/work/Downloads/gt_labels.pkl')
    print('ok')
def prepare_pkl():
    # info_path = '/home/work/Downloads/BEVFormer/data/nuscenes/nuscenes_infos_temporal_train.pkl'
    # generate_boxes(info_path)
    label_path = '/home/work/Downloads/ps_label_cls_score0.0_dts.pkl'
    gt_car_path = '/home/work/Downloads/car.pkl'
    # gt_labels_path = '/home/work/Downloads/gt_labels.pkl'
    # ps_car_path = '/home/work/Downloads/ps_labels.pkl'
    generate_ps_boxes(label_path, gt_car_path)
    # remove_boxes(gt_car_path, ps_car_path)
def vis_boxes():
    path = '/home/work/Downloads/DTS/vis_cls_score_distance'
    source = '/home/work/Downloads/BEVFormer/data/nuscenes/samples/LIDAR_TOP'
    gt_labels_path = '/home/work/Downloads/gt_labels.pkl'
    ps_car_path = '/home/work/Downloads/ps_labels_dts_cls.pkl'
    gt_info = pkl.load(open(gt_labels_path, 'rb'))
    ps_info = pkl.load(open(ps_car_path, 'rb'))
    for i in range(len(gt_info['lidar_path'])):
        ps_box_high = []
        ps_box_low = []
        file_name = gt_info['lidar_path'][i]
        source_path = osp.join(source, file_name)
        gt_box = gt_info['gt_boxes'][i]
        ps_box = np.array(ps_info['gt_boxes'][i])
        distance = np.sqrt(np.sum(ps_box[:, :2]**2, axis=1))
        for j in range(len(ps_info['cls_scores'][i])):
            if distance[j] <= 30:
                if ps_info['cls_scores'][i][j] >= 0.35:
                    ps_box_high.append(ps_info['gt_boxes'][i][j])
                else:
                    ps_box_low.append(ps_info['gt_boxes'][i][j])
            elif (distance[j] > 30) & (distance[j] <= 50):
                if ps_info['cls_scores'][i][j] >= 0.35:
                    ps_box_high.append(ps_info['gt_boxes'][i][j])
                else:
                    ps_box_low.append(ps_info['gt_boxes'][i][j])
            else:
                if ps_info['cls_scores'][i][j] >= 0.35:
                    ps_box_high.append(ps_info['gt_boxes'][i][j])
                else:
                    ps_box_low.append(ps_info['gt_boxes'][i][j])
            # if ps_info['cls_scores'][i][j] >= 0.4:
            #     ps_box_high.append(ps_info['gt_boxes'][i][j])
            # else:
            #     ps_box_low.append(ps_info['gt_boxes'][i][j])

        points = np.fromfile(source_path, dtype=np.float32)
        points = points.reshape(-1, 5)
        points = points[:, :4]
        visualizer = Det3DLocalVisualizer()
        # set point cloud in visualizer
        visualizer.set_points(points)
        bboxes_3d_low = LiDARInstance3DBoxes(
            torch.tensor(ps_box_low), origin=(0.5, 0.5, 0.5))
        bboxes_3d_high = LiDARInstance3DBoxes(
            torch.tensor(ps_box_high), origin=(0.5, 0.5, 0.5))
        gt_bboxes_3d = LiDARInstance3DBoxes(
            torch.tensor(gt_box), origin=(0.5, 0.5, 0.5))
        # Draw 3D bboxe
        visualizer.draw_bboxes_3d(bboxes_3d_high, bboxes_3d_low, gt_bboxes_3d)
        name = str(i) + '_gt' + str(gt_info['num_car'][i]) + str('_pslabels') + str(ps_info['num_car'][i]) + str('_pos') + str(len(ps_box_high))
        img_path = osp.join(path, '{}.pang'.format(name))
        visualizer.show(save_path=img_path)
def main():
    # prepare_pkl()
    vis_boxes()
if __name__ == '__main__':
    main()


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