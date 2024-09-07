import numpy as np
import os
from os import path as osp
import argparse
from tqdm import tqdm

from mmdet3d.structures import LiDARInstance3DBoxes
# from pipeline_vis import frame_visualization
from mmdet3d.visualization.utils import get_obj_dict_from_bin_file, get_pc_from_time_stamp
from mmdet3d.visualization.visualizer import BBox
import pickle as pkl
from mmengine import load
# bin_path = '/home/work/Downloads/waymo/result.bin'
# save_folder = '/home/work/Downloads/waymo/vis_3d'
# idx_ts_path = '/home/work/Downloads/waymo/kitti_format/idx2timestamp.pkl'
def pre_3dbox():
    bin_path = '/home/work/Downloads/waymo/result_remove.bin'
    idx_ts_path = '/home/work/Downloads/waymo/kitti_format/idx2timestamp.pkl'
    pred_dict = get_obj_dict_from_bin_file(bin_path, debug=False)

    # ts_list = sorted(list(pred_dict.keys()))

    #pc = get_pc_from_time_stamp(ts_list[10],idx_ts_path ,
    #                                 split='training')
    idx2ts = None
    if idx2ts is None:
        with open(idx_ts_path, 'rb') as fr:
            idx2ts = pkl.load(fr)
        print('Read idx2ts')
    # ts2idx = {}
    # for idx, ts in idx2ts.items():
    #     ts2idx[ts] = idx
    # print(ts_list[0],ts2idx[ts_list[0]])
    # print(idx2ts['1001190'])
    # print(idx2ts['1007000'])
    dets = pred_dict[idx2ts['1006073']]
    # type_list = [BBox.type(det) for det in dets]
    # print(type_list)
    dets_list = []
    for det in dets:
        if BBox.type(det) == 1.0:
            dets_list.append(BBox.bbox_7dim(det))
    dets_array = np.vstack(dets_list)
    # print(dets_array.shape)
    return dets_array
def gt_3dbox():
    info_file = load('/home/work/Downloads/waymo/kitti_format/waymo_infos_val.pkl')
    for list in info_file['data_list']:
        if list['sample_idx'] == 1006073:
            info = list
            break
    bboxes = []
    for instance in info['instances']:
        if instance['bbox_label_3d'] == 0:  #0:veh 1:ped 2:cyc
            bboxes.append(instance['bbox_3d'])
    gt_bboxes_3d = np.array(bboxes, dtype=np.float32)
    return gt_bboxes_3d
if __name__ == '__main__':
    pre_3dbox().shape
    gt_3dbox().shape
    print('ok')
    # print(type(pred_dict))
    # print(type(dets))
    # print(len(dets))
    # print(dets[:3])
    # print(ts_list[0])
    # print(pc[:3, :])
    # print(type(pc))
    # # print(pc.dtype)
    # print(pc.shape)
    # print(dets_array)
    #print(type(BBox.bbox2array(det)))

# pc_idx:1007000
#对应时间戳的第一个检测框[-35.76898575   4.07700586   1.08919846   5.02015972   2.204561  1.91004789   3.12882616]