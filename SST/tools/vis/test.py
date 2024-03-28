import numpy as np
import os
from os import path as osp
import argparse
from ipdb import set_trace
from tqdm import tqdm

# from pipeline_vis import frame_visualization
from visualizer import Visualizer2D, Visualizer3D
from utils import get_obj_dict_from_bin_file, get_pc_from_time_stamp
from visualizer import BBox
import pickle as pkl
bin_path='/data/zhaoshenao/code/SST/work_dirs/ctrl_veh_24e/result.bin'
save_folder='/data/zhaoshenao/code/SST/work_dirs/ctrl_veh_24e/vis_folder_3d'


def pre_box():
    idx_ts_path = '/data/zhaoshenao/code/SST/data/waymo/kitti_format/idx2timestamp.pkl'
    bin_path = '/data/zhaoshenao/code/SST/work_dirs/ctrl_veh_24e/result.bin'
    gt_bin_path = '/data/zhaoshenao/code/SST/data/waymo/waymo_format/gt.bin'
    # pred_dict = get_obj_dict_from_bin_file(bin_path, debug=False)
    gt_dict = get_obj_dict_from_bin_file(gt_bin_path, debug=False)

    # if gt_dict is not None:
    #     ts_list = sorted(list(gt_dict.keys()))
    # else:
    ts_list = sorted(list(gt_dict.keys()))

    # with open


    dets = gt_dict[ts_list[50]]


    #pc = get_pc_from_time_stamp(ts_list[0],idx_ts_path ,
                                   # split='training')
    idx2ts = None
    if idx2ts is None:
        with open(idx_ts_path, 'rb') as fr:
            idx2ts = pkl.load(fr)
        print('Read idx2ts')
    ts2idx = {}
    for idx, ts in idx2ts.items():
        ts2idx[ts] = idx
    print(ts2idx[ts_list[50]])
    # print(ts_list[0],ts2idx[ts_list[0]])
    # print(type(pred_dict))
    # print(type(dets))
    # print(len(dets))
    # print(dets[:3])
    # print(ts_list[0])
    # print(pc[:3, :])
    #print(type(pc))
    # print(pc.dtype)
    #print(pc.shape)
    dets_list = [BBox.bbox_7dim(det) for det in dets]
    dets_array = np.vstack(dets_list)
    return  dets_array
    #print(type(BBox.bbox2array(det)))
# pc_idx:1007000
#对应时间戳的第一个检测框[-35.76898575   4.07700586   1.08919846   5.02015972   2.204561  1.91004789   3.12882616]