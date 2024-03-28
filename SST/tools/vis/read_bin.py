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
bin_path='/data/zhaoshenao/code/SST/data/ctrl_bins/training/fsd6f6e_vehicle_full_train.bin'
#第一条数据：object {
#   box {
#     center_x: 18.98877525329567
#     center_y: 12.61245536804239
#     center_z: 0.9655585289001749
#     width: 2.2819108963012695
#     length: 5.457942962646484
#     height: 1.96392822265625
#     heading: -1.6304353379653167
#   }
#   metadata {
#   }
#   type: TYPE_VEHICLE
#   id: "1_0"
# }
# score: 0.8879907727241516
# context_name: "10017090168044687777_6380_000_6400_000"
# frame_timestamp_micros: 1550083467346370

# idx_ts_path='/data/zhaoshenao/code/SST/data/waymo/kitti_format/idx2timestamp.pkl'
if __name__ == '__main__':
    bin_path = osp.abspath(bin_path)

    pred_dict = get_obj_dict_from_bin_file(bin_path, debug=True)

    ts_list = sorted(list(pred_dict.keys()))

    # with open


    dets = pred_dict[ts_list[10]]


    # # pc = get_pc_from_time_stamp(ts_list[10],idx_ts_path ,
    # #                                 split='training')
    # idx2ts = None
    # if idx2ts is None:
    #     with open(idx_ts_path, 'rb') as fr:
    #         idx2ts = pkl.load(fr)
    #     print('Read idx2ts')
    # ts2idx = {}
    # for idx, ts in idx2ts.items():
    #     ts2idx[ts] = idx
    # print(ts2idx[ts_list[10]])
    # # print(type(pred_dict))
    # # print(type(dets))
    # # print(len(dets))
    # # print(dets[:3])
    # # print(ts_list[0])
    # # print(pc[:3, :])
    # print(type(pc))
    # # print(pc.dtype)
    # print(pc.shape)
    for det in dets:
        print(BBox.bbox_7dim(det))
    #print(type(BBox.bbox2array(det)))

# pc_idx:1007000
#对应时间戳的第一个检测框[-35.76898575   4.07700586   1.08919846   5.02015972   2.204561  1.91004789   3.12882616]