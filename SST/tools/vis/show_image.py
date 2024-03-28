import numpy as np
import mmcv
from PIL import Image
from torchvision import transforms
from mmdet3d.core import show_multi_modality_result, show_result
from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
from test import pre_box
from mmengine import load
from visualizer import Visualizer2D,Visualizer3D
from visualizer import BBox
import pickle as pkl
from utils import get_obj_dict_from_bin_file, get_pc_from_time_stamp
gt_bin_path = '/data/zhaoshenao/code/SST/data/waymo/waymo_format/gt.bin'
path = '/data/zhaoshenao/code/SST/data/waymo/kitti_format/idx2timestamp.pkl'
gt_dict = get_obj_dict_from_bin_file(gt_bin_path, debug=False)
idx2ts = None
if idx2ts is None:
    with open(path, 'rb') as fr:
        idx2ts = pkl.load(fr)
    print('Read idx2ts')
gt_boxes = gt_dict[idx2ts['1003120']]
list = [BBox.type(box) for box in gt_boxes]
k1, k2, k3, k4 = 0, 0, 0, 0
for i in range(len(list)):
    if list[i] == 1.0:
        k1 = k1+1
    if list[i] == 2.0:
        k2 = k2+1
    if list[i] == 3.0:
        k3 = k3+1
    else:
        k4 = k4+1
print(k1,k2,k3,k4)
print(len(gt_boxes))
