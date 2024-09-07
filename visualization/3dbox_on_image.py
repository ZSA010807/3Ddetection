import mmcv
import numpy as np
from mmengine import load

from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes, Box3DMode
from mmdet3d.structures import CameraInstance3DBoxes
from mmdet3d.visualization.test import pre_3dbox
cam = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_SIDE_LEFT', 'CAM_SIDE_RIGHT']
info_file = load('/home/work/Downloads/waymo/kitti_format/waymo_infos_val.pkl')
for list in info_file['data_list']:
    if list['sample_idx'] == 1124081:
        info=list
        break
# trans = info['images']['CAM_FRONT']['lidar2img']
# row = [0.0, 0.0, 0.0, 1.0]
# trans.append(row)
cam_list = []
trans = []
row = [0.0, 0.0, 0.0, 1.0]
for i in cam:
    trans = info['images'][i]['lidar2img']
    trans.append(row)
    cam_list.append(trans)

boxes = pre_3dbox()
gt_bboxes_3d = np.array(boxes, dtype=np.float32)
gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d)

for i in range(5):
    lidar2img = np.array(cam_list[i], dtype=np.float32)
    input_meta = {'lidar2img': lidar2img}

    visualizer = Det3DLocalVisualizer()
    path = '/home/work/Downloads/waymo/kitti_format/training/image_' + str(i) + '/1124081.jpg'
    img = mmcv.imread(path)
    img = mmcv.imconvert(img, 'bgr', 'rgb')
    visualizer.set_image(img)
    # project 3D bboxes to image
    visualizer.draw_proj_bboxes_3d(gt_bboxes_3d, input_meta)
    visualizer.show()
    visualizer.show(save_path=img_path)