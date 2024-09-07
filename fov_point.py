from os import path as osp
import os
import numpy as np
from mmengine import  load
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes
def cart_to_hom(pts):
    pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
    return pts_hom
def lidar_to_rect(pts_lidar, v2c):
    pts_lidar_hom = cart_to_hom(pts_lidar)
    pts_rect = np.dot(pts_lidar_hom, v2c.T)
    return pts_rect
def rect_to_img(pts_rect, p):
    pts_rect_hom = cart_to_hom(pts_rect)
    pts_2d_hom = np.dot(pts_rect_hom, p.T)
    pts_rect_hom[:, 2][pts_rect_hom[:, 2] == 0] = 1e-9
    pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T
    pts_rect_depth = pts_2d_hom[:, 2] - p.T[3,2]
    return pts_img, pts_rect_depth
def lidar_to_img(pts_lidar,lidar2img):
    pts_lidar_hom = cart_to_hom(pts_lidar)
    pts_2d_hom = np.dot(pts_lidar_hom, lidar2img.T)
    pts_rect_depth = pts_2d_hom[:, 2]
    return pts_2d_hom, pts_rect_depth
def get_valid_flag(pts_rect, pts_img, pts_rect_depth):
    img_shape = [1280, 1920]
    PC_AREA_SCOPE = [[-40, 40], [-1, 3], [0, 70.4]]
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    x_range, y_range, z_range = PC_AREA_SCOPE
    # pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
    # range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
    #              & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
    #              & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
    # pts_valid_flag = pts_valid_flag & range_flag
    return pts_valid_flag
def main():
    info_file = load('/home/work/Downloads/waymo/kitti_format/waymo_infos_val.pkl')
    path = '/home/work/Downloads/waymo/kitti_format/training/velodyne/1000001.bin'
    # for list in info_file['data_list']:
    #     if list['sample_idx'] == 1000001:
    #         info = list
    #         break
    t1 = [[-0.0017267926596105099, -0.9998841285705566, 0.0151247913017869, -0.052593063563108444],
          [0.0028620122466236353, -0.015129693783819675, -0.9998814463615417, 2.110311508178711],
          [0.9999943971633911, -0.001683300593867898, 0.002887806622311473, -1.5501034259796143],
          [0.0, 0.0, 0.0, 1.0]]
    t2 = [[2059.047119140625, 0.0, 935.1248168945312, 0.0],
          [0.0, 2059.047119140625, 635.052490234375, 0.0],
          [0.0, 0.0, 1.0, 0.0]]
    t3 = [[931.5640258789062, -2060.382568359375, 33.843116760253906, -1557.831787109375],
          [640.9419555664062, -32.221736907958984, -2056.96923828125, 3360.833740234375],
          [0.9999943971633911, -0.001683300593867898, 0.002887806622311473, -1.5501034259796143]]
    lidar2cam = np.array(t1)[:3]
    cam2img = np.array(t2)
    lidar2img = np.array(t3)
    points = np.fromfile(path, dtype=np.float32)
    points = points.reshape(-1, 6)
    points = points[:, :4]
    #
    points_rect = lidar_to_rect(points[:, 0:3], lidar2cam)
    pts_img, pts_rect_depth = rect_to_img(points_rect, cam2img)
    # pts_img, pts_rect_depth = lidar_to_img(points[:, 0:3], lidar2img)
    points_rect = 0
    pts_valid = get_valid_flag(points_rect, pts_img, pts_rect_depth)
    points = points[pts_valid]
    visualizer = Det3DLocalVisualizer()
    visualizer.set_points(points)
    visualizer.show(save_path='/home/work/Downloads/fov_point_2.png')
if __name__ == '__main__':
    main()

