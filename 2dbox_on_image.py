import  torch
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from mmengine.visualization import Visualizer
import pickle as pkl
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes, Box3DMode
from os import path as osp
from itertools import chain

def remove_close(points, radius=1.0):
    x = np.abs(points[:, 0]) < radius
    y = np.abs(points[:, 1]) < radius
    not_close = np.logical_not(np.logical_and(x, y))
    return points[not_close]
def project_from_ego_to_cam(pts_3d, extrinsic):
    pts_3d_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
    uv_cam = extrinsic.dot(pts_3d_hom.transpose()).transpose()[:, 0:3]
    return uv_cam


def project_cam_to_image(intrinsic, points_rect):
    hom = np.hstack((points_rect, np.ones((points_rect.shape[0], 1))))
    pts_2d = np.dot(hom, np.transpose(intrinsic)) # nx3
    pts_2d[:,0] /= pts_2d[:,2]
    pts_2d[:,1] /= pts_2d[:,2]
    return pts_2d[:, :2]

def CCL(points, projected_points, bboxes, depth):
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse.csgraph import connected_components
    # 输入数据：
    # points: 3D点云的坐标, 形状为 (N, 3)
    # projected_points: 对应的2D投影点, 形状为 (N, 2)
    # bboxes: 2D边界框的坐标, 形状为 (M, 4), 每个框为 [x_min, y_min, x_max, y_max]

    def point_in_bbox(point, bbox):
        """ 判断点是否在边界框内 """
        x_min, y_min, x_max, y_max = bbox
        x, y = point
        return x_min <= x <= x_max and y_min <= y <= y_max

    # Step 1: 关联3D点云与2D边界框
    bbox_labels = []  # 记录每个3D点所属的边界框标签

    for p2d in projected_points:
        found_label = False
        for i, bbox in enumerate(bboxes):
            if point_in_bbox(p2d, bbox):
                bbox_labels.append(i)
                found_label = True
                break
        if not found_label:
            bbox_labels.append(-1)  # 如果没有匹配到边界框, 则设为-1（或忽略）

    # Step 2: 根据2D边界框标签分组3D点云
    bbox_to_points = {}
    bbox_to_points_depth = {}
    for i, label in enumerate(bbox_labels):
        if label == -1:
            continue  # 跳过没有匹配到边界框的点
        if label not in bbox_to_points:
            bbox_to_points[label] = []
            bbox_to_points_depth[label] = []
        bbox_to_points[label].append(points[i])
        bbox_to_points_depth[label].append(depth[i])

    # 绘制直方图
    # plt.hist(bbox_to_points_depth[9], bins=30, color='blue', alpha=0.7)
    #
    # # 设置标题和轴标签
    # plt.title('Depth Values Distribution')
    # plt.xlabel('Depth Value')
    # plt.ylabel('Frequency')
    #
    # # 显示图表
    # plt.show()
    # Step 3: 在3D空间中对每个分组进行CCL聚类
    epsilon = 1.5  # 定义邻域距离
    clusters = []

    for bbox, points_in_bbox in bbox_to_points.items():
        points_in_bbox = np.array(points_in_bbox)

        if points_in_bbox.shape[0] < 1:
            clusters.append(points_in_bbox)
            continue
        tem_cluster = []
        lengths = []
        nbrs = NearestNeighbors(radius=epsilon, algorithm='ball_tree').fit(points_in_bbox)
        adjacency_matrix = nbrs.radius_neighbors_graph(points_in_bbox).toarray()
        num_labels, labels = connected_components(csgraph=adjacency_matrix, directed=False)

        for i in range(1, num_labels + 1):
            cluster = points_in_bbox[labels == i]
            tem_cluster.append(cluster)
        lengths = [len(tem) for tem in tem_cluster]
        max_length = max(lengths)
        result = tem_cluster[lengths.index(max_length)]
        clusters.append(result)
    return clusters

def generate_3d_boxes(bbox_2d, lidar2cam, cam2img, points):
    from sklearn.cluster import DBSCAN
    points_3d = points[:, :3]
    points_3d = remove_close(points_3d, radius=1.5)
    # 将3D点投影到2D空间
    points_3d_cam = project_from_ego_to_cam(points_3d, lidar2cam)
    # 转换为实际的2D坐标
    points_2d = project_cam_to_image(cam2img, points_3d_cam)
    mask = points_3d_cam[:, 2] > 0
    depth = points_3d_cam[:, 2][mask]
    points_3d = points_3d[mask]
    points_2d = points_2d[mask]
    # # 找到2D边界框内的点
    # mask = (points_2d[:, 0] >= bbox_2d[0]) & (points_2d[:, 0] <= bbox_2d[2]) & \
    #        (points_2d[:, 1] >= bbox_2d[1]) & (points_2d[:, 1] <= bbox_2d[3])
    # inliers = points_3d[mask]  # 位于2D框内的3D点
    # # 对这些点进行聚类以消除异常值
    # clustering = DBSCAN(eps=0.5, min_samples=5).fit(inliers)
    # labels = clustering.labels_
    # # 获取主要聚类的点，即标签不为-1的点
    # core_points = inliers[labels != -1]
    clusters = CCL(points_3d, points_2d, bbox_2d, depth)
    bbox_size = np.array([2.0, 2.0, 2.0])  # [length, width, height]
    yaw = 0.0  # 固定的yaw角度
    bboxes_3d = []
    for cluster in clusters:
        if cluster.shape[0] == 0:
            continue  # 跳过空簇
        # 计算簇的中心点
        # center = np.mean(cluster, axis=0)
        x = (np.min(cluster[:, 0]) + np.max(cluster[:, 0])) / 2
        y = (np.min(cluster[:, 1]) + np.max(cluster[:, 1])) / 2
        z = (np.min(cluster[:, 2]) + np.max(cluster[:, 2])) / 2
        center = np.array((x, y, z))
        # 构造7维边界框 (x_center, y_center, z_center, length, width, height, yaw)
        bbox_3d = np.hstack((center, bbox_size, yaw))
        bboxes_3d.append(bbox_3d)
    return bboxes_3d
def main():
    info_path = '/home/work/Downloads/BEVFormer/data/nuscenes/nuscenes_infos_temporal_train.pkl'
    base_path = '/home/work/Downloads/BEVFormer'
    im_height, im_width = (900, 1600)
    infos = pkl.load(open(info_path, 'rb'))
    info = infos['infos'][93]
    lidar_path = '/home/work/Downloads/BEVFormer/data/nuscenes/samples/LIDAR_TOP/n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385098400887.pcd.bin'
    img_path = '/home/work/Downloads/BEVFormer/data/nuscenes/samples/CAM_FRONT/n008-2018-08-27-11-48-51-0400__CAM_FRONT__1535385098362404.jpg'
    sensor2lidar_rotation = info['cams']['CAM_FRONT']['sensor2lidar_rotation']  # 3*3
    sensor2lidar_translation = info['cams']['CAM_FRONT']['sensor2lidar_translation']  # 3,
    cam_intrinsic = info['cams']['CAM_FRONT']['cam_intrinsic']  # 3*3
    boxes_3d = []
    for i in range(len(info['gt_names'])):
        if info['gt_names'][i] == 'car':
            boxes_3d.append(info['gt_boxes'][i])
    print(len(boxes_3d))
    cam2lidar_4 = np.eye(4)
    cam2lidar = np.concatenate((sensor2lidar_rotation, sensor2lidar_translation.reshape(3, 1)), axis=1)
    cam2lidar_4[:3, :] = cam2lidar
    lidar2cam = np.linalg.inv(cam2lidar_4)
    cam2img = np.eye(4)
    cam2img[:3, :3] = cam_intrinsic
    lidar2img = cam2img @ lidar2cam

    input_meta = {'lidar2img': lidar2img}
    gt_bboxes_3d = np.array(boxes_3d, dtype=np.float32)
    gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d, origin=(0.5, 0.5, 0.5))
    corners = gt_bboxes_3d.corners.numpy()
    bboxs = []
    for corner in corners:
        uv_cam = project_from_ego_to_cam(corner, lidar2cam)
        uv = project_cam_to_image(cam2img, uv_cam)
        bbox = list(chain(np.min(uv, axis=0).tolist()[0:2], np.max(uv, axis=0).tolist()[0:2]))
        # inside = (0 <= bbox[1] < im_height and 0 < bbox[3] <= im_height) and (
        #         0 <= bbox[0] < im_width and 0 < bbox[2] <= im_width) and np.min(uv_cam[:, 2], axis=0) > 0
        valid = (0 <= bbox[1] < im_height or 0 < bbox[3] <= im_height) and (
                0 <= bbox[0] < im_width or 0 < bbox[2] <= im_width) and np.min(uv_cam[:, 2], axis=0) > 0
        if valid:
            bboxs.append(bbox)

    vis_3d_on_image = False
    vis_3d_on_point = True
    if vis_3d_on_image:
        visualizer = Det3DLocalVisualizer()
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        visualizer.set_image(img)
        # project 3D bboxes to image
        visualizer.draw_proj_bboxes_3d(gt_bboxes_3d, input_meta)
        visualizer.show()
        visualizer.show()
    elif vis_3d_on_point:
        points = np.fromfile(lidar_path, dtype=np.float32)
        points = points.reshape(-1, 5)
        points = points[:, :4]
        # box_3d_ps = []
        # for bbox in bboxs:
        #     box_3d = generate_3d_boxes(bbox, lidar2cam, cam2img, points)
        #     box_3d_ps.append(box_3d)
        box_3d_ps = generate_3d_boxes(bboxs, lidar2cam, cam2img, points)
        ps_bboxes_3d = np.array(box_3d_ps, dtype=np.float32)
        ps_bboxes_3d = LiDARInstance3DBoxes(ps_bboxes_3d, origin=(0.5, 0.5, 0.5))
        bboxes_3d_low = np.zeros((1, 7))
        bboxes_3d_low = LiDARInstance3DBoxes(bboxes_3d_low, origin=(0.5, 0.5, 0.5))
        visualizer = Det3DLocalVisualizer()
        # set point cloud in visualizer
        visualizer.set_points(points)
        visualizer.draw_bboxes_3d(ps_bboxes_3d, bboxes_3d_low, gt_bboxes_3d)
        visualizer.show(save_path='/home/work/Downloads/generate_3dbox_depth.png')
    else:
        import cv2
        img = cv2.imread(img_path)
        # 将最小和最大值转换为整数
        color = (0, 255, 0)  # 绿色
        thickness = 1  # 边框线条粗细

        # 遍历并处理每个边界框
        for box in bboxs:
            x_min, y_min, x_max, y_max = box

            # 裁剪边界框的坐标
            x_min = int(max(0, x_min))
            y_min = int(max(0, y_min))
            x_max = int(min(im_width - 1, x_max))
            y_max = int(min(im_height - 1, y_max))
            # 仅在有效的情况下绘制边界框
            if x_min < x_max and y_min < y_max:
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

        # 显示结果
        cv2.imshow('Image with Boxes', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 或者保存图像
        cv2.imwrite('/home/work/Downloads/output.jpg', img)

if __name__ == '__main__':
    main()


