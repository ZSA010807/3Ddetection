import numpy as np
from mmengine import  load
from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes
import torch
from functools import partial
from sklearn.cluster import KMeans
from sklearn import cluster
def beam_label(theta, beam):
    estimator=KMeans(n_clusters=beam)
    res=estimator.fit_predict(theta.reshape(-1, 1))
    label=estimator.labels_
    centroids=estimator.cluster_centers_
    return label, centroids[:,0]
def label_point_cloud_beam(self, polar_image, points, beam=32):
    if polar_image.shape[0] <= beam:
        print("too small point cloud!")
        return np.arange(polar_image.shape[0])
    beam_labels, centroids = beam_label(polar_image[:, 1], beam)
    idx = np.argsort(centroids)
    rev_idx = np.zeros_like(idx)
    for i, t in enumerate(idx):
        rev_idx[t] = i
    beam_labels = rev_idx[beam_labels]
    return beam_labels
def compute_angles(pc_np):
    tan_theta = pc_np[:, 2] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    theta = np.arctan(tan_theta)

    sin_phi = pc_np[:, 1] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    phi_ = np.arcsin(sin_phi)

    cos_phi = pc_np[:, 0] / (pc_np[:, 0]**2 + pc_np[:, 1]**2)**(0.5)
    phi = np.arccos(cos_phi)

    phi[phi_ < 0] = 2*np.pi - phi[phi_ < 0]
    phi[phi == 2*np.pi] = 0

    return theta, phi
def get_polar_image(points):
    theta, phi = compute_angles(points[:, :3])
    r = np.sqrt(np.sum(points[:, :3] ** 2, axis=1))
    polar_image = points.copy()
    polar_image[:, 0] = phi
    polar_image[:, 1] = theta
    polar_image[:, 2] = r
    return polar_image

def random_beam_upsample(points, config=None):
    if points is None:
        return partial(random_beam_upsample, config=config)

    polar_image = get_polar_image(points)
    beam_label = label_point_cloud_beam(polar_image, points, config['BEAM'])
    new_pcs = []
    phi = polar_image[:, 0]
    for i in range(config['BEAM'] - 1):
        if np.random.rand() < config['BEAM_PROB'][i]:
            cur_beam_mask = (beam_label == i)
            next_beam_mask = (beam_label == i + 1)
            delta_phi = np.abs(phi[cur_beam_mask, np.newaxis] - phi[np.newaxis, next_beam_mask])
            corr_idx = np.argmin(delta_phi, 1)
            min_delta = np.min(delta_phi, 1)
            mask = min_delta < config['PHI_THRESHOLD']
            cur_beam = polar_image[cur_beam_mask][mask]
            next_beam = polar_image[next_beam_mask][corr_idx[mask]]
            new_beam = (cur_beam + next_beam) / 2
            new_pc = new_beam.copy()
            new_pc[:, 0] = np.cos(new_beam[:, 1]) * np.cos(new_beam[:, 0]) * new_beam[:, 2]
            new_pc[:, 1] = np.cos(new_beam[:, 1]) * np.sin(new_beam[:, 0]) * new_beam[:, 2]
            new_pc[:, 2] = np.sin(new_beam[:, 1]) * new_beam[:, 2]
            new_pcs.append(new_pc)
    pcs = np.concatenate(new_pcs, 0)
    points = np.concatenate((pcs, points), 0)
    return points
def main():
    config = {
        'NAME': random_beam_upsample,
        'BEAM': 32,
        'BEAM_PROB': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        'PHI_THRESHOLD': 0.03
    }

    path = '/home/work/Downloads/BEVFormer/data/nuscenes/samples/LIDAR_TOP/n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385095949554.pcd.bin'
    points = np.fromfile(path, dtype=np.float32)
    points = points.reshape(-1, 5)
    points = points[:, :4]
    num1 = num2 = 0
    for i in range(len(points)):
        if points[i][3] == 0.0:
            num1 = num1 + 1
    print(len(points), num1)
    points = random_beam_upsample(points, config)
    for i in range(len(points)):
        if points[i][3] == 0.0:
            num2 = num2 + 1
    print(len(points), num2)
    # visualizer = Det3DLocalVisualizer()
    # visualizer.set_points(points)
    # box1 = []
    # box2 = []
    # box3 = []
    # bboxes_3d = LiDARInstance3DBoxes(
    #     torch.tensor(box1), origin=(0.5, 0.5, 0.5))
    # bboxes_3d_da = LiDARInstance3DBoxes(
    #     torch.tensor(box2), origin=(0.5, 0.5, 0.5))
    # gt_bboxes_3d = LiDARInstance3DBoxes(
    #     torch.tensor(box3), origin=(0.5, 0.5, 0.5))
    # # Draw 3D bboxe
    # visualizer.draw_bboxes_3d(bboxes_3d, bboxes_3d_da, gt_bboxes_3d)
    # visualizer.show(save_path='/home/work/Downloads/point_nusc_old')

if __name__ == '__main__':
    main()