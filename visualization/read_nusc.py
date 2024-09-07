import pickle as pkl
import mmcv
import mmengine
import numpy as np
from mmengine import load
import json
from os import path as osp
import os
def read_pkl():
    #/home/work/fsdownload/waymo_infos_val.pkl
    #/home/work/fsdownload/nuscenes_infos_val.pkl
    info_file = load('/home/work/fsdownload/nuscenes_infos_val.pkl')
    yaw_list = []
    for info in info_file['infos']:
        for box in info['gt_boxes']:
            yaw_list.append(box)
    print(len(yaw_list),yaw_list[-1])
    # for info in info_file:
    #     for instance in info['annos']['rotation_y']:
    #         yaw_list.append(instance)
    # yaw = sorted(yaw_list)
    # print(yaw[0], yaw[-1])

def read_json():
    yaw_list = []
    with open('/home/work/fsdownload/results_nusc.json', 'r') as file:
        info_file = json.load(file)
    for token, info in info_file['results'].items():
        for det in info:
            yaw_list.append(det['size'])
    print(len(yaw_list),yaw_list[-1])
    # yaw = sorted(yaw_list)
    # print(yaw[0], yaw[-1])

def pred():
    with open('/home/work/fsdownload/results_nusc.json', 'r') as file:#底部中心，
        info_file = json.load(file)
    token = list(info_file['results'].keys())
    print(token[10])
    boxes = []
    dets = info_file['results'][token[10]]
    for det in dets:
        boxes.append(np.array((det['box_3d'])))
    return boxes, token[10]
def gt_boxes(token):
    info_file = load('/home/work/fsdownload/nuscenes_infos_val.pkl')#中心为几何中心
    boxes = []
    for info in info_file['infos']:
        if info['token'] == token:
            print(info['lidar_path'])
            for box in info['gt_boxes']:
                box[2] = box[2] - box[5]/2.0
                boxes.append(np.array(box))
            break
    return boxes
def gt_boxes_car(token):
    info_file = load('/home/work/fsdownload/nuscenes_infos_val.pkl')#中心为几何中心
    boxes = []
    for info in info_file['infos']:
        if info['token'] == token:
            print(info['lidar_path'])
            for i in range(len(info['gt_names'])):
                if info['gt_names'][i] == 'car':
                    info['gt_boxes'][i][2] = info['gt_boxes'][i][2]-info['gt_boxes'][i][5]/2.0
                    boxes.append(np.array(info['gt_names'][i]))
            break
    return boxes

def filter_boxes():
    gt_info_file = load('/home/work/fsdownload/nuscenes_infos_val.pkl')  # 中心为几何中心
    with open('/home/work/fsdownload/da_nusc_results.json', 'r') as nusc:  # 几何中心，
        info_file = json.load(nusc)
    # da_waymo_results.json
    with open('/home/work/fsdownload/da_waymo_results.json', 'r') as file:  # 几何中心，
        info_file_waymo = json.load(file)
    tokens = list(info_file['results'].keys())
    # print(choose)
    results = {}
    boxes_list = []
    scores = []
    l1, l2, l3 = 0, 0, 0
    t1, t2, t3 = 0, 0, 0
    for token in tokens:
        boxes = []
        gt_boxes = []
        boxes_2 = []
        path = None
        dets = info_file['results'][token]
        dets_2 = info_file_waymo['results'][token]
        for det in dets:
            if det['detection_name'] == 'car':
                if det['detection_score'] >= 0.95:
                    boxes.append(np.array((det['box_3d'])))
                    if np.sum(np.array((det['box_3d']))[:2] ** 2) < 2500:
                        t1 += 1
        for det_2 in dets_2:
            if det_2['detection_name'] == 'car':
                if det_2['detection_score'] >= 0.95:
                    boxes_2.append(np.array((det_2['box_3d'])))
                    if np.sum(np.array((det_2['box_3d']))[:2] ** 2) < 2500:
                        t2 += 1
        for info in gt_info_file['infos']:
            if info['token'] == token:
                path = info['lidar_path']
                for i in range(len(info['gt_names'])):
                    if info['gt_names'][i] == 'car':
                        gt_boxes.append(np.array(info['gt_boxes'][i]))
                        if np.sum(np.array(info['gt_boxes'][i])[:2] ** 2) < 2500:
                            t3 += 1
        l1 += len(boxes)
        l2 += len(boxes_2)
        l3 += len(gt_boxes)
    # scores.sort()
    print("number of boxes on nusc, waymo, gt:",l1,l2,l3)
    print("filter by distance of 50m:",t1,t2,t3)
def box_pred_gt():
    gt_info_file = load('/home/work/fsdownload/nuscenes_infos_val.pkl')  # 中心为几何中心
    with open('/home/work/fsdownload/da_nusc_results.json', 'r') as nusc:#几何中心，
        info_file = json.load(nusc)
    #da_waymo_results.json
    with open('/home/work/fsdownload/da_waymo_results.json', 'r') as file:#几何中心，
        info_file_waymo = json.load(file)
    tokens = list(info_file['results'].keys())
    choose = tokens[:6000:3000]
    # print(choose)
    results={}
    boxes_list = []
    for token in choose:
        boxes = []
        gt_boxes =[]
        boxes_2 = []
        path = None
        dets = info_file['results'][token]
        dets_2 = info_file_waymo['results'][token]
        for det in dets:
            if det['detection_name'] =='car':
                boxes.append(np.array((det['box_3d'])))
        for det_2 in dets_2:
            if det_2['detection_name'] =='car':
                boxes_2.append(np.array((det_2['box_3d'])))
        for info in gt_info_file['infos']:
            if info['token'] == token:
                path = info['lidar_path']
                for i in range(len(info['gt_names'])):
                    if info['gt_names'][i] == 'car':
                        gt_boxes.append(np.array(info['gt_boxes'][i]))
        result = dict(
            lidarpath=path,
            sample_token=token,
            pred_box=boxes,
            pred_box_da=boxes_2,
            gt_box=gt_boxes
        )
        boxes_list.append(result)
        results[token] = result
    return boxes_list
if __name__ == '__main__':
    # pred_boxes, token = pred()
    # gt_box = gt_boxes_car(token)
    # # print(len(pred_boxes), pred_boxes)
    # print(len(gt_box), gt_box)
    # # read_pkl()
    # # read_json()
    # file_name = '/home/work/fsdownload/nusc_boxes.pkl'
    # boxes = box_pred_gt()
    # mmengine.dump(boxes, file_name)
    # print(boxes[1])
    # print(len(boxes))
    filter_boxes()