# author  : jerrylee
# date    : 2022/2/15
# time    : 18:42
# encoding: utf-8
import json
import os

import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils_yolo.datasets import letterbox
from utils_yolo.general import non_max_suppression, scale_coords
from utils_yolo.torch_utils import select_device


class Detector:

    def __init__(self):
        """
        初始化相关配置及其权重
        """
        base_path = os.path.dirname(os.getcwd())
        config_path = base_path + r'\web_final\src\templates\detector_config.json'
        with open(config_path, 'rb') as _config:
            config = json.load(_config)
        self.img_size = config['img_size']  # 大小
        self.threshold = config['threshold']  # 置信度
        self.stride = config['stride']  # 步长
        self.weights = "yolo_weight/" + config['weight']  # 权重文件
        self.device = config['device'] if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.float()

        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB
        img = np.ascontiguousarray(img)  # 地址连续化
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def detect(self, im):

        im0, img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.4)
        boxes = []
        for det in pred:

            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if lbl not in ['person', 'car', 'bus', 'truck']:
                        # if lbl not in ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']:
                        continue
                    pass
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return boxes


def shot_accident(img, bboxes):
    base_path = os.path.dirname(os.getcwd())
    save_path = base_path + r'\web_final\src\static\save_accident'
    if len(bboxes) > 0:
        for x1, y1, x2, y2, _, _id, _ in bboxes:
            c1, c2 = (x1, y1), (x2, y2)
            crop = img[c1[1]:c2[1] - c2[0], c2[0]:c2[1] - c1[0]]
            cv2.imwrite(save_path + r'\{}.jpg'.format(_id), crop)


def shot_img(img, c1, c2, track_id):
    if not img.any():
        pass
    else:
        base_path = os.path.dirname(os.getcwd())
        save_path = base_path + r'\web_final\src\static\save_img'
        crop = img[c1[1]-15:c2[1]+1, c1[0]-10:c2[0]+20]
        if crop.any():
            path = save_path + r'\{}.jpg'.format(track_id)
            print(path)
            cv2.imwrite(path, crop)


class Detector_sg:

    def __init__(self):
        """
        初始化相关配置及其权重
        """
        base_path = os.path.dirname(os.getcwd())
        config_path = base_path + r'\web_final\src\templates\detector_sg_config.json'
        with open(config_path, 'rb') as _config:
            config = json.load(_config)
        self.img_size = config['img_size']  # 大小
        self.threshold = config['threshold']  # 置信度
        self.stride = config['stride']  # 步长
        self.weights = "yolo_weight/" + config['weight']  # 权重文件
        self.device = config['device'] if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.float()

        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB
        img = np.ascontiguousarray(img)  # 地址连续化
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def detect(self, im):

        im0, img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.4)
        boxes = []
        for det in pred:

            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if lbl not in ['sg']:
                        continue
                    pass
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    boxes.append(
                        (x1, y1, x2, y2, 'accident', conf))

        return boxes
