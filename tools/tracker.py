# author  : jerrylee
# date    : 2022/2/15
# time    : 18:47
# encoding: utf-8
import random
import cv2
import torch
import numpy as np

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from utils_yolo.kalmanfilter import KalmanFilter

cfg = get_config()
cfg.merge_from_file(r'\deep_sort\configs\deep_sort.yaml')

"""
DeepSort 函数配置与初始化

ToDo
"""
kf = KalmanFilter()
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)

cx = None
cy = None


def draw_bboxes(image, bboxes, line_thickness, frame_sum):
    """
    绘制边框
    :param frame_sum: 当前帧数
    :rtype: object
    :param image: 传入图片
    :param bboxes: 位置，参数信息
    :param line_thickness: 线宽
    :return: 标记后图片
    """
    line_thickness = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) * 0.5) + 1

    list_pts = []
    point_radius = 4
    for (x1, y1, x2, y2, cls_id, pos_id, dict_box) in bboxes:

        # 轨迹点追踪
        if frame_sum > 0:
            if len(dict_box[pos_id]) < 20:
                for i in range(1, len(dict_box[pos_id])):
                    if dict_box[pos_id][i - 1] is None or dict_box[pos_id][i] is None:
                        continue
                    # cv2.line(image, tuple(map(int, dict_box[pos_id][i - 1])), tuple(map(int, dict_box[pos_id][i])),
                    #          COLORS_10[pos_id % len(COLORS_10)], 2)
                    cv2.circle(image, tuple(map(int, dict_box[pos_id][i])), 2, COLORS_10[pos_id % len(COLORS_10)], -1)
            else:
                for i in range(len(dict_box[pos_id]) - 20, len(dict_box[pos_id])):
                    if dict_box[pos_id][i - 1] is None or dict_box[pos_id][i] is None:
                        continue
                    # cv2.line(image, tuple(map(int, dict_box[pos_id][i - 1])), tuple(map(int, dict_box[pos_id][i])),
                    #          COLORS_10[pos_id % len(COLORS_10)], 2)
                    cv2.circle(image, tuple(map(int, dict_box[pos_id][i])), 2, COLORS_10[pos_id % len(COLORS_10)], -1)

        # 轨迹点预测
        # if frame_sum > 5:
        #     for i in range(5):
        #         predicted = kf.predict(tuple(map(int, dict_box[pos_id][-1]))[0],
        #                                tuple(map(int, dict_box[pos_id][-1]))[1])
        #         cv2.circle(image, (predicted[0], predicted[1]), 4, (0, 255, 0), -1)

        # bbox_img = im0[y1:y2, x1:x2]
        # 撞线的点
        color = COLORS_10[pos_id % len(COLORS_10)]
        check_point_x = x1
        check_point_y = int(y1 + ((y2 - y1) * 0.6))

        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

        font_thickness = max(line_thickness - 1, 1)
        t_size = cv2.getTextSize(cls_id, 0, fontScale=line_thickness / 3, thickness=font_thickness)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, line_thickness / 3,
                    [225, 255, 255], thickness=font_thickness, lineType=cv2.LINE_AA)

        # 关键点绘制
        list_pts.append([check_point_x - point_radius, check_point_y - point_radius])
        list_pts.append([check_point_x - point_radius, check_point_y + point_radius])
        list_pts.append([check_point_x + point_radius, check_point_y + point_radius])
        list_pts.append([check_point_x + point_radius, check_point_y - point_radius])

        ndarray_pts = np.array(list_pts, np.int32)

        cv2.fillPoly(image, [ndarray_pts], color=(0, 255, 255))

        list_pts.clear()

    return image


def update(bboxes, image, dict_box):
    bbox_xywh = []
    confs = []
    bboxes2draw = []
    if len(bboxes) > 0:
        for x1, y1, x2, y2, lbl, conf in bboxes:
            obj = [
                int((x1 + x2) * 0.5), int((y1 + y2) * 0.5),
                x2 - x1, y2 - y1
            ]
            bbox_xywh.append(obj)
            confs.append(conf)

        xywhs = torch.Tensor(bbox_xywh)
        confss = torch.Tensor(confs)

        outputs = deepsort.update(xywhs, confss, image)
        for x1, y1, x2, y2, track_id in list(outputs):
            center_x = (x1 + x2) * 0.5
            center_y = (y1 + y2) * 0.5

            center = (center_x, center_y)
            label = search_label(center_x=center_x, center_y=center_y,
                                 bboxes_xyxy=bboxes, max_dist_threshold=20.0)
            dict_box.setdefault(track_id, []).append(center)

            bboxes2draw.append((x1, y1, x2, y2, label, track_id, dict_box))
        pass
    pass

    return bboxes2draw


def search_label(center_x, center_y, bboxes_xyxy, max_dist_threshold):
    """
    在 yolov5 的 bbox 中搜索中心点最接近的label
    :param center_x:
    :param center_y:
    :param bboxes_xyxy:
    :param max_dist_threshold:
    :return: 字符串
    """
    label = ''
    # min_label = ''
    min_dist = -1.0

    for x1, y1, x2, y2, lbl, conf in bboxes_xyxy:
        center_x2 = (x1 + x2) * 0.5
        center_y2 = (y1 + y2) * 0.5

        # 横纵距离都小于 max_dist
        min_x = abs(center_x2 - center_x)
        min_y = abs(center_y2 - center_y)

        if min_x < max_dist_threshold and min_y < max_dist_threshold:
            # 距离阈值，判断是否在允许误差范围内
            # 取 x, y 方向上的距离平均值
            avg_dist = (min_x + min_y) * 0.5
            if min_dist == -1.0:
                # 第一次赋值
                min_dist = avg_dist
                # 赋值label
                label = lbl
                pass
            else:
                # 若不是第一次，则距离小的优先
                if avg_dist < min_dist:
                    min_dist = avg_dist
                    # label
                    label = lbl
                pass
            pass
        pass

    return label


def xyxy2tlwh(x):
    """
    (top left x, top left y,width, height)
    """
    y = torch.zeros_like(x) if isinstance(x,
                                          torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1]
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


# 颜色定义
COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]
