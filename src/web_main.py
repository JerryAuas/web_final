# author  : jerrylee
# data    : 2022/1/14
# time    : 20:30
# encoding: utf-8
import fileinput
import os

import numpy as np
from flask import Response, request
from flask import Flask
from flask import render_template
from tools import tracker
from tools.detector import Detector, Detector_sg, shot_accident, shot_img
import time
import torch
import json
import cv2

# 定义flask应用app入口

app = Flask(__name__)

# 视频流地址
camera = None
file_streams = None

# 默认视频播放路径
video_path = "C://Users//bz//Desktop//1.mp4"

# 撞线点
list_pts_left_1 = []
list_pts_up_1 = []
list_pts_right_1 = []
list_pts_down_1 = []
# 期待得到视频尺寸 (宽高
w = 0
h = 0


@app.route('/')
def web_shou():
    return render_template('start_v.html')


@app.route('/start_v.html')
def start():
    return render_template('start_v.html')


@app.route('/Web-shou.html')
def web_shou_():
    return render_template('Web-shou.html')


@app.route('/web_config.html')
def config():
    return render_template('web_config.html')


@app.route('/web.html', methods=['GET', 'POST'])
def web():
    return render_template('web.html')


@app.route('/photo_1.html')
def photo_1():
    return render_template('photo_1.html')


@app.route('/photo_2.html')
def photo_2():
    return render_template('photo_2.html')


@app.route('/web-analyse.html')
def web_analyse():
    return render_template('web-analyse.html')


@app.route('/web_get', methods=['GET'])
def web_get():
    text_1 = request.args.getlist('text')
    text = []
    flags = 0
    # 默认画布尺寸 1152 x 648 -> 1920 x 1080
    for i in text_1:
        if flags % 2 == 0:
            text.append(int(float(i) * 3.7))
        else:
            text.append(int(float(i) * 12.8))
        flags += 1
    global list_pts_left_1
    list_pts_left_1 = [[text[0], text[1]], [text[2], text[3]], [text[4], text[5]], [text[6], text[7]]]
    global list_pts_up_1
    list_pts_up_1 = [[text[8], text[9]], [text[10], text[11]], [text[12], text[13]], [text[14], text[15]]]
    global list_pts_right_1
    list_pts_right_1 = [[text[16], text[17]], [text[18], text[19]], [text[20], text[21]], [text[22], text[23]]]
    global list_pts_down_1
    list_pts_down_1 = [[text[24], text[25]], [text[26], text[27]], [text[28], text[29]], [text[30], text[31]]]
    print(text_1)
    print(text)
    return render_template('web_use.html')


@app.route('/web_get_file', methods=['GET'])
def web_get_file():
    global camera
    camera = request.args.get('btn')
    # global file_streams
    # file_streams = request.args.get('file_name')
    cap = cv2.VideoCapture('C://Users//bz//Desktop//1.mp4')
    _, frame = cap.read()
    if _:
        cv2.imwrite('./src/static/photo/1.jpg', frame)
    return render_template('web.html')


@app.route('/web_use1.html')
def web_use():
    return render_template('web_use1.html')


@app.route("/json_dict_style.json", methods=['GET'])
def get_json_1():
    return render_template('json_dict_style.json')


@app.route("/json_dict.json", methods=['GET'])
def get_json_2():
    return render_template('json_dict.json')


@app.route("/json_dict_sum.json", methods=['GET'])
def get_json_3():
    return render_template('json_dict_sum.json')


@app.route("/json_dict_label.json", methods=['GET'])
def get_json_5():
    return render_template('json_dict_label.json')


@app.route("/detector_config.json", methods=['GET'])
def get_json_4():
    return render_template('detector_config.json')


@app.route("/get_config", methods=['GET'])
def get_config():
    img_size = request.args.get('img_size')
    threshold = request.args.get('threshold')
    stride = request.args.get('stride')
    weight = request.args.get('weight')
    device = request.args.get('device')
    # todo
    t = {'msg': '修改成功'}
    return json.dumps(t, ensure_ascii=False)


def detect_gen():
    global camera, file_streams, h, w, video_path
    video_path = "C://Users//bz//Desktop//1.mp4"
    if not camera:
        video_path = 0
    elif file_streams:
        file_list = []
        for line in fileinput.input('videopath_activate.txt'):
            file_list.append(line)
        video_path = file_list[int(file_streams)]
    try:
        cap = cv2.VideoCapture(video_path)
        _, img = cap.read()
        size_img = img.shape
        w = size_img[1]
        h = size_img[0]
    except ValueError:
        print('The video or camera path is error!')

    global list_pts_left_1
    global list_pts_up_1
    global list_pts_right_1
    global list_pts_down_1

    # 原视频大小, (宽高
    img_size_start_h = int(h)
    img_size_start_w = int(w)
    # 变换后进入处理视频的尺寸大小 (宽/2, 高/2
    img_size_w = int(img_size_start_w / 2)
    img_size_h = int(img_size_start_h / 2)

    # 定位框位置，判断下一步方向
    # TODO
    # list_pts_left = [[480, 583], [140, 790], [165, 850], [570, 600]]
    # list_pts_up = [[690, 530], [900, 550], [870, 610], [650, 590]]
    # list_pts_right = [[930, 950], [735, 1920], [830, 1920], [1000, 970]]
    # list_pts_down = [[10, 1200], [50, 1150], [630, 1920], [540, 1920]]
    list_pts_left = list_pts_left_1
    list_pts_up = list_pts_up_1
    list_pts_right = list_pts_right_1
    list_pts_down = list_pts_down_1

    # 初始化4个撞线polygon
    # 根据视频尺寸，填充一个polygon，供撞线计算使用
    mask_image_temp = np.zeros((img_size_start_w, img_size_start_h), dtype=np.uint8)
    ndarray_pts_left = np.array(list_pts_left, np.int32)
    polygon_value_left = cv2.fillPoly(mask_image_temp, [ndarray_pts_left], color=1)
    polygon_value_left = polygon_value_left[:, :, np.newaxis]

    # 填充第二个polygon
    mask_image_temp = np.zeros((img_size_start_w, img_size_start_h), dtype=np.uint8)
    ndarray_pts_up = np.array(list_pts_up, np.int32)
    polygon_value_up = cv2.fillPoly(mask_image_temp, [ndarray_pts_up], color=2)
    polygon_value_up = polygon_value_up[:, :, np.newaxis]

    # 填充第三个polygon
    mask_image_temp = np.zeros((img_size_start_w, img_size_start_h), dtype=np.uint8)
    ndarray_pts_right = np.array(list_pts_right, np.int32)
    polygon_value_right = cv2.fillPoly(mask_image_temp, [ndarray_pts_right], color=3)
    polygon_value_right = polygon_value_right[:, :, np.newaxis]

    # 填充第四个polygon
    mask_image_temp = np.zeros((img_size_start_w, img_size_start_h), dtype=np.uint8)
    ndarray_pts_down = np.array(list_pts_down, np.int32)
    polygon_value_down = cv2.fillPoly(mask_image_temp, [ndarray_pts_down], color=4)
    polygon_value_down = polygon_value_down[:, :, np.newaxis]

    # 撞线检测用mask，包含4个polygon，（值范围 0 1 2 3 4），供撞线计算使用
    polygon_mask_left_up_right_down = polygon_value_left + polygon_value_up + polygon_value_right + polygon_value_down

    # 缩小尺寸
    polygon_mask_left_up_right_down = cv2.resize(polygon_mask_left_up_right_down, (img_size_w, img_size_h))

    # 左 色盘 与 polygon图片
    left_color_plate = [255, 0, 0]
    left_image = np.array(polygon_value_left * left_color_plate, np.uint8)

    # 上 色盘 与 polygon图片
    up_color_plate = [0, 255, 255]
    up_image = np.array(polygon_value_up * up_color_plate, np.uint8)

    # 右 色盘 与 polygon图片
    right_color_plate = [0, 255, 0]
    right_image = np.array(polygon_value_right * right_color_plate, np.uint8)

    # 下 色盘 与 polygon图片
    down_color_plate = [255, 0, 255]
    down_image = np.array(polygon_value_down * down_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = left_image + up_image + right_image + down_image

    # 缩小尺寸
    color_polygons_image = cv2.resize(color_polygons_image, (img_size_w, img_size_h))

    # list 与左侧polygon重叠
    list_overlapping_left_polygon = []

    # list 与上侧polygon重叠
    list_overlapping_up_polygon = []

    # list 与右侧polygon重叠
    list_overlapping_right_polygon = []

    # list 与下侧polygon重叠
    list_overlapping_down_polygon = []

    # 从左到右
    left2right_count = 0
    # 从左到上
    left2up_count = 0
    # 从左到下
    left2down_count = 0

    # 从上到左
    up2left_count = 0
    # 从上到右
    up2right_count = 0
    # 从上到下
    up2down_count = 0

    # 从右到左
    right2left_count = 0
    # 从右到上
    right2up_count = 0
    # 从右到下
    right2down_count = 0

    # 从下到左
    down2left_count = 0
    # 从下到上
    down2up_count = 0
    # 从下到右
    down2right_count = 0

    # 初始化 yolov5
    detector = Detector()
    # 初始化第二层网络（事故检测层）
    detector_sg = Detector_sg()

    # 打开视频
    capture = cv2.VideoCapture(video_path)

    # 设备使用情况
    if torch.cuda.is_available():
        print('当前使用的设备为 {}'.format(torch.cuda.get_device_name()))
    else:
        print('当前使用的设备为 {}'.format('CPU'))

    # 初始化存放点位信息的字典
    dict_box = dict()
    # json转换数据存储字典
    json_dict = dict()
    # json转换数据状态字典
    json_dict_style = dict()
    # json转换数据label与id字典
    json_dict_label = dict()
    # json转换数据经流总数字典
    json_dict_sum = dict()
    # 用于转换的信息数组,[] to dict()
    list_dic = []
    # 初始化当前帧总数
    frame_sum = 0
    #
    list_sum = []

    while True:
        # 读取每帧图片
        _, im = capture.read()
        if im is None:
            print('The path or video/image is None')
            break

        # 初始化person与vehicle总数量
        # 定义为局部变量,每一次循环进行一次更新
        sum_person = 0
        sum_vehicle = 0
        # 计算实时帧率开始时间
        start_time = time.time()

        # 缩小尺寸
        im = cv2.resize(im, (img_size_w, img_size_h))

        # 视频检测
        list_bboxs = []
        bboxes = detector.detect(im)
        # 车祸检测
        list_bboxs_sg = []
        bboxes_sg = detector_sg.detect(im)

        # 如果画面中 有bbox
        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im, dict_box)
            # 画框
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=1, frame_sum=frame_sum)
            frame_sum += 1
            # 如果画面中有事故bbox
            if len(bboxes_sg) > 0:
                # list_bboxs_sg = tracker.draw_bboxes(im, list_bboxs_sg, line_thickness=3, frame_sum=frame_sum)
                # 标记事故框
                tracker.draw_bboxes(output_image_frame, list_bboxs_sg, line_thickness=3, frame_sum=frame_sum)
                # 保存事故图片
                shot_accident(output_image_frame, list_bboxs_sg)
            pass
        else:
            # 如果画面中 没有bbox
            output_image_frame = im
        pass

        # 输出图片(带有标注线)
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)

        if len(list_bboxs) > 0:
            # 实时交通状态显示
            if len(list_bboxs) > 20:
                style = '拥堵'
            else:
                style = '正常'
            json_dict['style'] = style
            json_dict['len_list'] = str(len(list_bboxs))
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id, _ = item_bbox

                # 保存点（对角点）
                c1, c2 = (x1, y1), (x2, y2)
                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0， x方向偏移量 0.0~1.0
                y1_offset = int(y1 + ((y2 - y1) * 0.6))
                x1_offset = int(x1 + ((x2 - x1) * 0.6))

                # 撞线的点
                y = y1_offset
                x = x1_offset
                # 统计实时流量
                for key in json_dict_label:
                    if key != track_id:
                        json_dict_label[track_id] = label
                list_sum.append(label)
                # 将流量总数存入json
                for label_sum in list_sum:
                    if label_sum in ['car', 'bus', 'truck']:
                        sum_vehicle += 1
                    if label_sum == 'person':
                        sum_person += 1
                list_sum.clear()
                json_dict_sum['Sum_person'] = sum_person
                json_dict_sum['Sum_vehicle'] = sum_vehicle
                # 如果撞 左polygon(左上右下)
                if polygon_mask_left_up_right_down[y, x] == 1:
                    if track_id not in list_overlapping_left_polygon:
                        list_overlapping_left_polygon.append(track_id)
                    pass

                    # 判断 上polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 由上到左方向
                    if track_id in list_overlapping_up_polygon:
                        # 上到左+1
                        up2left_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除 上polygon list 中的此id
                        list_overlapping_up_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        shot_img(im, c1, c2, track_id)
                        pass
                    # 判断 右polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 由右到左方向
                    elif track_id in list_overlapping_right_polygon:
                        # 右到左+1
                        right2left_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除 右polygon list 中的此id
                        list_overlapping_right_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        shot_img(im, c1, c2, track_id)
                        pass
                    # 判断 下polygon list 是否有此 track_id
                    # 有此 track_id, 则认为是 由下到左方向
                    elif track_id in list_overlapping_down_polygon:
                        # 下到左+1
                        down2left_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除 下polygon list 中的此id
                        list_overlapping_down_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        shot_img(im, c1, c2, track_id)
                        pass
                    else:
                        # 没有此id
                        pass

                elif polygon_mask_left_up_right_down[y, x] == 2:
                    # 如果撞 上polygon
                    if track_id not in list_overlapping_up_polygon:
                        list_overlapping_up_polygon.append(track_id)
                    pass

                    # 判断 左polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 左到上方向
                    if track_id in list_overlapping_left_polygon:
                        # 左到上+1
                        left2up_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除 左polygon list 中的此id
                        list_overlapping_left_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        shot_img(im, c1, c2, track_id)

                        pass
                    elif track_id in list_overlapping_right_polygon:
                        # 右到上+1
                        right2up_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除 右polygon list 中的此id
                        list_overlapping_right_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        shot_img(im, c1, c2, track_id)
                        pass

                    elif track_id in list_overlapping_down_polygon:
                        # 下到上+1
                        down2up_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除 下polygon list 中的此id
                        list_overlapping_down_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        shot_img(im, c1, c2, track_id)
                        pass

                    else:
                        # 无此 track_id，不做其他操作
                        pass
                    pass
                elif polygon_mask_left_up_right_down[y, x] == 3:
                    # 如果撞 右polygon
                    if track_id not in list_overlapping_right_polygon:
                        list_overlapping_right_polygon.append(track_id)
                    pass

                    # 判断 左polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 左到右方向
                    if track_id in list_overlapping_left_polygon:
                        # 左到右+1
                        left2right_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除 左polygon list 中的此id
                        list_overlapping_left_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        shot_img(im, c1, c2, track_id)

                        pass
                    # 如果在上中
                    elif track_id in list_overlapping_up_polygon:
                        # 上到右+1
                        up2right_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除上polygon中此id
                        list_overlapping_up_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        shot_img(im, c1, c2, track_id)
                        pass
                    # 如果在下中
                    elif track_id in list_overlapping_down_polygon:
                        # 下到右+=1
                        down2right_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除下polygon中此id
                        list_overlapping_down_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        shot_img(im, c1, c2, track_id)

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass
                    pass
                elif polygon_mask_left_up_right_down[y, x] == 4:
                    # 如果撞 下polygon
                    if track_id not in list_overlapping_down_polygon:
                        list_overlapping_down_polygon.append(track_id)
                    pass

                    # 判断 左polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 左到下方向
                    if track_id in list_overlapping_left_polygon:
                        # 左到右+1
                        left2down_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除 左polygon list 中的此id
                        list_overlapping_left_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        shot_img(im, c1, c2, track_id)

                        pass
                    # 上
                    elif track_id in list_overlapping_up_polygon:
                        # 上到下+=1
                        up2down_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除上中此id
                        list_overlapping_up_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        shot_img(im, c1, c2, track_id)
                        pass

                    # 右
                    elif track_id in list_overlapping_right_polygon:
                        # 右到下+=1
                        right2down_count += 1
                        # 空的字典,用于数据转移
                        dict_temp = dict()
                        if {"id": str(track_id), "label": label} not in list_dic:
                            dict_temp['id'] = str(track_id)
                            dict_temp['label'] = label
                            list_dic.append(dict_temp)

                        # 删除右中此 id
                        list_overlapping_right_polygon.remove(track_id)

                        # 保存通过界面最后压线图片
                        shot_img(im, c1, c2, track_id)
                        pass

                else:
                    pass
                pass

            pass

            # ----------------------清除无用id----------------------
            list_overlapping_all = list_overlapping_left_polygon + list_overlapping_up_polygon \
                                   + list_overlapping_right_polygon + list_overlapping_down_polygon
            for id1 in list_overlapping_all:
                is_found = False
                for _, _, _, _, _, bbox_id, _ in list_bboxs:
                    if bbox_id == id1:
                        is_found = True
                        break
                    pass
                pass

                if not is_found:
                    # 如果没找到，删除id
                    if id1 in list_overlapping_left_polygon:
                        list_overlapping_left_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_up_polygon:
                        list_overlapping_up_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_right_polygon:
                        list_overlapping_right_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_down_polygon:
                        list_overlapping_down_polygon.remove(id1)
                    pass
                pass
            # 清空list
            list_overlapping_all.clear()
            pass

            list_bboxs.clear()

            pass
        else:
            # 如果图像中没有任何的bbox，则清空list
            list_overlapping_left_polygon.clear()
            list_overlapping_up_polygon.clear()
            list_overlapping_right_polygon.clear()
            list_overlapping_down_polygon.clear()
            pass
        pass

        json_dict_style.setdefault('UP_LEFT', str(up2left_count))
        json_dict_style.setdefault('UP_RIGHT', str(up2right_count))
        json_dict_style.setdefault('UP_DOWN', str(up2down_count))
        json_dict_style.setdefault('LEFT_RIGHT', str(left2right_count))
        json_dict_style.setdefault('LEFT_UP', str(left2up_count))
        json_dict_style.setdefault('LEFT_DOWN', str(left2down_count))
        json_dict_style.setdefault('RIGHT_UP', str(right2up_count))
        json_dict_style.setdefault('RIGHT_LEFT', str(right2left_count))
        json_dict_style.setdefault('RIGHT_DOWN', str(right2down_count))
        json_dict_style.setdefault('DOWN_LEFT', str(down2left_count))
        json_dict_style.setdefault('DOWN_UP', str(down2up_count))
        json_dict_style.setdefault('DOWN_RIGHT', str(down2right_count))

        # 计算实时帧率结束时间
        end_time = time.time()
        # 计算fps并打印
        seconds = end_time - start_time
        fps = 1 / seconds
        fps = '%.2f' % fps
        json_dict_style.setdefault('FPS', fps)

        # # 判断label文件内容是否唯一
        # length = len(list_dic)
        # for i in range(length):
        #     for j in range(i):
        #         if list_dic[i] == list_dic[j]:
        #             list_dic.remove(list_dic[i])
        # print(list_dic)
        # 判断存入字典的条件以及将各个字典转换为json文件
        base_filename = os.getcwd() + r'\src'
        try:
            if json_dict:
                with open(base_filename + r'\templates\json_dict.json', 'w+') as f_1:
                    json.dump(json_dict, f_1)
            with open(base_filename + r'\templates\json_dict_style.json', 'w+') as f_2:
                json.dump(json_dict_style, f_2)
            with open(base_filename + r'\templates\json_dict_label.json', 'w+') as f_3:
                json.dump(list_dic, f_3)
            with open(base_filename + r'\templates\json_dict_sum.json', 'w+') as f_4:
                json.dump(json_dict_sum, f_4)
        except IOError:
            print("文件写入失败")

        output_image_frame = cv2.resize(output_image_frame, (img_size_start_w, img_size_start_h))
        frame = output_image_frame
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

        # 清空方式以及总数相关信息的字典, 方便后面写入以及json的数据的读取
        json_dict.clear()
        json_dict_style.clear()


@app.route('/video_feed/<feed_type>')
def video_feed(feed_type):
    """Video streaming route. Put this in the src attribute of an img tag."""
    if feed_type == 'Camera_0':
        return Response(detect_gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')


def Run():
    app.run(host='0.0.0.0', port="5000", threaded=True, debug=True)
