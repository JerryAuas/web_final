# author  : jerrylee
# data    : 2022/2/15
# time    : 18:44
# encoding: utf-8
import cv2
import numpy as np

capture = cv2.VideoCapture('D://yoloVideo//1.mp4')

while True:
    _, img = capture.read()
    if img is None:
        print('错误')
        break
    H = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    W = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(H, '  ', W)
    # img = cv2.resize(img, (960, 540))
    list_pts_left = [[450, 563],  [400, 571],  [360, 578],  [320, 584],  [280, 592],
                      [240, 597],  [200, 606],  [160, 614],  [120, 620],
                      [170, 632],  [210, 625],  [250, 617], [270, 614],
                     [310, 606],  [350, 597],  [390, 590],  [430, 580],
                     [470, 570]]
    # list_pts_up = [[70, 150], [730, 150]]
    # list_pts_right = [[850, 150], [885, 450]]
    ndarray_pts_left = np.array(list_pts_left, np.int32)
    polygon_left_value_1 = cv2.fillPoly(img, [ndarray_pts_left], color=(0, 0, 255))
    # ndarray_pts_up = np.array(list_pts_up, np.int32)
    # polygon_up_value_1 = cv2.fillPoly(img, [ndarray_pts_up], color=(255, 0, 255))
    # ndarray_pts_right = np.array(list_pts_right, np.int32)
    # polygon_right_value_1 = cv2.fillPoly(img, [ndarray_pts_right], color=(255, 255, 0))
    cv2.imshow('asd', img)
    if cv2.waitKey(30) == 'q':
        break
capture.release()
cv2.destroyAllWindows()
