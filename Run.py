# author  : jerrylee
# date    : 2022/2/15
# time    : 18:42
# encoding: utf-8
import json
import os
from src.web_main import Run

if __name__ == '__main__':

    # 图片保存路径
    base_path = r'src/static/save_img'
    base_path_accident = r'src/static/save_accident'
    # 对保存的照片进行重置
    for filename in os.listdir(r'src/static/save_img'):
        if filename.endswith('jpg'):
            os.remove(base_path + '/' + filename)
    # 对保存的事故图片进行重置
    for filename_sg in os.listdir(r'src/static/save_accident'):
        if filename_sg.endswith('jpg'):
            os.remove(base_path_accident + '/' + filename_sg)
    # 重复运行前,先对已有json表进行清空
    open('./src/templates/json_dict.json', 'w').close()
    open('./src/templates/json_dict_style.json', 'w').close()
    open('./src/templates/json_dict_label.json', 'w').close()
    open('./src/templates/json_dict_sum.json', 'w').close()

    # 打印配置信息
    with open('./src/templates/detector_config.json', 'rb') as show_config:
        config = json.load(show_config)
    print('\n[INFO]: Config information are {}'.format(config))
    with open('./src/templates/detector_sg_config.json', 'rb') as show_sg_config:
        config_sg = json.load(show_sg_config)
    print('\n[INFO]: Config accident information are {}\n'.format(config_sg))

    Run()
