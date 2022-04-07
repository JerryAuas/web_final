# import json
#
# data = {
#     'img_size': 640,
#     'threshold': 0.25,
#     'stride': 1,
#     'weight': 'C:\\Users\\bz\\Desktop\\web_final\\traffic.pt',
#     'device': '0'
# }
#
# with open('../src/templates/detector_config.json', 'w') as f:
#     json.dump(data, f)
#
# with open('../src/templates/detector_config.json', 'rb') as f_1:
#     dic = json.load(f_1)
#
# print(dic)
# print(type(dic))
# print(dic['img_size'])
# import os
#
#
# path_config = os.path.abspath('detector_config.json')
# print(path_config)
# c = os.path.dirname(os.path.dirname(os.getcwd()))
# print(c)
#
# a = '1'
#
# if a == '1':
#     print(1)
# if a == 0:
#     print(0)

list_ = []

a = {'name': 'lijie', 'age': 18}
di = dict()
di['name'] = 'lijie'
di['age'] = 18

list_.append(di)
di.clear()

di['name'] = 'wa'
di['age'] = 1
list_.append(di)
di.clear()

di['name'] = 'lijei'
di['age'] = 18
list_.append(di)

list_s = list_.copy()
print(i for i in list_ if i in list_s)
