import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lg


test = [(652.0, 148.5), (652.5, 148.5), (652.5, 149.0), (653.0, 150.0), (652.5, 150.0), (700.0, 127.0), (700.0, 126.0),
        (699.5, 126.5), (698.5, 127.0), (697.0, 127.5), (696.0, 128.5), (695.5, 129.0), (696.0, 129.0), (695.0, 129.5),
        (694.5, 130.0)]
list_x = []
list_y = []
for i in test:
    _x = i[0]
    _y = i[1]
    list_x.append(_x)
    list_y.append(_y)
    print(_y, _x)

_x = np.array(list_x)
_y = np.array(list_y)
t = np.arange(1, 17, 1)
y = np.array([4, 6.4, 8, 8.8, 9.22, 9.5, 9.7, 9.86, 10, 10.20, 10.32, 10.42, 10.5, 10.55, 10.58, 10.6])
plt.figure()
plt.plot(_x, _y, 'k*')
# y=at^2+bt+c

A = np.c_[_x ** 2, _x, np.ones(_x.shape)]

w = lg.inv(A.T.dot(A)).dot(A.T).dot(_y)

plt.plot(_x, w[0] * _x ** 2 + w[1] * _x + w[2])
plt.show()
