import cv2

im = cv2.imread('2.png')

img = cv2.resize(im, (590, 410))

cv2.imwrite("5.jpg", img)

cap = cv2.VideoCapture("C://Users//bz//Desktop//3.mp4")

while True:
    _, img = cap.read()
    print(img.shape)
