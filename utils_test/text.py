import cv2

cap = cv2.VideoCapture('http://localhost:63342/25abae96-b0ab-4238-a23f-76b305787fff')
# cap = cv2.VideoCapture('C://Users//86185//Desktop//1.mp4')

while True:
    _, img = cap.read()

    cv2.imshow("xsad", img)
    cv2.waitKey(1)