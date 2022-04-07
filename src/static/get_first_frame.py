import statistics

import cv2
def get_first_frame():
    cap = cv2.VideoCapture('C://Users//86185//Desktop//1.mp4')
    _, frame = cap.read()
    if _:
        print(11111)
        cv2.imwrite('photo/1.jpg', frame)


if __name__ == '__main__':
    get_first_frame()
