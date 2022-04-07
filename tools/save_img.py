import os
import cv2

path = "../output"
path3 = "../../web_final"
w = 1920
h = 1080
img_total = []
txt_total = []
file = os.listdir(path)
for filename in file:
    first, last = os.path.splitext(filename)
    if last == ".jpg":
        img_total.append(first)
    else:
        txt_total.append(first)
for img_ in img_total:
    if img_ in txt_total:
        filename_img = img_ + ".jpg"
        path1 = os.path.join(path, filename_img)
        img = cv2.imread(path1)
        filename_txt = img_ + ".txt"
        n = 1
        with open(os.path.join(path, filename_txt), "r+", encoding="utf-8", errors="ignore") as f:
            for line in f:
                aa = line.split(" ")
                x_center = w * float(aa[1])
                y_center = h * float(aa[2])
                width = int(w * float(aa[3]))
                height = int(h * float(aa[4]))
                lefttopx = int(x_center - width / 2.0)
                lefttopy = int(y_center - height / 2.0)
                roi = img[lefttopy + 1:lefttopy + height - 1, lefttopx + 1:lefttopx + width - 1]
                filename_last = img_ + "_" + str(n) + ".jpg"
                print(filename_last)
                path2 = os.path.join(path3, "roi")
                cv2.imwrite(os.path.join(path2, filename_last), roi)
                n = n + 1
    else:
        continue
