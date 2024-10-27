import cv2
import numpy as np
import glob as gb
import os
import imageio

COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

def match_obj(obj_list, obj_id):
    try:
        index = obj_list.index(obj_id)
    except:
        index = -1
    return index


def txt2video(i,img):
    # img = cv2.imread("../dancetrack0026/test/" + "{}.jpg".format(i))
    # img2 = img

    with open('dancetrack0026.txt', 'r') as f:
        lines = f.readlines()
        object_list = []
        center_list = []
        img_list = []
        for line in lines:
            img_id = line.split(',')[0]
            if img_id == str(i):
                object_id = line.split(',')[1]
                object_list.append(object_id)
                x, y, w, h = int(float(line.split(',')[2])), int(float(line.split(',')[3])), int(float(line.split(',')[4])), int(
                    float(line.split(',')[5]))
                center1 = (int(int(x) + int(w) / 2), int(int(y) + int(h) / 2))
                center_list.append(center1)
                color = [int(c) for c in COLORS[int(object_id) % len(COLORS)]]
            # if img_id == str(int(i) + 1):
            #     object_id = line.split(',')[1]
            #     index = match_obj(object_list, object_id)
            #     x, y, w, h = int(float(line.split(',')[2])), int(float(line.split(',')[3])), int(float(line.split(',')[4])), int(
            #         float(line.split(',')[5]))
            #     center2 = (int(int(x) + int(w) / 2), int(int(y) + int(h) / 2))
            #     color = [int(c) for c in COLORS[int(object_id) % len(COLORS)]]
            #     if index != -1:
                img2 = cv2.rectangle(img, (x, y), (x + w, y + h), color)
                #img2 = cv2.arrowedLine(img2, center_list[1], center1, color, 1, 8, 0, 0.5)
                img2 = cv2.putText(img2, 'ID-' + str(int(object_id)), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                img_list.append(img2)

        # out.write(img2)
        # out.release()
            # print(img_list)
            # for item in img_list:
            #     out.write(img2)
        return img2
if __name__ == '__main__':
    img_path = "1.jpg"
    img = cv2.imread(img_path)
    txt2video(1,img)
    cv2.imwrite("asdfgedec.jpg",img)
