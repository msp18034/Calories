# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 20:06:55 2019

@author: zzx
"""

import numpy as np
import cv2
import random


def parse_line(line):
    '''
    Given a line from the training/test txt file, return parsed info.
    return:
        line_idx: int64
        pic_path: string.
        boxes: shape [N, 4], N is the ground truth count, elements in the second
            dimension are [x_min, y_min, x_max, y_max]
        labels: shape [N]. class index.
    '''
    if 'str' not in str(type(line)):
        line = line.decode()
    s = line.strip().split(' ')
    line_idx = int(s[0])
    pic_path = s[1]
    s = s[2:]
    box_cnt = len(s) // 5
    boxes = []
    labels = []
    for i in range(box_cnt):
        label, x_min, y_min, x_max, y_max = int(s[i * 5]), float(s[i * 5 + 1]), float(s[i * 5 + 2]), float(
            s[i * 5 + 3]), float(s[i * 5 + 4])
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)
    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int64)
    return line_idx, pic_path, boxes, labels

def plot_one_box(img, coord, label="test", color=None, line_thickness=4):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    img: img to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    '''
    tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)
    return img

# 读取图片并进行解码
input_file = "./new_dataset.txt"
f = open(input_file,'r')
for line in f:
    if len(line)>0:
        index,path,boxes,labels = parse_line(line)
        img = cv2.imread(path)
        for box in boxes:
            #img.shape: (y,x,3)
            #box: (x1, y1, x2, y2)
            if(box[2]>img.shape[1] or box[3]>img.shape[0]):
                print(index,path, box,img.shape)
            #plot = plot_one_box(img, box)
            #cv2.imwrite("./ver/"+str(path[6:]),plot)
print("done")    


