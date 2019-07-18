# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:28:42 2019

@author: zzx
"""
import cv2
import csv
last_info = ["./",".jpg",-1]
with open("bb_integrate.csv",'w',newline = '') as bb:
    writer = csv.writer(bb)
    with open("integrate-order.csv", 'r') as order:
        csv_reader = csv.reader(order) 
        
        #如果当前文件名和上一个不等，将上一个的数据写到新文件中,last_info更新为现有数据
        #如果当前文件名和上一个相等，扩展last_info
        for s in csv_reader:  
            #s:[class_idx, path, img_name, bbox]
            name = s[2]
            if (name == last_info[1]):
                img1 = cv2.imread(last_info[0]+last_info[1])
                img2 = cv2.imread(s[1]+s[2])
                if(img1.shape[0]==img2.shape[0] and img1.shape[1]==img2.shape[1]):
                    last_info.append(s[0])
                    last_info.append(s[3])
                else: #同名不同图
                    writer.writerow(last_info)#把上一次的写下来
                    last_info = [s[1], s[2],s[0], s[3]]
            else: 
                writer.writerow(last_info)#把上一次的写下来
                last_info = [s[1], s[2],s[0], s[3]]
        writer.writerow(last_info)#最后一个数据