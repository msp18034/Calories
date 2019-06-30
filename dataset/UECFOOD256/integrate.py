# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:19:43 2019

@author: zzx
"""

import csv

list = [2,3,4,5,8,10,14,15,16,21,24,25,26,28,29,30,32,33,
        35,36,37,39,40,41,43,44,45,46,47,48,49,50,52,53,54,
        57,58,60,62,63,65,66,69,70,71,75,76,77,78,79,80,83,89,
        90,93,95,96,99,100,101,102,104,105,106,107,108,109,110,
        111,112,113,115,116,121,123,124,125,128,129,131,134,135,
        136,137,138,139,140,141,142,143,144,146,147,149,150,153,
        154,155,157,163,168,174,181,182,183,184,186,187,190,191,
        192,193,196,198,200,202,203,204,205,207,208,211,212,213,
        214,216,217,218,219,220,221,222,223,226,227,228,229,230,
        231,233,234,235,236,237,238] #删除的类
class_idx = 0

with open ('integrate.csv','w',newline='') as csvfile:
     writer = csv.writer(csvfile)
     writer.writerow(["class_idx","path","img","bbox"])
     
     for i in range(256):
        num = i + 1
        #如果在删除的类别里，跳过
        if (num in list):
            continue
        
        path = "./" + str(num) + "/"
        bb_path = path + "bb_info.txt"
        
        with open (bb_path,'r') as bb:
            for lines in bb:
                lines = lines.strip('\n')
                s = lines.split(' ',1)
                img_name = s[0] + ".jpg"
                #写入新的类别序号，原图片路径，图片名，bbox
                writer.writerow([class_idx, path, img_name, s[1] ])
        
        class_idx += 1
    
