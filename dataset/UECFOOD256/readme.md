# **UEC FOOD 256**

Data cleaning and transformation for yolov3 object detection

1. Merge useful bb_info.txt files into one file 

   Run `integrate.py` , then order the csv file by Excel, save as integrate-order.csv

   Run `integrate_bb.py`, merge the bbox of same image

2. Check bboxes 

   some image in the dataset have been rotated, thus their bbox is incorrect

   Modify the wrong info to clean the dataset

3.  `new_dataset.txt` is created for tensorflow YOLOv3 training, the format is referred to [https://github.com/wizyoung/YOLOv3_TensorFlow]( https://github.com/wizyoung/YOLOv3_TensorFlow)

4. Divide the dataset for train and validation
