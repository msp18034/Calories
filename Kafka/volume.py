import numpy as np
import cv2
from PIL import Image
import math


class NutritionCalculator(object):

    def __init__(self):
        self.spoon_size = 20  # cm2
        #self.para_path = '/home/hduser/Calories/dataset/shape_density.csv'
        # [编号，shape_type, 参数, 密度g/ml]
        self.para_path = '../dataset/shape_density.csv'
        self.para = np.loadtxt(self.para_path, delimiter=',')
        #self.nutrition_path = '/home/hduser/Calories/dataset/nutrition.csv'
        # [编号，热量，碳水化合物，脂肪，蛋白质，纤维素]
        self.nutrition_path = '../dataset/nutrition.csv'
        self.nutrition = np.loadtxt(self.nutrition_path, delimiter=',')

    def get_pixel_num(self, image):
        # The image of here is a cropped image in PIL.Image format
        img = np.asarray(image)
        mask = np.zeros(img.shape[:2], np.uint8)
        SIZE = (1, 65)
        bgdModle = np.zeros(SIZE, np.float64)
        fgdModle = np.zeros(SIZE, np.float64)
        rect = (1, 1, img.shape[1], img.shape[0])

        cv2.grabCut(img, mask, rect, bgdModle, fgdModle, 8, cv2.GC_INIT_WITH_RECT)
        pixel_num = np.bincount(mask.reshape(-1))[3]

        return pixel_num

    def get_scale(self, spoon_img):
        pix = []
        for i in range(8):
            pix.append(self.get_pixel_num(spoon_img))
        return self.spoon_size/np.median(pix)

    def get_volume(self, food_imgs, food_classes, spoon_img):
        scale = self.get_scale(spoon_img)
        volumes = []
        for i in range(len(food_imgs)):
            para = self.para[food_classes[i]]
            shape_type = para[1]
            pixel_num = self.get_pixel_num(food_imgs[i])
            size_2d = pixel_num * scale
            if shape_type == 1:    # 1: cylinder (e.g. 汤菜) 参数为高与直径的比例
                v = math.pow(size_2d/math.pi, 1.5) * 0.65 * para[2]
            elif shape_type == 2:  # 2: ball (e.g. 卤蛋)
                v = math.pow(size_2d/math.pi, 1.5) * 1.2
            elif shape_type == 3:  # 3: half-ball (e.g. 汤面，鼓起来的菜)
                v = math.pow(size_2d/math.pi, 1.5) * 0.6
            elif shape_type == 4:  # 4: cone (e.g. 比较平的炒菜) 高 = 3cm
                v = size_2d * 0.7
            else:    # shape_type == 5: fixed-height  平面*高（参数）
                v = size_2d *0.7 * para[2]
            v = round(v, 2)
            print("volume:",v)
            volumes.append(v)
        return volumes

    def calculate_nutrition(self, food_imgs, food_classes, spoon_img):
        if not spoon_img:
            return [0]
        if len(food_classes) > 0:
            volumes = self.get_volume(food_imgs, food_classes, spoon_img)
        results = []
        for i in range(len(food_classes)):
            n = volumes[i] / self.para[food_classes[i]][3] / 100 * self.nutrition[food_classes[i]][1:]
            print(food_classes[i],self.para[food_classes[i]][3],self.nutrition[food_classes[i]][1:])
            # 体积ml / 密度(g/ml) / 100 * 参考量(每100g)
            results.append(n)
        return np.round(results, 3)


if __name__ == '__main__':
    nc = NutritionCalculator()
    image = Image.open("../8.jpg")
    spoon_img = image.crop((276, 139, 329, 330))
    food_imgs = [image.crop((7, 15,272, 258)), image.crop((330, 122,531, 301)),
                 image.crop((407, 42,536, 141)), image.crop((262, 10,416, 138))]
    food_classes = [123,172,71,118]
    result = nc.calculate_nutrition(food_imgs, food_classes, spoon_img)
    print(result)
    print(np.sum(result, axis=0))
'''
Found 8 boxes for 8.img
diningtable 0.77 (14, 4) (539, 327)
carrot 0.31 (107, 60) (160, 90)
carrot 0.47 (115, 159) (174, 182)
bowl 0.92 (7, 15) (272, 258)
bowl 0.98 (330, 122) (531, 301)
bowl 0.98 (407, 42) (536, 141)
bowl 0.99 (262, 10) (416, 138)
spoon 0.92 (276, 139) (329, 330)
'''

