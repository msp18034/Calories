import numpy as np
import cv2
from PIL import Image
import math


class NutritionCalculator(object):

    def __init__(self):
        self.spoon_size = 25  # cm2
        self.para_path = '/home/hduser/Calories/dataset/shape_density.csv'
        # [编号，shape_type, 参数, 密度g/ml]
        #self.para_path = '../dataset/shape_density.csv'
        self.para = np.loadtxt(self.para_path, delimiter=',')
        self.nutrition_path = '/home/hduser/Calories/dataset/nutrition.csv'
        # [编号，热量，碳水化合物，脂肪，蛋白质，纤维素]
        #self.nutrition_path = '../dataset/nutrition.csv'
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
        spoon_pixel = self.get_pixel_num(spoon_img)
        return spoon_pixel/self.spoon_size

    def get_volume(self, food_imgs, food_classes, spoon_img):
        spoon_pixel = self.get_pixel_num(spoon_img)
        scale = self.spoon_size / spoon_pixel
        volumes = []
        for i in range(len(food_imgs)):
            shape_type = self.para[food_classes[i]][1]
            pixel_num = self.get_pixel_num(food_imgs[i])
            size_2d = pixel_num * scale
            if shape_type == 1:    # 1: cylinder (e.g. 汤菜) 参数为高与直径的比例
                v = math.sqrt(size_2d/math.pi) * 2 * self.para[food_classes[i]][2] * size_2d
            elif shape_type == 2:  # 2: ball (e.g. 卤蛋)
                v = math.sqrt(size_2d/math.pi) * 4/3 * size_2d
            elif shape_type == 3:  # 3: half-ball (e.g. 汤面，鼓起来的菜)
                v = math.sqrt(size_2d/math.pi) * 2/3 * size_2d
            elif shape_type == 4:  # 4: cone (e.g. 比较平的炒菜) 高 = 3cm
                v = size_2d
            else:    # shape_type == 5: fixed-height  平面*高（参数）
                v = size_2d * self.para[food_classes[i]][2]
            v = round(v, 2)
            print(v)
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
            # 体积ml / 密度(g/ml) / 100 * 参考量(每100g)
            results.append(n)
        return np.round(results, 3)


if __name__ == '__main__':
    nc = NutritionCalculator()
    image = Image.open("../6.jpg")
    spoon_img = image.crop((906, 703, 1056, 1088))
    food_imgs = [image.crop((198, 385, 481, 647)), image.crop((585, 600, 933, 1010))]
    food_classes = [2, 15]
    result = nc.calculate_nutrition(food_imgs, food_classes, spoon_img)
    print(result)
