import numpy as np
import cv2
from PIL import Image


class Volume(object):

    def __init__(self):
        self.spoon_size = 30  # cm2

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

    def get_type(self, food_classes):
        return 1

    def get_volume(self,food_imgs,food_classes,spoon_img):
        scale = self.get_scale(spoon_img)
        volume = []
        for i in range(len(food_imgs)):
            type = self.get_type(food_classes[i])
            pixel_num = self.get_pixel_num(food_imgs[i])
            size_2d = pixel_num * scale


        return volume







if __name__ == '__main__':
    v = Volume()
    spoon_img = ''
    food_imgs = []
    food_classes = []
    volumes = v.get_volume(food_imgs, food_classes, spoon_img)
