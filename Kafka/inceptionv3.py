import numpy as np
import colorsys
import keras


# TODO: 代码模块化
class Inceptionv3(object):

    def __init__(self):
        self.model_path = "/home/hduser/model_weights/inception.h5"
        self.class_name_path = "/home/hduser/Calories/dataset/172FoodList.txt"
        self.class_names = self.read_class_names()
        self.class_num = len(self.class_names)
        self.get_color()
        self.model = keras.models.load_model(self.model_path)
        self.model._make_predict_function()

    def read_class_names(self):
        names = {}
        with open(self.class_name_path, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names

    def get_color(self):
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / self.class_num, 1., 1.)
                      for x in range(self.class_num)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

    def eval(self, single_foods):
        result = []
        for food in single_foods:
            images = []
            images.append(food)
            images = np.array(images)
            ingredients, actual_class = self.model.predict(images)
            index = np.argmax(actual_class)
            print('class index:', index)
            result.append(index)
        classes = [self.class_names[x] for x in result]
        return result, classes
