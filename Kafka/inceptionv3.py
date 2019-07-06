import numpy as np
import colorsys


# TODO: 代码模块化
class inceptionv3(object):

    def __init__(self):
        self.model_path = "/home/hduser/model_weights/inception.h5"
        self.class_name_path = "/home/hduser/Calories/dataset/coco.names"
        self.classes = _read_class_names(self.class_name_path)
        self.class_num = len(self.classes)
        _get_color(self.class_num)

    def _read_class_names(path):
            names = {}
        with open(path, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names

    def _get_color(self):
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.