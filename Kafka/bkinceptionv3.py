import numpy as np
import colorsys
import keras
from keras.models import load_model, model_from_json
import cv2


class Inceptionv3:

    def __init__(self):
        self.model_path = "/home/hduser/model_weights/cusine.h5"
        self.class_name_path = "/home/hduser/Calories/dataset/172FoodList.txt"
        self.class_names = self.read_class_names()
        self.class_num = len(self.class_names)
        self.get_color()
        self.model = load_model(self.model_path)
        self.model._make_predict_function()
        #self.model_dic = self.serialize_keras_model(model)

    def deserialize_keras_model(self, dictionary):
        """Deserialized the Keras model using the specified dictionary."""
        architecture = dictionary['model']
        weights = dictionary['weights']
        model = model_from_json(architecture)
        model.set_weights(weights)

        return model

    def serialize_keras_model(self, model):
        """Serializes the specified Keras model into a dictionary."""
        dictionary = {}
        dictionary['model'] = model.to_json()
        dictionary['weights'] = model.get_weights()

        return dictionary

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
        #model = self.deserialize_keras_model(self.model_dic)
        result = []
        # TODO:一起预测多张图
        for food in single_foods:
            food = cv2.cvtColor(np.asarray(food), cv2.COLOR_RGB2BGR)
            image = cv2.resize(food, (256, 256),
                           interpolation=cv2.INTER_CUBIC)
            image = np.array(image, dtype='float32')
            image /= 255.
            image = np.expand_dims(image, axis=0)

            #ingredients, actual_class = self.model.predict(image)
            ingredients, actual_class = self.model.predict(image)
            index = np.argmax(actual_class)
            print('class index:', index)
            result.append(index)
        classes = [self.class_names[x] for x in result]
        return result, classes

