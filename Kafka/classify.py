"""
@Time:   2019-08-18
@author: msp18034
"""

import numpy as np
import colorsys
import keras
from keras.models import load_model, model_from_json
import cv2
from PIL import Image


# model_path = "/home/hduser/model_weights/cusine.h5"
# class_name_path = "/home/hduser/Calories/dataset/172FoodList.txt"
# class_names = read_class_names(class_name_path)
# model = load_model(model_path)
# model._make_predict_function()
# model_dic = serialize_keras_model(model)


def deserialize_keras_model(dictionary):
    """Deserialized the Keras model using the specified dictionary."""
    architecture = dictionary['model']
    weights = dictionary['weights']
    model = model_from_json(architecture)
    model.set_weights(weights)

    return model


def serialize_keras_model(model):
    """Serializes the specified Keras model into a dictionary."""
    dictionary = {}
    dictionary['model'] = model.to_json()
    dictionary['weights'] = model.get_weights()

    return dictionary


def read_class_names(class_name_path):
    names = {}
    with open(class_name_path, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names


def get_color(class_num):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / class_num, 1., 1.)
                  for x in range(class_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors


def process_img(food_imgs):
    pimgs = []
    for food in food_imgs:
        image = cv2.resize(food, (256, 256),
                       interpolation=cv2.INTER_CUBIC)
        image = np.array(image, dtype='float32')
        image /= 255.
        image=np.expand_dims(image,axis=0)
        pimgs.append(image)
    pimgs=np.concatenate([x for x in pimgs])
    return pimgs

'''
def eval(single_foods):
    pimg = process_img(single_foods)
    result = model.predict(pimg)
    index = [np.argmax(i) for i in result]
    classes = [class_names[x] for x in result]
    return index, classes
'''


def drawboxes(img, boxes, indices, final_classes, calories):
    colors = get_color(173)
    #img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    tl = 5  # line thickness

    for i in range(len(boxes)):
        cls = final_classes[i]
        box = boxes[i]
        cal = calories[i]
        label = '{} {}cal'.format(cls, int(cal))
        color = colors[indices[i]]
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        # label
        tf = 1  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 8, thickness=tf)[0]
        if box[1] - t_size[1] < 0:
            c1 = c1[0], c1[1] + t_size[1]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 9,
                    [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return image


