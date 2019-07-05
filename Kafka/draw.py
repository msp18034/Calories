# Utility imports
from __future__ import print_function
import base64
import json
import numpy as np
from timeit import default_timer as timer
from PIL import Image, ImageDraw, ImageFont
from random import randint
from io import BytesIO
import keras
# Streaming imports
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from kafka import KafkaProducer
import random


def drawboxes(image, boxes, final_classes, calories):
    font = ImageFont.truetype(font='/home/hduser/Calories/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    for i in range(len(boxes)):
        cls = final_classes[i]
        box = boxes[i]
        cal = calories[i]
        label = '{} {}cal'.format(cls, cal)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)
        top, left, bottom, right = box
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            #TODO:这里使用的颜色序号有问题！请使用分类模型生成颜色！
            for j in range(thickness):
                draw.rectangle([left + j, top + j, right - j, bottom - j],
                    outline=self.model_od.colors[i])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.model_od.colors[i])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        return image


