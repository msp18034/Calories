# Utility imports
from __future__ import print_function
import base64
import json
import numpy as np
import time
from timeit import default_timer as timer
from PIL import Image
import cv2
from random import randint
from io import BytesIO

# Streaming imports
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from kafka import KafkaProducer
from keras.models import load_model

#cassandra imports
from cassandra.cluster import Cluster
from datetime import datetime

# Model imports
import yolov3 
import classify
import volume


def evalPar(iterator):
    for record in iterator:
        start_p=time.time()
        event = json.loads(record[1])
        decoded = base64.b64decode(event['image'])
        stream = BytesIO(decoded)
        image = Image.open(stream)
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        # image: PIL Image
        # img: cv2 image

        # YOLO part
        start_y =time.time()
        # graph = tf.get_default_graph()
        # with graph.as_default():
        pimage = yolov3.process_image(img)
        outs = bdmodel_od.value.predict(pimage)
        boxes, classes, scores = yolov3._yolo_out(outs, img.shape)
        spoon_img, bowl_boxes, food_imgs = yolov3.fliter(img, boxes, scores, classes)

        if len(spoon_img) > 0 and len(bowl_boxes) > 0:
            # classification part
            start_c = time.time()
            pimg = classify.process_img(food_imgs)
            _, p_result = bdmodel_cls.value.predict(pimg)
            indices = [np.argmax(i) for i in p_result]
            food_classes = [class_names[x] for x in indices]

            # volume part
            start_v = time.time()
            c_result = volume.calculate_nutrition(food_imgs, indices, spoon_img, para, nutrition)
            calories = c_result[:, 0].tolist()
            carbo = c_result[:, 1].tolist()
            protein = c_result[:, 3].tolist()
            fat = c_result[:, 2].tolist()
            fiber = c_result[:, 4].tolist()

            # draw part
            start_d = time.time()
            drawn_image = classify.drawboxes(img, bowl_boxes, indices, food_classes, calories)

        else:
            food_classes = ['Not Found']
            calories = [0]
            carbo = [0]
            fat = [0]
            fiber = [0]
            protein = [0]
            drawn_image = image
            start_c = timer()
            start_v = timer()
            start_d = timer()

        # output
        img_out_buffer = BytesIO()
        drawn_image.save(img_out_buffer, format='png')
        byte_data = img_out_buffer.getvalue()
        drawn_image_b = base64.b64encode(byte_data).decode('utf-8')

        end = time.time()
        delta = end - start_p

        output = {'user': event['user'],
                  'start': event['start'],
                  'start_p':start_p-event['start'],
                  'class': food_classes,
                  'calories': calories,
                  'fat': fat,
                  'protein': protein,
                  'carbo': carbo,
                  'fiber': fiber,
                  'yolo': start_c - start_y,
                  'classification': start_v - start_c,
                  'volume': start_d - start_v,
                  'drawn_img': drawn_img_b,
                  'process_time': delta,
                  'end':end
                  }
        output = json.dumps(output)
        yield output


def handler(timestamp, message):
    """Collect messages, detect object and send to kafka endpoint"""
    start_r = time.time()
    records = message.collect()
    # For performance reasons, we only want to process the newest message
    logger.info('\033[3' + str(randint(1, 7)) + ';1m' +  # Color
                    '-' * 20 +
                    '[ PROCESSED IMAGE: ' + str(len(records)) + ' ]'
                    + '-' * 20 +
                    '\033[0m')  # End color
    for record in records:
        print(time.time()-start_r)
        record = json.dumps({**json.loads(record), **{"endtime": time.time()-start_r}})
        #record['end']=timer()
        outputResult(record)        


def outputResult(message):
    logger.info("Now sending out....")
   # message = json.dumps({**json.loads(message), **{"endtime": timer()}})

    producer.send(topic_for_produce, message.encode('utf-8'))
    producer.flush()
    time = datetime.now()

    jsondata = json.loads(message)
    query = "INSERT INTO records (id, time, photo, food, calorie, carbo, protein, fat, fiber)" \
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    session.execute(query, (jsondata['user'], time, jsondata['drawn_img'], jsondata['class'],
                                 jsondata['calories'], jsondata['carbo'], jsondata['protein'],
                                 jsondata['fat'], jsondata['fiber']))



topic_to_consume = {"inputImage": 0, "inputImage": 1, "inputImage": 2}
topic_for_produce = "outputResult"
kafka_endpoint = "G401:9092,G402:9092,G403:9092,G404:9092,"\
               "G405:9092,G406:9092,G407:9092,G408:9092,G409:9092,G410:9092,"\
               "G411:9092,G412:9092,G413:9092,G414:9092,G415:9092"
producer = KafkaProducer(bootstrap_servers=kafka_endpoint)

# Load Spark Context
sc = SparkContext(appName='MultiFood_detection')
ssc = StreamingContext(sc, 0.4)  # odcast(producer)Z oo

# Make Spark logging less extensive
log4jLogger = sc._jvm.org.apache.log4j
log_level = log4jLogger.Level.ERROR
log4jLogger.LogManager.getLogger('org').setLevel(log_level)
log4jLogger.LogManager.getLogger('akka').setLevel(log_level)
log4jLogger.LogManager.getLogger('kafka').setLevel(log_level)
logger = log4jLogger.LogManager.getLogger(__name__)

# connect to cassandra
cluster = Cluster(['G401', 'G402'])  # 随意写两个就能找到整个集群
session = cluster.connect("fooddiary")

# load and broadcast model
model_od = load_model("/home/hduser/model_weights/yolo.h5")
print("loaded model object detection")
bdmodel_od = sc.broadcast(model_od)
print("broadcasted model detection")

model_cls = load_model("/home/hduser/model_weights/cusine.h5")
print("loaded model classification")
bdmodel_cls = sc.broadcast(model_cls)
print("broadcasted model classification")

class_name_path = "/home/hduser/Calories/dataset/172FoodList.txt"
class_names = classify.read_class_names(class_name_path)
para_path = '/home/hduser/Calories/dataset/shape_density.csv'
# [编号，shape_type, 参数, 密度g/ml]
para = np.loadtxt(para_path, delimiter=',')
nutrition_path = '/home/hduser/Calories/dataset/nutrition.csv'
# [编号，热量，碳水化合物，脂肪，蛋白质，纤维素]
nutrition = np.loadtxt(nutrition_path, delimiter=',')


zookeeper = "G4master:2181,G401:2181,G402:2181,G403:2181,G404:2181,G405:2181,G406:2181,G407:2181," \
            "G408:2181,G409:2181,G410:2181,G411:2181,G412:2181,G413:2181,G414:2181,G415:2181"
groupid = "test-consumer-group"

"""Start consuming from Kafka endpoint and detect objects."""
kvs = KafkaUtils.createStream(ssc, zookeeper, groupid, topic_to_consume)
kvs = kvs.mapPartitions(evalPar)
#kvs = KafkaUtils.createDirectStream(ssc, topics=['inputImage2'],kafkaParams = {"metadata.broker.list":kafka_endpoint})
kvs.foreachRDD(handler)
ssc.start()
ssc.awaitTermination()

