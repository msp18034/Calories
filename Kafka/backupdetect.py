# Utility imports
from __future__ import print_function
import base64
import json
import numpy as np
from timeit import default_timer as timer
from PIL import Image, ImageDraw, ImageFont
import cv2
from random import randint
from io import BytesIO
import keras
from keras.models import load_model
# Streaming imports
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from kafka import KafkaProducer

#cassandra imports
from cassandra.cluster import Cluster
from datetime import datetime

# Model imports
import yolov3
import classify
import volume


class Spark_Calorie_Calculator():
    """Stream Food Images to Kafka Endpoint."""

    def __init__(self,
                 topic_to_consume='instream',
                 topic_for_produce='ourstream',
                 kafka_endpoint='127.0.0.1:9092'):
        """Initialize Spark & TensorFlow environment."""
        self.topic_to_consume = topic_to_consume
        self.topic_for_produce = topic_for_produce
        self.kafka_endpoint = kafka_endpoint
        self.producer = KafkaProducer(bootstrap_servers=kafka_endpoint)

        #connect to cassandra
        cluster = Cluster(['G401', 'G402'])  # 随意写两个就能找到整个集群
        self.session = cluster.connect("fooddiary")

        # Load Spark Context
        sc = SparkContext(appName='MultiFood_detection')
        self.ssc = StreamingContext(sc, 2)  # , 3)

        # Make Spark logging less extensive
        log4jLogger = sc._jvm.org.apache.log4j
        log_level = log4jLogger.Level.ERROR
        log4jLogger.LogManager.getLogger('org').setLevel(log_level)
        log4jLogger.LogManager.getLogger('akka').setLevel(log_level)
        log4jLogger.LogManager.getLogger('kafka').setLevel(log_level)
        self.logger = log4jLogger.LogManager.getLogger(__name__)

        # Load Network Model & Broadcast to Worker Nodes
        self.model_od = load_model("/home/hduser/model_weights/yolo.h5")
        self.model_od._make_predict_function()
        print("loaded model object detection")

        self.model_cls = load_model("/home/hduser/model_weights/cusine.h5")
        self.model_cls._make_predict_function()
        print("loaded model classification")

        class_name_path = "/home/hduser/Calories/dataset/172FoodList-en.txt"
        self.class_names = classify.read_class_names(class_name_path)
        para_path = '/home/hduser/Calories/dataset/shape_density.csv'
        # [编号，shape_type, 参数, 密度g/ml]
        self.para = np.loadtxt(para_path, delimiter=',')
        nutrition_path = '/home/hduser/Calories/dataset/nutrition.csv'
        # [编号，热量，碳水化合物，脂肪，蛋白质，纤维素]
        self.nutrition = np.loadtxt(nutrition_path, delimiter=',')


    def start_processing(self):
        zookeeper = "G4master:2181,G401:2181,G402:2181,G403:2181,G404:2181,G405:2181,G406:2181,G407:2181," \
                    "G408:2181,G409:2181,G410:2181,G411:2181,G412:2181,G413:2181,G414:2181,G415:2181"
        groupid = "test-consumer-group"

        """Start consuming from Kafka endpoint and detect objects."""
        kvs = KafkaUtils.createStream(self.ssc, zookeeper, groupid, self.topic_to_consume)

        kvs.foreachRDD(self.handler)
        self.ssc.start()
        self.ssc.awaitTermination()
        #self.model_od_bc.close_session() #End of model predict

    def handler(self, timestamp, message):
        """Collect messages, detect object and send to kafka endpoint."""
        records = message.collect()
        # For performance reasons, we only want to process the newest message
        self.logger.info('\033[3' + str(randint(1, 7)) + ';1m' +  # Color
                         '-' * 25 +
                         '[ NEW MESSAGES: ' + str(len(records)) + ' ]'
                         + '-' * 25 +
                         '\033[0m')  # End color
        start = timer()

        for record in records:

            start_p = timer()
            event = json.loads(record[1])
            decoded = base64.b64decode(event['image'])
            stream = BytesIO(decoded)
            image = Image.open(stream)
            img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            # image: PIL Image
            # img: cv2 image 这个最好cache一下！

            #YOLO part
            start_y = timer()
            pimage = yolov3.process_image(img)
            outs = self.model_od.predict(pimage)
            boxes, classes, scores = yolov3._yolo_out(outs, img.shape)
            spoon_img, bowl_boxes, food_imgs = yolov3.fliter(img, boxes, scores, classes)
            drawn_image = []
            if len(spoon_img) > 0 and len(bowl_boxes) > 0:
                # classification part
                start_c = timer()
                pimg = classify.process_img(food_imgs)
                _, p_result = self.model_cls.predict(pimg)
                indices = [np.argmax(i) for i in p_result]
                food_classes = [self.class_names[x] for x in indices]
                self.logger.info('classification complete! time:'+str(timer()-start_c))

                # volume part
                start_v = timer()
                c_result = volume.calculate_nutrition(food_imgs, indices, spoon_img, self.para, self.nutrition)
                #热量，碳水化合物，脂肪，蛋白质，纤维素
                calories = c_result[:, 0].tolist()
                carbo = c_result[:, 1].tolist()
                protein = c_result[:, 3].tolist()
                fat = c_result[:, 2].tolist()
                fiber = c_result[:, 4].tolist()
                self.logger.info('volume complete! time:' + str(timer()-start_v))

                # draw part
                start_d = timer()
                drawn_image = classify.drawboxes(img, bowl_boxes, indices, food_classes, calories)
                self.logger.info('draw complete! time:' + str(timer()-start_d))

            else:
                food_classes = ['Not Found']
                calories = [0]
                carbo = [0]
                fat = [0]
                fiber = [0]
                protein = [0]
                drawn_image = image

            #output
            img_out_buffer = BytesIO()
            drawn_image.save(img_out_buffer, format='png')
            byte_data = img_out_buffer.getvalue()
            drawn_image_b = base64.b64encode(byte_data).decode('utf-8')

            end = timer()
            delta = end - start_p
            self.logger.info('Done after ' + str(delta) + ' seconds.')

            output = {'user': event['user'],
                     # 'start': event['start'],
                      'class': food_classes,
                      'calories': calories,
                      'fat': fat,
                      'protein': protein,
                      'carbo': carbo,
                      'fiber': fiber,
                      'drawn_img': drawn_image_b,
                      'process_time': delta
                      }

            self.outputResult(output)

    def outputResult(self, jsondata):
        message = json.dumps(jsondata)
        self.logger.info("Now sending out....")
        self.producer.send(self.topic_for_produce, message.encode('utf-8'))
        self.producer.flush()
        time = datetime.now()

        query = "INSERT INTO records (id, time, photo, food, calorie, carbo, protein, fat, fiber)"\
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        self.session.execute(query, (jsondata['user'], time, jsondata['drawn_img'], jsondata['class'],
                                     jsondata['calories'], jsondata['carbo'], jsondata['protein'],
                                     jsondata['fat'], jsondata['fiber']))


if __name__ == '__main__':
    sod = Spark_Calorie_Calculator(
        topic_to_consume={"inputImage": 0, "inputImage": 1, "inputImage": 2},
        topic_for_produce="outputResult1",
        kafka_endpoint="G4master:9092,G401:9092,G402:9092,G403:9092,G404:9092,"
                       "G405:9092,G406:9092,G407:9092,G408:9092,G409:9092,G410:9092,"
                       "G411:9092,G412:9092,G413:9092,G414:9092,G415:9092")
    sod.start_processing()
