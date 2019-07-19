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
# Streaming imports
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from kafka import KafkaProducer

# Model imports
from yolov3 import YOLO
from inceptionv3 import Inceptionv3
from volume import NutritionCalculator


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

        # Load Spark Context
        self.sc = SparkContext(appName='MultiFood_detection')
        self.ssc = StreamingContext(self.sc, 2)  # , 3)

        # Make Spark logging less extensive
        log4jLogger = self.sc._jvm.org.apache.log4j
        log_level = log4jLogger.Level.ERROR
        log4jLogger.LogManager.getLogger('org').setLevel(log_level)
        log4jLogger.LogManager.getLogger('akka').setLevel(log_level)
        log4jLogger.LogManager.getLogger('kafka').setLevel(log_level)
        self.logger = log4jLogger.LogManager.getLogger(__name__)

        # Load Network Model & Broadcast to Worker Nodes
        self.model_od = YOLO()
        self.classifier = Inceptionv3()
        self.calorie = NutritionCalculator()


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

            event = json.loads(record[1])
            self.logger.info('Received Message from:' + event['user'])
            decoded = base64.b64decode(event['image'])
            stream = BytesIO(decoded)
            image = Image.open(stream)
            self.logger.info('image open! size'+str(image.size))

            boxes, food_imgs, spoon_img = self.model_od.detect_food(image)
            if spoon_img != 0 and len(boxes) > 0:
                start_c = timer()
                indices, food_classes = self.classifier.eval(food_imgs)
                self.logger.info('classification complete! time:'+str(timer()-start_c))
                start_v = timer()
                scale = self.calorie.get_scale(spoon_img)
                img_ind = list(zip(food_imgs, indices))
                img_ind = self.sc.parallelize(img_ind)
                result = img_ind.map(lambda x: self.calorie.calculate_nutrition(scale, x))
                result.collect()
                calories = result[:, 0].tolist()
                self.logger.info('volume complete! time:'+ str(timer()-start_v))
                drawn_img = self.drawboxes(image, boxes, indices, food_classes, calories)

            else:
                food_classes = ['Not Found']
                calories = [0]
                drawn_img = image

            img_out_buffer = BytesIO()
            drawn_img.save(img_out_buffer, format='png')
            byte_data = img_out_buffer.getvalue()
            drawn_img_b = base64.b64encode(byte_data).decode('utf-8')

            end = timer()
            delta = start-event['start']
            self.logger.info('Started at ' + str(event['start']) + ' seconds.')
            self.logger.info('Done after ' + str(delta) + ' seconds.')
            self.logger.info('Find'+str(len(boxes)) + 'dish(s).')

            result = {'user': event['user'],
                      'start': event['start'],
                      'class': food_classes,
                      'calories': calories,
                      # 'drawn_img': drawn_img_b,
                      'process_time': delta
                      }

            self.outputResult(json.dumps(result))

    def outputResult(self, message):
        self.logger.info("Now sending out....")
        self.producer.send(self.topic_for_produce, message.encode('utf-8'))
        self.producer.flush()

    def drawboxes(self, image, boxes, indices, final_classes, calories):
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        tl = 5  # line thickness

        for i in range(len(boxes)):
            cls = final_classes[i]
            box = boxes[i]
            cal = calories[i]
            label = '{} {}cal'.format(cls, int(cal))
            color = self.classifier.colors[indices[i]]
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


if __name__ == '__main__':
    sod = Spark_Calorie_Calculator(
        topic_to_consume={"inputImage": 0, "inputImage": 1, "inputImage": 2},
        topic_for_produce="outputResult",
        kafka_endpoint="G4master:9092,G401:9092,G402:9092,G403:9092,G404:9092,"
                       "G405:9092,G406:9092,G407:9092,G408:9092,G409:9092,G410:9092,"
                       "G411:9092,G412:9092,G413:9092,G414:9092,G415:9092")
    sod.start_processing()
