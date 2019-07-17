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

# Model imports
from yolov3_keras.yolo import YOLO
from inceptionv3 import Inceptionv3

model_od = YOLO()
classifier = Inceptionv3()


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
        #self.model_od = YOLO()
        #self.classifier = Inceptionv3()


    def start_processing(self):
        zookeeper = "G4master:2181,G401:2181,G402:2181,G403:2181,G404:2181,G405:2181,G406:2181,G407:2181," \
                    "G408:2181,G409:2181,G410:2181,G411:2181,G412:2181,G413:2181,G414:2181,G415:2181"
        groupid = "test-consumer-group"

        """Start consuming from Kafka endpoint and detect objects."""
        # kvs = KafkaUtils.createDirectStream(self.ssc,["inputImage"],{"bootstrap.servers":self.kafka_endpoint})
        kvs = KafkaUtils.createStream(self.ssc, zookeeper, groupid, self.topic_to_consume)
        kvs=kvs.map(lambda x:json.loads(x[1]))
        kvs=kvs.map(lambda event:base64.b64decode(event['image']))
        kvs=kvs.map(lambda image: BytesIO(image))
        kvs=kvs.map(lambda image: Image.open(image))
        # kvs=kvs.map(lambda x: classifier.eval(x))

        kvs.foreachRDD(self.handler)
        self.ssc.start()
        self.ssc.awaitTermination()
        #self.model_od_bc.close_session() #End of model predict

    def handler(self, timestamp, message):
        """Collect messages, detect object and send to kafka endpoint."""
        kvs=message.map(lambda x: classifier.eval(x))
        records =kvs.collect()
        # For performance reasons, we only want to process the newest message
        self.logger.info('\033[3' + str(randint(1, 7)) + ';1m' +  # Color
                         '-' * 25 +
                         '[ NEW MESSAGES: ' + str(len(records)) + ' ]'
                         + '-' * 25 +
                         '\033[0m')  # End color
        start = timer()
        for record in records:
            self.processImage(start, record)
    def test(self,x):
        return x
    def processImage(self,start,image):
        #boxes, single_foods, spoon_box, spoon_img = self.model_od.detect_image(image)

        indices, classes = self.classifier.eval(image)

        calories = []
        for dish in classes:
            calorie = randint(100, 500)
            calories.append(calorie)

        #drawn_img = self.drawboxes(image, boxes, indices, classes, calories)
        #img_out_buffer = BytesIO()
        #drawn_img.save(img_out_buffer, format='png')
        #byte_data = img_out_buffer.getvalue()
        #drawn_img_b = base64.b64encode(byte_data).decode('utf-8')

        end = timer()
        #delta = start-event['start']
        #self.logger.info('Started at ' + str(event['start']) + ' seconds.')
        #self.logger.info('Done after ' + str(delta) + ' seconds.')
        #self.logger.info('Find'+str(len(boxes)) + 'dish(s).')
        '''
        result = {'user': event['user'],
                  'start': event['start'],
                  'class': classes,
                  'calories': calories,
                  # 'drawn_img': drawn_img_b,
                  'process_time': delta
                  }
                  '''
        result = {'user': 'test',
                  'process_time': end-start
                  }

        self.outputResult(json.dumps(result))
        return 1

    def outputResult(self, message):
        self.logger.info("Now sending out....")
        self.producer.send(self.topic_for_produce, message.encode('utf-8'))
        self.producer.flush()

    def drawboxes(self, image, boxes, indices, final_classes, calories):
        font = ImageFont.truetype(font='/home/hduser/Calories/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
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

            thickness = 3
            for j in range(thickness):
                draw.rectangle(
                    [left + j, top + j, right - j, bottom - j],
                    outline=self.classifier.colors[indices[i]])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.classifier.colors[indices[i]])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        return image


if __name__ == '__main__':
    sod = Spark_Calorie_Calculator(
        topic_to_consume={"inputImage": 0, "inputImage": 1, "inputImage": 2},
        topic_for_produce="outputResult",
        kafka_endpoint="G4master:9092,G401:9092,G402:9092,G403:9092,G404:9092,"
                       "G405:9092,G406:9092,G407:9092,G408:9092,G409:9092,G410:9092,"
                       "G411:9092,G412:9092,G413:9092,G414:9092,G415:9092")
    sod.start_processing()
