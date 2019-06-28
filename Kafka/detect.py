# Utility imports
from __future__ import print_function
import base64
import json
import numpy as np
from timeit import default_timer as timer
from PIL import Image, ImageDraw, ImageFont
from random import randint
from io import BytesIO

# Streaming imports
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from kafka import KafkaProducer

# Model imports
from yolov3_keras.yolo import YOLO


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
        model_od = YOLO()
        self.model_od_bc = sc.broadcast(model_od)

    def start_processing(self):
        zookeeper = "G4master:2181,G401:2181,G402:2181,G403:2181,G404:2181,G405:2181,G406:2181,G407:2181," \
                    "G408:2181,G409:2181,G410:2181,G411:2181,G412:2181,G413:2181,G414:2181,G415:2181"
        topic = {"test": 0, "test": 1, "test": 2}
        groupid = "test-consumer-group"

        """Start consuming from Kafka endpoint and detect objects."""
        kvs = KafkaUtils.createStream(self.ssc, zookeeper, groupid, topic)

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

        for record in records:
            event = json.loads(record[1])
            self.logger.info('Received Message from: ' + event['user'])
            decoded = base64.b64decode(event['image'])
            stream = BytesIO(decoded)
            image = Image.open(stream)
            start = timer()
            pre_classes, boxes, single_food_imgs = self.model_od_bc.detect_image(image)
            end = timer()
            delta = end - start

            #get calories
            calories = []
            for dish in single_food_imgs:
                calorie = 100
                calories.append(calorie)

            drawed_img = self.drawboxes(image, boxes, pre_classes, calories)

            self.logger.info('Done after ' + str(delta) + ' seconds.')
            self.logger.info('Find'+len(pre_classes) + 'dish(s).')
            #TODO: 这里的操作需要用map写吗？后续操作：每个切完片的图进一步预测分类和营养成分，将最终画好的图返回客户端，
            result={'user': event['user'],
                    'class':"Fish"}
            self.outputResult(json.dumps(result))
    
    def outputResult(self,message):
        self.logger.info("Now sending out....")
        self.producer.send(self.topic_for_produce,message.encode('utf-8'))
        self.producer.flush()

    def drawboxes(self, image, boxes, final_classes, calories):
        font = ImageFont.truetype(size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
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
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.model_od_bc.colors[i])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.model_od_bc.colors[i])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
        return image


if __name__ == '__main__':
    sod = Spark_Calorie_Calculator(
        topic_to_consume="test",
        topic_for_produce='test',
        kafka_endpoint="G4master:9092,G401:9092,G402:9092,G403:9092,G404:9092,"
                       "G405:9092,G406:9092,G407:9092,G408:9092,G409:9092,G410:9092,"
                       "G411:9092,G412:9092,G413:9092,G414:9092,G415:9092")
    sod.start_processing()
