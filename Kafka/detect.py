# Utility imports
from __future__ import print_function
import base64
import json
import numpy as np
from timeit import default_timer as timer
from PIL import Image
import datetime as dt
from random import randint
from io import BytesIO
# Streaming imports
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from kafka import KafkaProducer

# Object detection imports
import tensorflow as tf


class Spark_Object_Detector():
    """Stream WebCam Images to Kafka Endpoint.
    Keyword arguments:
    source -- Index of Video Device or Filename of Video-File
    interval -- Interval for capturing images in seconds (default 5)
    server -- Host + Port of Kafka Endpoint (default '127.0.0.1:9092')
    """

    def __init__(self,
                 topic_to_consume='pycturestream',
                 topic_for_produce='resultstream',
                 kafka_endpoint='127.0.0.1:9092'):
        """Initialize Spark & TensorFlow environment."""
        self.topic_to_consume = topic_to_consume
        self.topic_for_produce = topic_for_produce
        self.kafka_endpoint = kafka_endpoint



        # Load Spark Context
        sc = SparkContext(appName='PyctureStream')
        self.ssc = StreamingContext(sc,2 )  # , 3)

        # Make Spark logging less extensive
        log4jLogger = sc._jvm.org.apache.log4j
        log_level = log4jLogger.Level.ERROR
        log4jLogger.LogManager.getLogger('org').setLevel(log_level)
        log4jLogger.LogManager.getLogger('akka').setLevel(log_level)
        log4jLogger.LogManager.getLogger('kafka').setLevel(log_level)
        self.logger = log4jLogger.LogManager.getLogger(__name__)



    def start_processing(self):
        zookeeper="G4master:2181,G401:2181,G402:2181,G403:2181,G404:2181,G405:2181,G406:2181,G407:2181,G408:2181,G409:2181,G410:2181,G411:2181,G412:2181,G413:2181,G414:2181,G415:2181"
        topic={"test":0,"test":1,"test":2}
        groupid="test-consumer-group"

        """Start consuming from Kafka endpoint and detect objects."""
        kvs = KafkaUtils.createStream(self.ssc, zookeeper,groupid,topic)

        kvs.foreachRDD(self.handler)
        self.ssc.start()
        self.ssc.awaitTermination()


    def handler(self, timestamp, message):
        """Collect messages, detect object and send to kafka endpoint."""
        records = message.collect()
        # For performance reasons, we only want to process the newest message
        # for every camera_id
        to_process = {}
        self.logger.info( '\033[3' + str(randint(1, 7)) + ';1m' +  # Color
            '-' * 25 +
            '[ NEW MESSAGES: ' + str(len(records)) + ' ]'
            + '-' * 25 +
            '\033[0m' # End color
            )
        dt_now = dt.datetime.now()
        for record in records:
            event = json.loads(record[1])
            self.logger.info('Received Message: ' +
                                event['user'] )
            image=event['image']
            decoded = base64.b64decode(event['image'])
            stream = BytesIO(decoded)
            image = Image.open(stream)
            image_np = self.load_image_into_numpy_array(image)
            print(image_np)
    
    def load_image_into_numpy_array(self, image):
        """Convert PIL image to numpy array."""
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)



if __name__ == '__main__':
    sod = Spark_Object_Detector(
        topic_to_consume="test",

        topic_for_produce='test',
        kafka_endpoint="G4master:9092,G401:9092,G402:9092,G403:9092,G404:9092,G405:9092,G406:9092,G407:9092,G408:9092,G409:9092,G410:9092,G411:9092,G412:9092,G413:9092,G414:9092,G415:9092")
    sod.start_processing()
