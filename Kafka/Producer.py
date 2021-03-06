"""
@Time:   2019-08-18
@author: msp18034
"""

from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
import base64
from io import BytesIO
from PIL import Image
import time
from timeit import default_timer as timer


class Kafka_producer():
    """使用kafka的生产模块"""

    def __init__(self, kafkahost, kafkaport, kafkatopic):
        self.kafkaHost = kafkahost
        self.kafkaPort = kafkaport
        self.kafkatopic = kafkatopic
        kafka_endpoint = "G4master:9092,G401:9092,G402:9092,G403:9092,G404:9092,"\
               "G405:9092,G406:9092,G407:9092,G408:9092,G409:9092,G410:9092,"\
               "G411:9092,G412:9092,G413:9092,G414:9092,G415:9092"
        self.producer = KafkaProducer(bootstrap_servers=kafka_endpoint)

    def sendjsondata(self, result):
        try:
            parmas_message = json.dumps(result)
            producer = self.producer
            producer.send(self.kafkatopic, parmas_message.encode('utf-8'))
            producer.flush()
        except KafkaError as e:
            print(e)


def image_to_base64(image_path):
    img = Image.open(image_path)
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str


def main():
    producer = Kafka_producer("G401", 9092, "inputImage")
    for j in range(10):
        for i in range(2):
            start = timer()
            image = image_to_base64("/home/hduser/Calories/0.jpg")
            result = {
                'start': start,
                'image': image,
                'user': "test"+str(i)
            }
            print(str(i+1), "ok")
            producer.sendjsondata(result)
            time.sleep(0.5)
        time.sleep(10)


if __name__ == '__main__':
    main()
