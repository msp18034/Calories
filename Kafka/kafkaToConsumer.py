from kafka import KafkaProducer
import cv2
import base64
import json
import time

def processImage(image):
    jpg=cv2.imread(image)
    jpg_as_text=base64.b64encode(jpg).decode('utf-8')
    result={
            'image':jpg_as_text,
            'user': "hduser"
            }
    send_to_kafka(result)

def send_to_kafka(data):
    producer = KafkaProducer(bootstrap_servers="G4master:9092",                                          value_serializer=lambda m: json.dumps(m).encode('utf8'))
    topic="test"
    producer.send(topic,data)


for i in range(100)
    processImage("AH.jpg")
    time.sleep(10)
