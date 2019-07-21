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
from keras.models import load_model, model_from_json
# Model imports
from yolov3 import YOLO
from inceptionv3 import Inceptionv3
from volume import NutritionCalculator


def handler(timestamp, message):
    """Collect messages, detect object and send to kafka endpoint."""
    records = message.collect()
    # For performance reasons, we only want to process the newest message
    logger.info('\033[3' + str(randint(1, 7)) + ';1m' +  # Color
                     '-' * 25 +
                     '[ NEW MESSAGES: ' + str(len(records)) + ' ]'
                     + '-' * 25 +
                     '\033[0m')  # End color
    start = timer()
    #for record in records:
    def eval(record):
        event = json.loads(record[1])
        decoded = base64.b64decode(event['image'])
        stream = BytesIO(decoded)
        image = Image.open(stream)
        food = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        image = cv2.resize(food, (256, 256), interpolation=cv2.INTER_CUBIC)
        image = np.array(image, dtype='float32')
        image /= 255.
        image = np.expand_dims(image, axis=0)

        #ingredients, actual_class = model.predict(image)
        ingredients, actual_class =bdmodel.value.predict(image)
        index = np.argmax(actual_class)
        print('class index:', index)
        #classes = [self.class_names[x] for x in result]
        return image

    result = message.map(lambda x: eval(x))
    print("------------------finished map--------------------------")
    print(result.count())

    def something():
        '''
        boxes, food_imgs, spoon_img = model_od.detect_food(image)
        if spoon_img != 0 and len(boxes) > 0:
            start_c = timer()
            indices, food_classes = classifier.eval(food_imgs)
            logger.info('classification complete! time:'+str(timer()-start_c))
            start_v = timer()
            result = calorie.calculate_nutrition(food_imgs, indices, spoon_img)
            calories = result[:, 0].tolist()
            logger.info('volume complete! time:'+ str(timer()-start_v))
            start_d = timer()
            drawn_img = drawboxes(image, boxes, indices, food_classes, calories)
            logger.info('draw complete! time:' + str(timer()-start_d))

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
        logger.info('Started at ' + str(event['start']) + ' seconds.')
        logger.info('Done after ' + str(delta) + ' seconds.')
        logger.info('Find'+str(len(boxes)) + 'dish(s).')

        result = {'user': event['user'],
                  'start': event['start'],
                  'class': food_classes,
                  'calories': calories,
                  # 'drawn_img': drawn_img_b,
                  'process_time': delta
                  }

        outputResult(json.dumps(result))
        '''

def outputResult(message):
    logger.info("Now sending out....")
    producer.send(topic_for_produce, message.encode('utf-8'))
    producer.flush()

'''
def drawboxes(image, boxes, indices, final_classes, calories):
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    tl = 5  # line thickness

    for i in range(len(boxes)):
        cls = final_classes[i]
        box = boxes[i]
        cal = calories[i]
        label = '{} {}cal'.format(cls, int(cal))
        color = classifier.colors[indices[i]]
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
'''


topic_to_consume = {"inputImage": 0, "inputImage": 1, "inputImage": 2},
topic_for_produce = "outputResult",
kafka_endpoint = "G4master:9092,G401:9092,G402:9092,G403:9092,G404:9092,"\
               "G405:9092,G406:9092,G407:9092,G408:9092,G409:9092,G410:9092,"\
               "G411:9092,G412:9092,G413:9092,G414:9092,G415:9092"
producer = KafkaProducer(bootstrap_servers=kafka_endpoint)

# Load Spark Context
sc = SparkContext(appName='MultiFood_detection')
ssc = StreamingContext(sc, 2)  # , 3)

# Make Spark logging less extensive
log4jLogger = sc._jvm.org.apache.log4j
log_level = log4jLogger.Level.ERROR
log4jLogger.LogManager.getLogger('org').setLevel(log_level)
log4jLogger.LogManager.getLogger('akka').setLevel(log_level)
log4jLogger.LogManager.getLogger('kafka').setLevel(log_level)
logger = log4jLogger.LogManager.getLogger(__name__)

# Load Network Model & Broadcast to Worker Nodes
#detect = YOLO()
#classifier = Inceptionv3()
#calorie = NutritionCalculator()
model = load_model("/home/hduser/model_weights/cusine.h5")
print("loaded model")
bdmodel = sc.broadcast(model)
print("broadcasted")

zookeeper = "G4master:2181,G401:2181,G402:2181,G403:2181,G404:2181,G405:2181,G406:2181,G407:2181," \
            "G408:2181,G409:2181,G410:2181,G411:2181,G412:2181,G413:2181,G414:2181,G415:2181"
groupid = "test-consumer-group"

"""Start consuming from Kafka endpoint and detect objects."""
kvs = KafkaUtils.createStream(ssc, zookeeper, groupid, topic_to_consume)

kvs.foreachRDD(handler)
ssc.start()
ssc.awaitTermination()




