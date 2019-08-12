from kafka import KafkaProducer
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import json
from timeit import default_timer as timer


class Kafka_consumer():
    '''
    使用Kafka—python的消费模块
    '''

    def __init__(self, kafkahost, kafkaport, kafkatopic, groupid):
        self.kafkaHost = kafkahost
        self.kafkaPort = kafkaport
        self.kafkatopic = kafkatopic
        self.groupid = groupid
        self.consumer = KafkaConsumer(self.kafkatopic,
                                      bootstrap_servers = '{kafka_host}:{kafka_port}'.format(
            kafka_host=self.kafkaHost,
            kafka_port=self.kafkaPort ))

    def consume_data(self):
        try:
            for message in self.consumer:
                res = json.loads(message.value)
                end = timer()
                #print(res.keys())
                #f.write(str(random.randint(0,9)))
                print("----------------------------------------------------")
                print( "YOLOv3 time", res['yolo'])
                print("Classification time",res['classification'])
                print("Volume time",res['volume'])
                print("Total time", str(res['process_time']))
                print(timer()-res['start'])
                print("----------------------------------------------------")
                yield res
        except KeyboardInterrupt:
            print("KeyboardInterrupt")


def main():

    ##测试生产模块
    #producer = Kafka_producer("127.0.0.1", 9092, "ranktest")
    #for id in range(10):
    #    params = '{abetst}:{null}---'+str(i)
    #    producer.sendjsondata(params)
    ##测试消费模块
    #消费模块的返回格式为ConsumerRecord(topic=u'ranktest', partition=0, offset=
    #\timestamp_type=None, key=None, value='"{abetst}:{null}---0"', checksum=-1
    #\serialized_key_size=-1, serialized_value_size=21)
    consumer = Kafka_consumer('G401', 9092,"outputResult","test-consumer-group")
    #f=open('f.txt','w')
    file="yolo.txt"
    f=open(file,'a+')
    message = consumer.consume_data()
    for res in message:
        #print(i)
        f.write(str(res['yolo'])+" "+str(res['classification'])+" "+str(res['volume'])+" "+str(res['process_time'])+'\n')


if __name__ == '__main__':
    main()
