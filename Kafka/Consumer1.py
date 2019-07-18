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

                print(res['class'])
                print(res['calories'])
                print( " Receive time: ", res['process_time'])
                print("Total time", str(end-res['start']))
                #yield message
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
    consumer = Kafka_consumer('G401', 9092, "outputResult", 'test-python-rankte')
    message = consumer.consume_data()
    for i in message:
        print(i)


if __name__ == '__main__':
    main()
