from kafka import KafkaConsumer
from kafka.errors import KafkaError
import json


class Kafka_consumer():

    def __init__(self, kafkahost, kafkaport, kafkatopic):
        self.kafkaHost = kafkahost
        self.kafkaPort = kafkaport
        self.kafkatopic = kafkatopic
        self.consumer = KafkaConsumer(self.kafkatopic, bootstrap_servers='{kafka_host}:{kafka_port}'.format(
            kafka_host=self.kafkaHost,
            kafka_port=self.kafkaPort
        ))

    def getUserFeedback(self, userid):
        try:
            #msg = next(self.consumer)
            for msg in self.consumer:
                dmsg = msg.value.decode("utf-8")
                jmsg = json.loads(dmsg)
                if jmsg['user'] == userid:
                    self.consumer.close()
                    return jmsg
        except KafkaError as e:
            print(e)


def main():
    consumer = Kafka_consumer("G4master", 9092, "outputResult")
    userid = input("userid: ")
    if userid == '':
        userid = "hduser"
    json_back = consumer.getUserFeedback(userid)
    print(json_back)


if __name__ == '__main__':
    main()