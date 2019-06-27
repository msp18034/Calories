from pyspark import SparkContext , SparkConf
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils



# Create a local StreamingContext with two working thread and batch interval of 1 second
conf = SparkConf().setAppName('Kafka')
sc = SparkContext(conf=conf)
ssc = StreamingContext(sc, 30)
zookeeper="G4master:2181,G401:2181,G402:2181,G403:2181,G404:2181,G405:2181,G406:2181,G407:2181,G408:2181,G409:2181,G410:2181,G411:2181,G412:2181,G413:2181,G414:2181,G415:2181"
topic={"test":0,"test":1,"test":2}
groupid="test-consumer-group"
lines=KafkaUtils.createStream(ssc,zookeeper,groupid,topic)
line=lines.map(lambda x:x[1])
line.foreachRDD(handler)
def handler(data):
    records=data.collect()
    for record in records:
        event=json.loads(record)
        print(event['user'])

# Print the first ten elements of each RDD generated in this DStream to the console

ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate
