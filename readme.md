git@github.com:msp18034/Calories.git

## 1.Send images from Kafka direct to Spark Streaming
This cloud be done on any machine from G4master, G401-G415 once the /Caloreis/Kafka/Prodocer.py and image(0.jpg) on it.
You can also send an image from the mobile application in Android Studio.
```
python Producer.py
```

## 2.Run Spark Streaming application on G4master
```
/opt/spark-2.4.3-bin-hadoop2.7/bin/spark-submit --master yarn --conf spark.scheduler.mode=FAIR  --conf spark.speculation=true --conf spark.streaming.unpersist=false --conf spark.yarn.executor.memoryOverhead=1024 --executor-memory 5G  --conf spark.streaming.concurrentJobs=27 --conf spark.memory.fraction=0.6 --conf spark.memory.storageFraction=0.6 --py-files /home/hduser/Calories/Kafka/classify.py,/home/hduser/Calories/Kafka/yolov3.py,/home/hduser/Calories/Kafka/volume.py --total-executor-cores 58 --executor-cores 2 --num-executors 29  --deploy-mode client --driver-memory 5G --jars /opt/kafka_2.11-2.2.0/libs/spark-streaming-kafka-0-8-assembly_2.11-2.4.3.jar --conf spark.task.cpus=2 --conf spark.streaming.blockInterval=90ms  --conf spark.locality.wait=150ms /home/hduser/Calories/Kafka/detect.py
```

## 3.Show output result using Kafka consumer
```
/opt/kafka_2.11-2.2.0/bin/kafka-console-consumer.sh --bootstrap-server G4master:9092 --topic outputResult
```

## Creation of Kafka topics
This has been done in our k8s cluster. <br>
Input topic
```
/opt/kafka_2.11-2.2.0/bin/kafka-topics.sh --create --replication-factor 3 --partitions 15 --topic inputImage --zookeeper G401:2181,G402:2181,G403:2181,G404:2181,G405:2181,G406:2181,G407:2181,G409:2181,G410:2181,G411:2181,G412:2181,G413:2181,G414:2181,G415:2181 
```
Output topic
```
/opt/kafka_2.11-2.2.0/bin/kafka-topics.sh --create --replication-factor 3 --partitions 3 --topic inputImage --zookeeper G401:2181,G402:2181,G403:2181,G404:2181,G405:2181,G406:2181,G407:2181,G409:2181,G410:2181,G411:2181,G412:2181,G413:2181,G414:2181,G415:2181 
```
