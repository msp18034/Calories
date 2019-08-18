# Image-based Dietary Assessment System in the Cloud

Web page: https://i.cs.hku.hk/~msp18034/



## Introduction

It is a real-time food recognition and nutrition estimating system on  **Spark Streaming**, which can:

- Find the location of each dish in a multi-dishes photo

- Identify the type of each dish with higher accuracy by additional information on ingredients or cooking methods

- Estimate calorie and nutrition of food in good accuracy

  ![](D:\0study\Project\code\Calories\images\objective.png)

## System design

In our project, we proposed a system that can directly detect, identify
dishes and calculate calories and nutrition values by a photo of a meal that token
on the top-view. 

- The [client end](https://github.com/msp18034/FoodDiary) of our system could give personalized
  assessment to users. 

- In terms of the [server end](https://github.com/msp18034/Calories), the strategy we proposed is
  utilizing YOLOv3 to detect table wares and further classifying dishes to
  realize multi-dish recognition. A multi-task DCNN model is trained to identify
  food category with cuisine method as auxiliary task, and [our model](https://github.com/msp18034/ClassificationModel) shows a higher
  accuracy than individual DCNN model with 6% improvement. To tackle the storage
  problem of big DCNN model in mobile application and the concurrency bottleneck
  of traditional server, we deploy our backend to a cloud cluster and develop a
  scalable spark streaming application that achieves nearly real-time inference. 

![](D:\0study\Project\code\Calories\images\workflow.png)



## Quick start

### Preparation

Build hadoop, spark, kafka, cassandra cluster on your machines.

![](D:\0study\Project\code\Calories\images\structure.png)

##### Our environment

| Applications | Version    | Specification                    |
| ------------ | ---------- | -------------------------------- |
| Hadoop       | 2.7.5      | 1 master node, 29 slave nodes    |
| Spark        | 2.4.3      | 1 driver, 29 executors           |
| Kafka        | 2.11-2.2.0 | 16 nodes for a zookeeper cluster |
| Cassandra    | 3.11.4     | 1 data centre with 16 nodes      |
| Python       | 3.7.3      |                                  |
| Anaconda     | 3.7        |                                  |

| Package       | Version | Package          | Version |
| ------------- | ------- | ---------------- | ------- |
| Tensorflow    | 1.14.0  | Keras            | 2.2.4   |
| opencv-python | 4.1.0   | PIL              | 6.1.0   |
| Numpy         | 1.16.4  | Cgi              | 2.6     |
| kafka-python  | 1.4.6   | cassandra-driver | 3.18.0  |



##### Create Kafka topics

###### Input topic

```shell
/opt/kafka_2.11-2.2.0/bin/kafka-topics.sh --create --replication-factor 3 --partitions 15 --topic inputImage --zookeeper G401:2181,G402:2181,G403:2181,G404:2181,G405:2181,G406:2181,G407:2181,G409:2181,G410:2181,G411:2181,G412:2181,G413:2181,G414:2181,G415:2181 
```
###### Output topic

```shell
/opt/kafka_2.11-2.2.0/bin/kafka-topics.sh --create --replication-factor 3 --partitions 3 --topic inputImage --zookeeper G401:2181,G402:2181,G403:2181,G404:2181,G405:2181,G406:2181,G407:2181,G409:2181,G410:2181,G411:2181,G412:2181,G413:2181,G414:2181,G415:2181 
```

##### Initialize Cassandra database

```
python  ./cassandra/initial.py
```

##### Train models

The pre-trained YOLOv3 model and our multi-task model could be download [here](https://drive.google.com/drive/folders/1zfG5s2SIJzflJkg7F-ULhEE9NgvPL6vj?usp=sharing)

### Execution steps


##### 1. Run Spark Streaming application on G4master
```shell
/opt/spark-2.4.3-bin-hadoop2.7/bin/spark-submit --master yarn --conf spark.scheduler.mode=FAIR  --conf spark.speculation=true --conf spark.streaming.unpersist=false --conf spark.yarn.executor.memoryOverhead=1024 --executor-memory 5G  --conf spark.streaming.concurrentJobs=27 --conf spark.memory.fraction=0.6 --conf spark.memory.storageFraction=0.6 --py-files /home/hduser/Calories/Kafka/classify.py,/home/hduser/Calories/Kafka/yolov3.py,/home/hduser/Calories/Kafka/volume.py --total-executor-cores 58 --executor-cores 2 --num-executors 29  --deploy-mode client --driver-memory 5G --jars /opt/kafka_2.11-2.2.0/libs/spark-streaming-kafka-0-8-assembly_2.11-2.4.3.jar --conf spark.task.cpus=2 --conf spark.streaming.blockInterval=90ms  --conf spark.locality.wait=150ms /home/hduser/Calories/Kafka/detect_db.py
```

##### 2. Send images from Kafka to Spark Streaming

This could be done on any machine from G4master, G401-G415 once the spark streaming application is started.

*Suggestion: before real application, first send some images to warm up.*

```
python ./Kafka/Producer.py
```

##### 3. Execute HTTP servers

Execute the servers to receive and response information to user clients. If you just want to testing the cluster performance by sending images directly from kafka, this step could be skipped.

```
python ./Kafka/server.py
python ./cassandra/server.py
```

##### 4. Show output result using Kafka consumer

If use mobile application, the output result will be shown directly. In the same time, result could also shown by:

```shell
/opt/kafka_2.11-2.2.0/bin/kafka-console-consumer.sh --bootstrap-server G4master:9092 --topic outputResult
```

