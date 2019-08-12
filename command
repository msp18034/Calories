git@github.com:msp18034/Calories.git

/opt/spark-2.4.3-bin-hadoop2.7/bin/spark-submit --master yarn --conf spark.scheduler.mode=FAIR  --conf spark.speculation=true --conf spark.streaming.unpersist=false --conf spark.yarn.executor.memoryOverhead=1024 --executor-memory 5G  --conf spark.streaming.concurrentJobs=15 --conf spark.memory.fraction=0.8 --conf spark.memory.storageFraction=0.6 --py-files /home/hduser/Calories/Kafka/classify.py,/home/hduser/Calories/Kafka/yolov3.py,/home/hduser/Calories/Kafka/volume.py --total-executor-cores 30 --executor-cores 2 --num-executors 15  --deploy-mode client --driver-memory 5G --jars /opt/kafka_2.11-2.2.0/libs/spark-streaming-kafka-0-8-assembly_2.11-2.4.3.jar --conf spark.task.cpus=2 --conf spark.streaming.blockInterval=400ms /home/hduser/Calories/Kafka/detect.py

/opt/spark-2.4.3-bin-hadoop2.7/bin/spark-submit --master yarn --conf spark.scheduler.mode=FAIR  --conf spark.speculation=true --conf spark.streaming.unpersist=false --conf spark.yarn.executor.memoryOverhead=1024 --executor-memory 5G  --conf spark.streaming.concurrentJobs=27 --conf spark.memory.fraction=0.6 --conf spark.memory.storageFraction=0.6 --py-files /home/hduser/Calories/Kafka/classify.py,/home/hduser/Calories/Kafka/yolov3.py,/home/hduser/Calories/Kafka/volume.py --total-executor-cores 58 --executor-cores 2 --num-executors 29  --deploy-mode client --driver-memory 5G --jars /opt/kafka_2.11-2.2.0/libs/spark-streaming-kafka-0-8-assembly_2.11-2.4.3.jar --conf spark.task.cpus=2 --conf spark.streaming.blockInterval=90ms  --conf spark.locality.wait=150ms /home/hduser/Calories/Kafka/detect.py


# show consumer message
/opt/kafka_2.11-2.2.0/bin/kafka-console-consumer.sh --bootstrap-server G4master:9092 --topic outputResult

#Train inception
/opt/spark-2.4.3-bin-hadoop2.7/bin/spark-submit --master yarn --deploy-mode client --jars hdfs:///tensorflow-hadoop-1.10.0.jar train.py --input hdfs:///test1.tfrecord --output=hdfs:///out
