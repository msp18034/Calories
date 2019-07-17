git@github.com:msp18034/Calories.git

# run process
/opt/spark-2.4.3-bin-hadoop2.7/bin/spark-submit --master yarn --conf spark.streaming.concurrentJobs=5 --total-executor-cores 30 --executor-cores 2 --num-executors 15  --deploy-mode client --jars /opt/kafka_2.11-2.2.0/libs/spark-streaming-kafka-0-8-assembly_2.11-2.4.3.jar Calories/Kafka/detect.py

# show consumer message
/opt/kafka_2.11-2.2.0/bin/kafka-console-consumer.sh --bootstrap-server G4master:9092 --topic outputResult

#Train inception
#
/opt/spark-2.4.3-bin-hadoop2.7/bin/spark-submit --master yarn --deploy-mode client --jars hdfs:///tensorflow-hadoop-1.10.0.jar train.py --input hdfs:///test1.tfrecord --output=hdfs:///out
