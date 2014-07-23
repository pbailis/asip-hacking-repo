#!/bin/bash

MASTER=ec2-54-202-185-65.us-west-2.compute.amazonaws.com

itStart=1
itEnd=32
itStep=2
regParam=0.01

# cd ~/spark; ./bin/spark-submit \
#     --class org.apache.spark.examples.mllib.research.SynchronousADMMTests \
#     examples/target/scala-*/spark-examples-*.jar \
#     --algorithm SVMADMM --regType L2 --regParam 1.0 \
#     --format bismarck \
#     --input hdfs://$MASTER:9000/user/root/bismarck_data/forest* \
#     --numPartitions 40 \
#     --ADMMmaxLocalIterations 100000 \
#     --sweepIterationStart 10 --sweepIterationEnd 10 --sweepIterationStep 1 \


# cd ~/spark; ./bin/spark-submit \
#     --class org.apache.spark.examples.mllib.research.SynchronousADMMTests \
#     examples/target/scala-*/spark-examples-*.jar \
#     --algorithm SVM --regType L2 --regParam 0.01 \
#     --format bismarck \
#     --input hdfs://$MASTER:9000/user/root/bismarck_data/forest* \
#     --numPartitions 40 \
#     --ADMMmaxLocalIterations 100000 \
#     --sweepIterationStart 1000 --sweepIterationEnd 1000 --sweepIterationStep 1 \
#     | grep RESULT | cut -c 9-


cd ~/spark; ./bin/spark-submit \
    --class org.apache.spark.examples.mllib.research.SynchronousADMMTests \
    examples/target/scala-*/spark-examples-*.jar \
    --algorithm SVMADMM --regType L2 --regParam $regParam \
    --format bismarck \
    --input hdfs://$MASTER:9000/user/root/bismarck_data/forest* \
    --numPartitions 40 \
    --ADMMmaxLocalIterations 10 \
    --sweepIterationStart $itStart --sweepIterationEnd $itEnd --sweepIterationStep $itStep \
    | grep RESULT | cut -c 9-


# cd ~/spark; ./bin/spark-submit \
#     --class org.apache.spark.examples.mllib.research.SynchronousADMMTests \
#     examples/target/scala-*/spark-examples-*.jar \
#     --algorithm SVM --regType L2 --regParam $regParam \
#     --format bismarck \
#     --input hdfs://$MASTER:9000/user/root/bismarck_data/forest* \
#     --sweepIterationStart $itStart --sweepIterationEnd $itEnd --sweepIterationStep $itStep \
#     --numPartitions 40 \
#     | grep RESULT | cut -c 9-
