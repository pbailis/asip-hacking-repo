#!/bin/bash

itStart=1
itEnd=32
itStep=2
regParam=0.0001


export SPARK_PREPEND_CLASSES=true

# cd ~/spark;

#
#cd ~/spark;
#
#sbt/sbt assembly;
#cd ~;
#spark-ec2/copy-dir spark;
#cd spark;
#sbin/stop-all.sh;
#
#sleep 5;
#
#sbin/start-all.sh


./bin/spark-submit --class org.apache.spark.examples.mllib.research.SynchronousADMMTests \
    examples/target/scala-*/spark-examples-*.jar \
    --algorithm SVMADMM \
    --regType L2 \
    --regParam $regParam \
    --format cloud \
    --numPartitions 40 \
    --pointCloudPointsPerPartition 1000000 \
    --pointCloudPartitionSkew 0.0 \
    --pointCloudLabelNoise 0.2 \
    --pointCloudDimension 100 \
    --runtimeMS 10000 \
    --ADMMmaxLocalIterations 100 \
    --localStats true \
    --ADMMLocalepsilon 1e-3 \
    --ADMMepsilon 1e-5
   # 2>&1 | grep -i "ADMM"
