#!/bin/bash

itStart=1
itEnd=32
itStep=2
regParam=0.001


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

export MASTER="local[4]"
./bin/spark-submit --class org.apache.spark.examples.mllib.research.SynchronousADMMTests \
    examples/target/scala-*/spark-examples-*.jar \
    --algorithm SVMADMMAsync \
    --regType L2 \
    --regParam $regParam \
    --format cloud \
    --numPartitions 4 \
    --pointCloudPointsPerPartition 100 \
    --pointCloudPartitionSkew 0 \
    --pointCloudLabelNoise 0.0 \
    --pointCloudDimension 3 \
    --sweepIterationStart 50 \
    --sweepIterationEnd  50 \
    --sweepIterationStep 10 \
    --localStats true \
    --ADMMLocalepsilon 1e-3 \
    --ADMMepsilon 1e-5 \

   # 2>&1 | grep -i "ADMM"
