#!/bin/bash

itStart=1
itEnd=32
itStep=2
regParam=0.01


# cd ~/spark;
export MASTER="local[4]"
./bin/spark-submit --class org.apache.spark.examples.mllib.research.SynchronousADMMTests \
    examples/target/scala-*/spark-examples-*.jar \
    --algorithm SVMADMM --regType L2 --regParam 0.01 \
    --format cloud \
    --numPartitions 20 \
    --pointCloudPointsPerPartition 100 \
    --pointCloudPartitionSkew 0 \
    --pointCloudLabelNoise 0 \
    --pointCloudDimension 3 \
    --sweepIterationStart 50 \
    --sweepIterationEnd  50 \
    --sweepIterationStep 10 \
    --localStats true \
    --ADMMLocalepsilon 1e-3 \
    --ADMMepsilon 1e-5 \

