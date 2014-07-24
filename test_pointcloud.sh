#!/bin/bash

itStart=1
itEnd=32
itStep=2
regParam=0.01


# cd ~/spark;
export MASTER="local[4]"
./bin/spark-submit --class org.apache.spark.examples.mllib.research.SynchronousADMMTests \
    examples/target/scala-*/spark-examples-*.jar \
    --algorithm SVMADMM --regType L2 --regParam 0.0000001 \
    --format cloud \
    --numPartitions 40 \
    --pointCloudPointsPerPartition 100 \
    --pointCloudPartitionSkew 0 \
    --pointCloudLabelNoise 0 \
    --pointCloudDimension 3 \
    --sweepIterationStart 1 \
    --sweepIterationEnd 100 \
    --sweepIterationStep 10 \
    --ADMMmaxLocalIterations 100 \
    --ADMMepsilon 0

