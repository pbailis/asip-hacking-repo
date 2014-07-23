#!/bin/bash

MASTER=ec2-54-202-185-65.us-west-2.compute.amazonaws.com

itStart=1
itEnd=32
itStep=2
regParam=0.01



cd ~/spark; 
./bin/spark-submit --class org.apache.spark.examples.mllib.research.SynchronousADMMTests \
    examples/target/scala-*/spark-examples-*.jar \
    --algorithm SVMADMM --regType L2 --regParam 1.0 \
    --format cloud \
    --numPartitions 4 \
    --pointCloudPointsPerPartition 10 \
    --pointCloudPartitionSkew 0 \
    --pointCloudLabelNoise 0 \
    --pointCloudDimension 3 \
    --sweepIterationStart 1 \
    --sweepIterationEnd 12 \
    --sweepIterationStep 2 \
    --ADMMmaxLocalIterations 100 
