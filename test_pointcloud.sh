#!/bin/bash

mainclass="edu.berkeley.emerson.Emerson"
emersonjar="emerson/target/scala-2.10/spark-emerson_2.10-1.1.0-SNAPSHOT.jar"
scoptjar="examples/target/scala-2.10/spark-examples-1.1.0-SNAPSHOT-hadoop1.0.4.jar"

# export SPARK_PREPEND_CLASSES=true

echo "./bin/spark-submit --class $mainclass $emersonjar "

./bin/spark-submit --class $mainclass $emersonjar \
    --algorithm SVMADMM \
    --regType L2 \
    --regParam $regParam \
    --format cloud \
    --numPartitions 128 \
    --pointCloudPointsPerPartition 10000 \
    --pointCloudPartitionSkew 0.0 \
    --pointCloudLabelNoise 0.2 \
    --pointCloudDimension 100 \
    --runtimeMS 10000 \
    --ADMMmaxLocalIterations 100 \
    --localStats true \
    --ADMMLocalepsilon 1e-3 \
    --ADMMepsilon 1e-5
   # 2>&1 | grep -i "ADMM"
