#!/bin/bash

mainclass="edu.berkeley.emerson.Emerson"
emersonjar="emerson/target/scala-2.10/spark-emerson_2.10-1.1.0-SNAPSHOT.jar"
scoptjar="examples/target/scala-2.10/spark-examples-1.1.0-SNAPSHOT-hadoop1.0.4.jar"

# export SPARK_PREPEND_CLASSES=true



./bin/spark-submit --class $mainclass --jars $scoptjar $emersonjar \
    --algorithm ADMM \
    --regType L2 \
    --regParam 1.0 \
    --format cloud \
    --numPartitions 128 \
    --pointCloudPointsPerPartition 10000 \
    --pointCloudPartitionSkew 0.0 \
    --pointCloudLabelNoise 0.1 \
    --pointCloudDimension 5 \
    --ADMMmaxLocalIterations 50000 \
    --localStats true \
    --ADMMLocalepsilon 1e-5 \
    --ADMMepsilon 0 \
    --iterations 200 \
    --runtimeMS 100000 \
    --ADMMrho 1.0 \
    --ADMMLagrangianrho 1.0
   # 2>&1 | grep -i "ADMM"
