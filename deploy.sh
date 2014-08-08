#!/bin/bash

set -e

cd /mnt/spark
sbt/sbt assembly

cd /mnt
/root/spark-ec2/copy-dir /mnt/spark


