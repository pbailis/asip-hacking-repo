#!/bin/bash

set -e

cd /mnt/spark
sbt/sbt "project emerson" "package"
sbin/stop-all.sh

cd /mnt
/root/spark-ec2/copy-dir /mnt/spark

cd /mnt/spark
sbin/start-all.sh



