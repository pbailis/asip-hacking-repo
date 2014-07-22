#!/bin/bash

cd ~/spark
sbin/stop-all.sh

cd ~
spark-ec2/copy-dir spark

cd ~/spark
sbin/start-all.sh
