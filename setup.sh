#!/bin/bash

echo "alias hadoop='~/ephemeral-hdfs/bin/hadoop'" >> ~/.bash_profile
echo "alias sbt='/mnt/spark/sbt/sbt'" >> ~/.bash_profile
echo "export EDITOR=emacs" >> ~/.bash_profile


yum install -y ncurses-devel tmux

cd ~/spark
sbin/stop-all.sh

cd /mnt
git clone https://github.com/jegonzal/spark.git

cd spark
git remote add peter https://github.com/pbailis/spark.git
git fetch --all
git checkout -b emerson origin/emerson

mv conf old_conf
cp -r ~/spark/conf /mnt/spark/conf


cd /mnt

wget http://ftp.wayne.edu/gnu/emacs/emacs-24.3.tar.gz

tar -xvf emacs-24.3.tar.gz 
cd emacs-24.3
./configure  --with-x-toolkit=no  --with-xpm=no --with-jpeg=no --with-png=no --with-gif=no --with-tiff=no
make -j8
make install


cd /mnt/spark
./deploy.sh


