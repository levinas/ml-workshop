#!/bin/bash

# Caffe dependencies
apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
apt-get install -y --no-install-recommends libboost-all-dev
apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
apt-get install -y libatlas-base-dev

apt-get install -y libffi-dev libssl-dev
apt-get install -y python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git libsnappy1 python-sklearn hdf5-tools libhdf5-openmpi-7 libopencv-ml2.4 python-opencv python-h5py

pip install --upgrade pip
pip install --upgrade requests[security]
pip install -U scikit-learn
pip install -U Theano

apt-add-repository ppa:webupd8team/java
apt-get update
apt-get install oracle-java8-installer
apt-get install libjansi-java
wget http://www.scala-lang.org/files/archive/scala-2.10.6.deb
dpkg -i scala-2.10.6.deb

mkdir /opt/torch
git clone https://github.com/torch/distro.git /opt/torch --recursive
cd /opt/torch; bash install-deps;
./install.sh
wget http://d3kbcqa49mib13.cloudfront.net/spark-1.6.1-bin-hadoop2.6.tgz
mkdir /opt/spark
tar -C /opt -xvf spark-1.6.1-bin-hadoop2.6.tgz
ln -s /opt/spark-1.6.1-bin-hadoop2.6/ /opt/spark-latest
# In /etc/bash.bashrc
. /opt/torch/install/bin/torch-activate
export PATH=${PATH}:/opt/spark-latest/bin



# startup.R
wget http://cran.es.r-project.org/bin/linux/ubuntu/trusty/r-base-core_3.2.5-1trusty0_amd64.deb
dpkg -i r-base-core_3.2.5-1trusty0_amd64.deb
R < startup.R

# caffe
cd
git clone https://github.com/BVLC/caffe.git
cd caffe
mkdir build
cd build
cmake ..
make all
make install
cp install/bin/* /usr/local/bin/
cp install/lib/* /usr/local/lib/


# 140.221.67.76 for ml-workshop1
# 140.221.67.77 for ml-workshop2
# 140.221.67.85 for ml-workshop3
