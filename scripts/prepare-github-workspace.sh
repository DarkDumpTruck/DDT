#!/bin/bash

cmake_version="3.27.7"
libtorch_version="2.1.1"

sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install gcc-10 g++-10 -y
sudo apt-get remove cmake -y
wget -O cmake.sh https://github.com/Kitware/CMake/releases/download/v$cmake_version/cmake-$cmake_version-linux-x86_64.sh
sudo mkdir /opt/cmake
sudo mkdir /opt/cmake
sudo chmod +x cmake.sh
sudo ./cmake.sh --skip-license --prefix=/opt/cmake
rm -f ./cmake.sh
export PATH=$PATH:/opt/cmake/bin

cd cpp

mkdir -p subprojects
cd subprojects
curl -o libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-$libtorch_version%2Bcpu.zip
unzip -qq libtorch.zip
rm -f libtorch.zip
echo "Downloaded libtorch, build version: $(cat ./libtorch/build-version), build hash: $(cat ./libtorch/build-hash)"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/libtorch/lib
export LIBRARY_PATH=$LIBRARY_PATH:$(pwd)/libtorch/lib
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$(pwd)/libtorch
cd ..

python3 -m pip install --upgrade pip
pip3 install meson
sudo apt-get install ninja-build -y

meson wrap install gtest
meson setup -Dcpp_args="-ffast-math -march=native" build --buildtype=release

