#! /bin/bash

mkdir -p /data/gg921

# Ceres Solver installation
cd /data/gg921
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
git checkout $(git describe --tags) 
mkdir build
cd build
cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX=/data/gg921
make -j
make install

# COLMAP installation
cd /data/gg921
git clone https://github.com/colmap/colmap.git
cd colmap
git checkout dev
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/data/gg921
make -j
make install
