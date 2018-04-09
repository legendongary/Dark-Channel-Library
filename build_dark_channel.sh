#!/usr/bin/env bash

CUDA_PATH=/usr/cuda/

# clear

cd src
echo "Compiling with NVCC..."
nvcc -c -o kernels.o kernels.cu -x cu -Xcompiler -fPIC -arch=sm_52

cd ../
echo "Running Python"
python3 build_dark_channel.py

cd src
rm kernels.o
cd ../
