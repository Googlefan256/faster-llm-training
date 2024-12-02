#!/bin/bash
set -e
nvcc -c ./kernels/kernel.cu -o ./kernels/libfusedkernel.o -O3 --gpu-code=sm_80 -arch=compute_80 -Xcompiler -fPIC
ar crus ./kernels/libfusedkernel.a ./kernels/libfusedkernel.o