#!/bin/bash
export BLISGEMM_DIR=$PWD
echo "BLISGEMM_DIR = $BLISGEMM_DIR"

# Manually set the target architecture.
export BLISGEMM_ARCH_MAJOR=x86_64
export BLISGEMM_ARCH_MINOR=sandybridge
export BLISGEMM_ARCH=$BLISGEMM_ARCH_MAJOR/$BLISGEMM_ARCH_MINOR
echo "BLISGEMM_ARCH = $BLISGEMM_ARCH"

# For macbook pro
#export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/opt/intel/lib:/opt/intel/mkl/lib
#echo "DYLD_LIBRARY_PATH = $DYLD_LIBRARY_PATH"

# Compiler options (if false, then use GNU compilers)
export BLISGEMM_USE_INTEL=true
echo "BLISGEMM_USE_INTEL = $BLISGEMM_USE_INTEL"

# Whether use BLAS or not?
export BLISGEMM_USE_BLAS=true
echo "BLISGEMM_USE_BLAS = $BLISGEMM_USE_BLAS"

# Manually set the mkl path
#export BLISGEMM_MKL_DIR=/opt/intel/mkl
export BLISGEMM_MKL_DIR=$TACC_MKL_DIR
echo "BLISGEMM_MKL_DIR = $BLISGEMM_MKL_DIR"

# Parallel options
export KMP_AFFINITY=compact,verbose
export OMP_NUM_THREADS=10
export BLISGEMM_IC_NT=10
#export OMP_NUM_THREADS=1
#export BLISGEMM_IC_NT=1
