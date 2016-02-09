#!/bin/bash
#export BLISLAB_DIR=$PWD
export BLISLAB_DIR=.
echo "BLISLAB_DIR = $BLISLAB_DIR"

# For macbook pro
#export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/opt/intel/lib:/opt/intel/mkl/lib
#echo "DYLD_LIBRARY_PATH = $DYLD_LIBRARY_PATH"

# Compiler options (if false, then use GNU compilers)
export BLISLAB_USE_INTEL=true
echo "BLISLAB_USE_INTEL = $BLISLAB_USE_INTEL"

# Whether use BLAS or not?
export BLISLAB_USE_BLAS=true
echo "BLISLAB_USE_BLAS = $BLISLAB_USE_BLAS"

# Manually set the mkl path
#export BLISLAB_MKL_DIR=/opt/intel/mkl
#export BLISLAB_MKL_DIR=$MKL_ROOT
export BLISLAB_MKL_DIR=$TACC_MKL_DIR
echo "BLISLAB_MKL_DIR = $BLISLAB_MKL_DIR"

# Parallel options
export KMP_AFFINITY=compact,verbose
#export OMP_NUM_THREADS=10
#export BLISLAB_IC_NT=10
export OMP_NUM_THREADS=1
export BLISLAB_IC_NT=1
