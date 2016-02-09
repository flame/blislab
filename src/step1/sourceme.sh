#!/bin/bash
#export BLISLAB_DIR=$PWD
export BLISLAB_DIR=.
echo "BLISLAB_DIR = $BLISLAB_DIR"

# Compiler options (if false, then use GNU compilers)
export BLISLAB_USE_INTEL=true
echo "BLISLAB_USE_INTEL = $BLISLAB_USE_INTEL"

# Whether use BLAS or not?
export BLISLAB_USE_BLAS=true
echo "BLISLAB_USE_BLAS = $BLISLAB_USE_BLAS"

# Optimization Level
export COMPILER_OPTIMIZATION_LEVEL=O2
echo "COMPILER_OPTIMIZATION_LEVEL = $COMPILER_OPTIMIZATION_LEVEL"

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

