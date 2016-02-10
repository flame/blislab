#!/bin/bash
export BLISLAB_DIR=.
echo "BLISLAB_DIR = $BLISLAB_DIR"

# Compiler Options (if false, then use GNU compilers)
export BLISLAB_USE_INTEL=true
echo "BLISLAB_USE_INTEL = $BLISLAB_USE_INTEL"

# Whether use BLAS or not?
export BLISLAB_USE_BLAS=true
echo "BLISLAB_USE_BLAS = $BLISLAB_USE_BLAS"

# Optimization Level
export COMPILER_OPT_LEVEL=O3
echo "COMPILER_OPT_LEVEL = $COMPILER_OPT_LEVEL"

# Manually set the mkl path
#export BLISLAB_MKL_DIR=/opt/intel/mkl
#export BLISLAB_MKL_DIR=$MKL_ROOT
export BLISLAB_MKL_DIR=$TACC_MKL_DIR
echo "BLISLAB_MKL_DIR = $BLISLAB_MKL_DIR"

# Parallel Options
export KMP_AFFINITY=compact,verbose
#export OMP_NUM_THREADS=10
#export BLISLAB_IC_NT=10
export OMP_NUM_THREADS=1
export BLISLAB_IC_NT=1

