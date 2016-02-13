#!/bin/bash
export BLISLAB_DIR=.
echo "BLISLAB_DIR = $BLISLAB_DIR"

# Compiler Options (true: Intel compiler; false: GNU compilers)
export BLISLAB_USE_INTEL=true
echo "BLISLAB_USE_INTEL = $BLISLAB_USE_INTEL"

# Whether use BLAS or not?
export BLISLAB_USE_BLAS=true
echo "BLISLAB_USE_BLAS = $BLISLAB_USE_BLAS"

# Optimization Level (O0, O1, O2, O3)
export COMPILER_OPT_LEVEL=O3
echo "COMPILER_OPT_LEVEL = $COMPILER_OPT_LEVEL"

# Manually set the BLAS path if BLIS_USE_BLAS=true and using GNU compiler.
export BLAS_DIR=/home/jianyu/lib/blis
echo "BLAS_DIR = $BLAS_DIR"

# Parallel Options
export KMP_AFFINITY=compact,verbose
#export OMP_NUM_THREADS=10
#export BLISLAB_IC_NT=10
export OMP_NUM_THREADS=1
export BLISLAB_IC_NT=1

