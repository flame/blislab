#!/bin/bash

#Single Thread
export KMP_AFFINITY=compact  #Rule to bind core to thread for OMP thread with Intel compiler for parallel version
export OMP_NUM_THREADS=1     #Set OMP number of threads for parallel version
export BLISLAB_IC_NT=1       #Set BLISLAB number of threads for parallel version
k_start=16
k_end=1024
k_blocksize=16
echo "run_step3_st=["
echo -e "%m\t%n\t%k\t%MY_GFLOPS\t%REF_GFLOPS"
for (( k=k_start; k<=k_end; k+=k_blocksize ))
do
    ./test_bl_sgemm.x     $k $k $k 
done
echo "];"


##Multi Thread
#export KMP_AFFINITY=compact  #Rule to bind core to thread for OMP thread with Intel compiler for parallel version
#export OMP_NUM_THREADS=10     #Set OMP number of threads for parallel version
#export BLISLAB_IC_NT=10       #Set BLISLAB number of threads for parallel version
#k_start=64
#k_end=4096
#k_blocksize=64
#echo "run_step3_mt=["
#echo -e "%m\t%n\t%k\t%MY_GFLOPS\t%REF_GFLOPS"
#for (( k=k_start; k<=k_end; k+=k_blocksize ))
#do
#    ./test_bl_sgemm.x     $k $k $k 
#done
#echo "];"

