#!/bin/bash
#export DYLD_LIBRARY_PATH=/opt/intel/lib:/opt/intel/mkl/lib

#export KMP_AFFINITY=compact
#export OMP_NUM_THREADS=1
#export BLISGEMM_IC_NT=1
##Single Thread
##m=1025
##n=1025
#kmax=1100
#kstep=15
#
#echo "run_step1=["
#for (( k=4; k<kmax; k+=kstep ))
#do
#    ./test_blis_dgemm.x     $k $k $k
#done
#echo "];"


export KMP_AFFINITY=compact
export OMP_NUM_THREADS=10
export BLISGEMM_IC_NT=10
#Single Thread
#m=1025
#n=1025
kmax=5000
kstep=31

echo "run_step1=["
for (( k=4; k<kmax; k+=kstep ))
#for (( k=4840; k<kmax; k+=kstep ))
do
    ./test_blis_dgemm.x     $k $k $k
done
echo "];"


