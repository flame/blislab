#!/bin/bash
#SBATCH -J blis_dgemm_job
#SBATCH -o step3_mt_%j.txt
#SBATCH -p gpu
#SBATCH -t 00:03:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A TRAINING-HPC 
export OMP_NUM_THREADS=10
export BLISGEMM_IC_NT=10
export KMP_AFFINITY=compact,verbose

ibrun tacc_affinity run_blis_dgemm.sh
