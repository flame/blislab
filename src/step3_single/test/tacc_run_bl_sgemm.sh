#!/bin/bash
#SBATCH -J bl_sgemm_job
#SBATCH -o bl_sgemm_output-%j.txt
#SBATCH -p gpu
#SBATCH -t 01:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A TRAINING-HPC 
export OMP_NUM_THREADS=10
export BLISGEMM_IC_NT=10
export KMP_AFFINITY=compact,verbose

ibrun tacc_affinity run_bl_sgemm.sh
