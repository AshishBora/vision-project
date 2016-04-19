#!/bin/bash

#SBATCH -A CS381V-Visual-Recogn
#SBATCH -n 20
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 12:00:00 
#SBATCH -e ./logs/test_10perclass.err
#SBATCH -o ./logs/test_10perclass.out
#SBATCH -J test_10perclass
#SBATCH --mail-user=as1992@cs.utexas.edu
#SBATCH --mail-type=all

echo job $JOB_ID execution at: `date`
NODE_HOSTNAME=`hostname -s`
echo "running on node $NODE_HOSTNAME"

cd ./CaffeNet

luajit testing.lua
