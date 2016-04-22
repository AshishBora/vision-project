#!/bin/bash

#SBATCH -A CS381V-Visual-Recogn
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 12:00:00 
#SBATCH -e ./logs/train_C.err
#SBATCH -o ./logs/train_C.out
#SBATCH -J train_C
#SBATCH --mail-user=ashish.bora@utexas.edu
#SBATCH --mail-type=all

echo job $JOB_ID execution at: `date`
NODE_HOSTNAME=`hostname -s`
echo "running on node $NODE_HOSTNAME"

cd ./CaffeNet

luajit train_C.lua
