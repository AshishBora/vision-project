#!/bin/bash

#SBATCH -A CS381V-Visual-Recogn
#SBATCH -n 20
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 12:00:00 
#SBATCH -e ./logs/torchModel_B.err
#SBATCH -o ./logs/torchModel_B.out
#SBATCH -J torchModel_B
#SBATCH --mail-user=ashish.bora@utexas.edu
#SBATCH --mail-type=all

echo job $JOB_ID execution at: `date`
NODE_HOSTNAME=`hostname -s`
echo "running on node $NODE_HOSTNAME"

# echo "Starting to tune the last layer"
# echo -e "\n\nTraining LeNet\n\n"

cd ./CaffeNet

luajit torchModel_B.lua
