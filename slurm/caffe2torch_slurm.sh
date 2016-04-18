#!/bin/bash

#SBATCH -A CS381V-Visual-Recogn
#SBATCH -n 20
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 12:00:00 
#SBATCH -e /work/04001/ashishb/maverick/vision-project/logs/caffe2torch.err
#SBATCH -o /work/04001/ashishb/maverick/vision-project/logs/caffe2torch.out
#SBATCH -J caffe2torch
#SBATCH --mail-user=ashish.bora@utexas.edu
#SBATCH --mail-type=all

echo job $JOB_ID execution at: `date`
NODE_HOSTNAME=`hostname -s`
echo "running on node $NODE_HOSTNAME"

# echo "Starting to tune the last layer"
# echo -e "\n\nTraining LeNet\n\n"

cd /work/04001/ashishb/maverick/vision-project/CaffeNet

luajit caffe2torch.lua
