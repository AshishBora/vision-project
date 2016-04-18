#!/bin/bash

#SBATCH -A CS381V-Visual-Recogn
#SBATCH -n 20
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH -t 12:00:00 
#SBATCH -e ./logs/test.err
#SBATCH -o ./logs/test.out
#SBATCH -J test
#SBATCH --mail-user=ashish.bora@utexas.edu
#SBATCH --mail-type=all

echo job $JOB_ID execution at: `date`
NODE_HOSTNAME=`hostname -s`
echo "running on node $NODE_HOSTNAME"

# echo "Starting to tune the last layer"
# echo -e "\n\nTraining LeNet\n\n"

cd ./VQA_LSTM_CNN

th eval.lua -input_img_h5 data_img.h5 -input_ques_h5 data_prepro.h5 -input_json data_prepro.json -model_path lstm.t7
