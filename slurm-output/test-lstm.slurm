#!/bin/bash
#SBATCH -J test-lstm-gpu
#SBATCH -o ../slurm-output/test-lstm.slurmout
#SBATCH --gres=gpu:1
#SBATCH -N 1 
#SBATCH --mem=50gb
#SBATCH -t 02:00:00
#SBATCH --mail-user=narandal@central.uh.edu
#SBATCH --mail-type=all
#SBATCH -p gpu

module load Anaconda3/python-3.6
python ../python-scripts/pol_lstm_nicholas.py -u 'cluster' -w 12 -f 'windowed_2000.pkl' -e 100 -l 'lstm_w100_f81_o4.h5'

