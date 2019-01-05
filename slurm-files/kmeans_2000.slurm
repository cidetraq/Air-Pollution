#!/bin/bash
PROJECT="/project/lindner/moving/summer2018"
#SBATCH -J pol_kmeans_2000
#SBATCH -o ../slurm-output/pol_kmeans_2000.slurmout
#SBATCH -N 1 -n 1
#SBATCH --mem=100gb
#SBATCH -t 10:00:00
#SBATCH --mail-user=narandal@central.uh.edu
#SBATCH --mail-type=all

module load Anaconda3/python-3.6
source activate "$PROJECT/2019/conda"
python ../python-scripts/kmeans.py
