#!/bin/bash
#SBATCH -J train_rf
#SBATCH -o /project/lindner/moving/summer2018/2019/slurm-output/train_rf.slurmout
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH --mail-user=csvance@uh.edu
#SBATCH --mail-type=all
#SBATCH -A lindner

PROJECT="/project/lindner/moving/summer2018"
module load Anaconda3/python-3.6
source activate "$PROJECT/2019/conda"

cd "$PROJECT/2019"
python python-scripts/train_rf.py --input-path="$PROJECT/2019/data-formatted/parallel" --output-path="$PROJECT/2019/models/csvance_best_rf.pickle"
