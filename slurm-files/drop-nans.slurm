#!/bin/bash
#SBATCH -J air-pollution-data-2000-13-dropnans
#SBATCH -o data-formatting-2000-13.slurmout
#SBATCH -N 1 -n 1
#SBATCH --mem=200gb
#SBATCH -t 00:15:00
#SBATCH --mail-user=nranda99@gmail.com
#SBaTCH --mail-type=all

module load python/3.7
python drop-nans.py
