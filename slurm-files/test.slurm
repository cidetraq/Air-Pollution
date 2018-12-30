#!/bin/bash
#SBATCH -J air-pollution-data-2000-14-format-pre-no
#SBATCH -o data-format-14-17-pre-no.slurmout
#SBATCH -N 1 -n 1
#SBATCH --mem=200gb
#SBATCH -t 02:00:00
#SBATCH --mail-user=nranda99@gmail.com
#SBATCH --mail-type=all

module load python/3.7
python data-formatting-2014-2017.py
