#!/bin/bash
#SBATCH -J descr_stats
#SBATCH -o ../slurm-output/descr_stats_%j.slurmout
#SBATCH -n 40 N 1
#SBATCH --mem=512G
#SBATCH -t 30:00:00
#SBATCH --mail-user=narandal@central.uh.edu
#SBATCH --mail-type=all
#SBATCH -A lindner

PROJECT="/project/lindner/air-pollution/current/2019/"

module load MPI4PY/impi-python3.6

cd "$PROJECT"
python3.6 python-scripts/exploration/descr_stats.py --input-path="/project/lindner/air-pollution/level3_data/" --output-path="/home/narandal/descr_stats/" -b=2016 -e=2017
