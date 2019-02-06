#!/bin/bash
#SBATCH -J ds2_2000_hou
#SBATCH -o /project/lindner/moving/summer2018/2019/slurm-output/ds2_2000_hou.slurmout
#SBATCH -N 3 -n 20
#SBATCH --mem=64G
#SBATCH -t 05:00:00
#SBATCH --mail-user=narandal@central.uh.edu
#SBATCH --mail-type=all
#SBATCH -A lindner

PROJECT="/project/lindner/moving/summer2018"

module load MPI4PY/impi-python3.6

cd "$PROJECT/2019"
mpirun -np 20 python3.6 python-scripts/format_ds2.py 
