#!/bin/bash
#SBATCH -J mpi_transform
#SBATCH -o ../slurm-output/mpi_transform_bysite%j.slurmout
#SBATCH -N 1
#SBATCH --mem=63G
#SBATCH -t 30:00:00
#SBATCH --mail-user=narandal@central.uh.edu
#SBATCH --mail-type=all
#SBATCH -A lindner

PROJECT="/project/lindner/air-pollution/current/2019/"

module load MPI4PY/impi-python3.6

cd "$PROJECT"
mpirun -np 20 python3.6 python-scripts/mpi_transform_bysite.py --input-path="/project/lindner/air-pollution/level3_data/" --output-path="$PROJECT/data-formatted/mpi-test/"
