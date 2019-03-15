#!/bin/bash
#SBATCH -J transform-houston-fillnan
#SBATCH -o /project/lindner/air-pollution/current/2019/slurm-output/transform-houston-fillnan.slurmout
#SBATCH -N 1
#SBATCH -n 21
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --mail-user=csvance@uh.edu
#SBATCH --mail-type=all
#SBATCH -A lindner

PROJECT="/project/lindner/air-pollution/current"

module load MPI4PY/impi-python3.6

cd "$PROJECT/2019"
mpirun -np 21 python3.6 python-scripts/mpi_transform.py --input-path="$PROJECT/Data_structure_3" --output-path="$PROJECT/2019/data-formatted/transform-houston-fillnan" --fillnan=100000
