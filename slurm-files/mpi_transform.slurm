#!/bin/bash
#SBATCH -J mpi_transform
#SBATCH -o /project/lindner/moving/summer2018/2019/slurm-output/mpi_transform.slurmout
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --mail-user=csvance@uh.edu
#SBATCH --mail-type=all
#SBATCH -A lindner

PROJECT="/project/lindner/moving/summer2018"

module load MPI4PY/impi-python3.6

cd "$PROJECT/2019"
mpirun -np 3 python3.6 python-scripts/mpi_transform.py --input-path="$PROJECT/Data_structure_3" --output-path="$PROJECT/2019/data-formatted/mpi-test"
