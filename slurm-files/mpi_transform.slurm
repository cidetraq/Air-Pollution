#!/bin/bash
#SBATCH -J mpi_transform
#SBATCH -o /project/lindner/moving/summer2018/2019/slurm-output/mpi_transform.slurmout
#SBATCH -N 4
#SBATCH -c 1
#SBATCH --mem=20G
#SBATCH -t 02:00:00
#SBATCH --mail-user=csvance@uh.edu
#SBATCH --mail-type=all
#SBATCH -A lindner

PROJECT="/project/lindner/moving/summer2018"

module load Anaconda3/python-3.6
module load OpenMPI/gcc/1.10.7
source activate "$PROJECT/2019/conda"

cd "$PROJECT/2019"
mpirun python python-scripts/mpi_transform.py --input-path="$PROJECT/Data_structure_3" --output-path="$PROJECT/2019/data-formatted/parallel"
