#!/bin/bash
#SBATCH -J Valid_data_conversion_all_years
#SBATCH -o Valid_data_conversion_all_years.o%j
#SBATCH -N 1 -n 1      
#SBATCH --mem=200gb  
#SBATCH -t 04:00:00  
#SBATCH --mail-user=csdalvi@gmail.com
#SBATCH --mail-type=all

module load python/3.7
python Valid_data_conversion_all_years_ibh.py