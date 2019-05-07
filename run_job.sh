#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=dist
#SBATCH --mem=10GB
#SBATCH --output=out.dist.%j

#SBATCH --nodes=3
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4

echo "Job started"

module purge
module load openmpi/intel/2.0.1

mpiexec -n 3 python dist_sst.py
# mpirun -n 3 python dist_sst.py

echo "Job completed"
