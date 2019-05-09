#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=dist
#SBATCH --mem=10GB
#SBATCH --output=out.dist.%j

#SBATCH --nodes=4
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4

echo "Job started"

module purge
# module load pytorch/python2.7/0.3.0_4
# module load openmpi/intel/2.0.1

# mpiexec -n 3 python dist_sst.py
# mpirun -n 3 python dist_sst.py

python main.py --data_dir ./data --pretrained_vector_dir /scratch/gs3011/CloudML/wiki-news-300d-1M.vec --model CNN

echo "Job completed"
