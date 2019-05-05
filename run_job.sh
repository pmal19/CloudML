#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=dist
#SBATCH --mem=10GB
#SBATCH --output=out.dist.%j

#SBATCH --nodes=3
#SBATCH --cpus-per-task=1

#SBATCH --ntasks-per-node=1

BIN_PATH="/home/am9031/anaconda3/bin"


echo "Job started"

# $BIN_PATH/mpirun -n $R hostname
# $BIN_PATH/mpirun -n $R ./lab3c1
# $BIN_PATH/mpirun -n $(($R-1)) ./lab3c2
#Uncomment to execute pytorch code
# $BIN_PATH/mpirun -n $R $BIN_PATH/python ./mpi_test.py
# $BIN_PATH/mpirun -n $(($R-1)) $BIN_PATH/python ./lab3c3.py
# $BIN_PATH/mpirun -n $R $BIN_PATH/python ./lab3c4.py

# module load openmpi/intel/2.0.1
# module load pytorch/python2.7/0.3.0_4
# module load pytorch/python3.6/0.3.0_4

# conda activate base
# module load python3/intel/3.6.3 cuda/9.0.176 nccl/cuda9.0/2.4.2

#source ~/pytorch_env/py3.6.3/bin/activate

#mpiexec -n 3 python dist1.py
mpirun -n 3 python dist1.py

# $BIN_PATH/mpirun -n 3 $BIN_PATH/python ./dist1.py

echo "Job completed"
