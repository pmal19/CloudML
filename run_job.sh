#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=dist
#SBATCH --mem=100GB
#SBATCH --output=out.dist.%j

##SBATCH --time=100:00:00
##SBATCH--gres=gpu:1

#SBATCH --mem=250GB
#SBATCH --nodes=3
#SBATCH --cpus-per-task=28
##SBATCH --exclusive
##SBATCH --time=00:60:00
##SBATCH --gres=gpu:k80:4
##SBATCH --gres=gpu:p40:4
##SBATCH --reservation=chung


##SBATCH --nodes=$R
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=1
##SBATCH --partition=c32_41
##SBATCH --output=out-$R.%j
##SBATCH --mem=10GB

echo "Job started"

# $BIN_PATH/mpirun -n $R hostname
# $BIN_PATH/mpirun -n $R ./lab3c1
# $BIN_PATH/mpirun -n $(($R-1)) ./lab3c2
#Uncomment to execute pytorch code
# $BIN_PATH/mpirun -n $R $BIN_PATH/python ./mpi_test.py
# $BIN_PATH/mpirun -n $(($R-1)) $BIN_PATH/python ./lab3c3.py
# $BIN_PATH/mpirun -n $R $BIN_PATH/python ./lab3c4.py

module load openmpi/intel/2.0.1
# module load pytorch/python2.7/0.3.0_4
# module load pytorch/python3.6/0.3.0_4

# conda activate base
module load python3/intel/3.6.3 cuda/9.0.176 nccl/cuda9.0/2.4.2

source ~/pytorch_env/py3.6.3/bin/activate

mpirun -np 3 python dist1.py

echo "Job completed"
