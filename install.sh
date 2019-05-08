#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=dist
#SBATCH --mem=10GB
#SBATCH --output=out.install.%j
#SBATCH --error=err.install.%j
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=100:00:00

. /scratch/pm2758/anaconda3/etc/profile.d/conda.sh

conda activate sst
# source activate sst

module purge
module load cuda/10.1.105
module load gcc/6.3.0
# module load openmpi/intel/2.0.1

conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -c pytorch magma-cuda90
conda install -c conda-forge openmpi
conda install -c conda-forge cmake

git submodule sync 
git submodule update --init --recursive

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
