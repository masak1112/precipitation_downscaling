#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=test-out.%j
#SBATCH --error=test-err.%j
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develbooster
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maximbr@post.bgu.ac.il

#module purge
module load Stages/2022 GCCcore/.11.2.0 dask/2021.9.1
module load PyTorch/1.11-CUDA-11.5
module load torchvision
module load xarray
#ml SciPy-bundle/2021.10
source ../env_setup/venv_booster/bin/activate


srun python ../main_scripts/dataset_temp_month.py
