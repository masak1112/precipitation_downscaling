#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=train-diffusion-out.%j
#SBATCH --error=train-diffusion-err.%j
#SBATCH --time=06:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=booster
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de


module purge
module load Stages/2022 GCCcore/.11.2.0 dask/2021.9.1
module load PyTorch/1.11-CUDA-11.5
module load torchvision/0.12.0-CUDA-11.5
module load xarray/0.20.1
module load netcdf4-python/1.5.7
source ../sc_venv_template/venv/bin/activate


save_dir=../results/exp_20240430_biascorrection
train_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_precipitation/precip_dataset/train
val_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_precipitation/precip_dataset/val
epochs=20
model_type=diffusion2

srun --overlap python ../main_scripts/main_train.py --train_dir ${train_dir} --val_dir ${val_dir} --save_dir ${save_dir} --epochs ${epochs} --model_type ${model_type} 
