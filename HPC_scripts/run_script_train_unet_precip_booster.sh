#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=train-unet-out.%j
#SBATCH --error=train-unet-err.%j
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH --partition=develbooster
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de

module purge
module load Stages/2022 GCCcore/.11.2.0 dask/2021.9.1
module load PyTorch/1.11-CUDA-11.5
module load torchvision/0.12.0-CUDA-11.5
module load xarray/0.20.1
source ../sc_venv_template/venv/bin/activate

train_dir=/p/home/jusers/gong1/juwels/scratch_bing/datasets/precip_dataset/train
val_dir=/p/home/jusers/gong1/juwels/scratch_bing/datasets/precip_dataset/val
save_dir=../results/exp_20230421_epoch20_repeat2

epochs=20
model_type=unet
srun --overlap python ../main_scripts/main_train.py --train_dir ${train_dir} --val_dir ${val_dir} --save_dir ${save_dir} --epochs ${epochs} --model_type ${model_type} --wandb_id ${model_type} 
