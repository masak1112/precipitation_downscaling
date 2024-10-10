#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=train-unet-out.%j
#SBATCH --error=train-unet-err.%j
#SBATCH --time=01:50:00
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

train_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_precipitation/precip_dataset/train
val_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_precipitation/precip_dataset/val
save_dir=../results/exp_20230421_epoch20_repeat2_n8_entire_domain_epoch20
checkpoint=/p/home/jusers/gong1/juwels/bing/precipitation_downscaling/results/exp_20230421_epoch20_repeat2_n8_entire_domain_epoch20/40000_G.pth
#wandb_id="9wqn49c0type_netunet"
epochs=20
model_type=unet
#srun python ../main_scripts/dataset_prep.py
srun --overlap python ../main_scripts/main_train.py --train_dir ${train_dir} --val_dir ${val_dir} --save_dir ${save_dir} --epochs ${epochs} --model_type ${model_type} 
--checkpoint ${checkpoint}
#--wandb_id ${wandb_id}
