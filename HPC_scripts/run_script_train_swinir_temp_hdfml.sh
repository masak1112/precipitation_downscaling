#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=train-swinIR-out.%j
#SBATCH --error=train-swinIR-err.%j
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de

#module purge
module load Stages/2022 GCCcore/.11.2.0 dask/2021.9.1
module load PyTorch/1.11-CUDA-11.5
module load torchvision/0.12.0-CUDA-11.5
module load xarray/0.20.1
module load matplotlib/3.4.3
module load netcdf4-python/1.5.7
#ml SciPy-bundle/2021.10
source ../env_setup/venv_hdfml/bin/activate


train_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_crea6/netcdf_data/all_files/downscaling_tier2_train.nc
val_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_crea6/netcdf_data/all_files/downscaling_tier2_val.nc
save_dir=../results/exp_20230227_swinIR_temp
# save_dir=/p/scratch/deepacf/deeprain/ji4/Downsacling/results/swinUnet_exp1017_origin_booster_3years
#train_dir=/p/scratch/deepacf/deeprain/ji4/Downsacling/preprocessing/preprocessed_ifs_radklim_full_disk/train
#val_dir=/p/scratch/deepacf/deeprain/ji4/Downsacling/preprocessing/preprocessed_ifs_radklim_full_disk/val
#save_dir=/p/scratch/deepacf/deeprain/ji4/Downsacling/results/swinSR_exp1110_origin_booster_3years_x2_5x4

epochs=20
dataset_type=temperature
#model_type=vitSR
model_type=swinIR
#model_type=unet
#model_type=swinUnet
srun --overlap python ../main_scripts/main_train.py --dataset_type ${dataset_type} --train_dir ${train_dir} --val_dir ${val_dir} --save_dir ${save_dir} --epochs ${epochs} --model_type ${model_type} 
