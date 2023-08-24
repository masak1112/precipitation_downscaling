#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=test-unet-out.%j
#SBATCH --error=test-unet-err.%j
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

train_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_precipitation/precip_dataset/train/
test_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_precipitation/precip_dataset/test
checkpoint_dir=../results/exp_20230421_epoch20_repeat2_n8_small_domain503_lr_multisteps/9135_G.pth
save_dir=../results/exp_20230421_epoch20_repeat2_n8_small_domain503_lr_multisteps_postprocess
model_type=unet

srun --overlap python ../main_scripts/main_test.py --test_dir ${test_dir} --stat_dir ${train_dir} --checkpoint ${checkpoint_dir}  --save_dir ${save_dir} --model_type ${model_type} 
