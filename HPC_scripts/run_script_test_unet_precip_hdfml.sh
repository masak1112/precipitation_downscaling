#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=train-unet-test.%j
#SBATCH --error=train-unet-test.%j
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de


module purge
module load Stages/2022 GCCcore/.11.2.0 dask/2021.9.1
module load PyTorch/1.11-CUDA-11.5
module load torchvision/0.12.0-CUDA-11.5
module load xarray/0.20.1
module load netcdf4-python/1.5.7
source ../sc_venv_template/venv/bin/activate

train_dir=/p/home/jusers/gong1/juwels/scratch_bing/datasets/precip_dataset/train
test_dir=/p/home/jusers/gong1/juwels/scratch_bing/datasets/precip_dataset/test
checkpoint_dir=../results/exp_20230421_epoch20_repeat2/341980_G.pth
save_dir=../results/exp_20230421_epoch20_repoeat_postprocess_repeat2/

model_type=unet

python ../main_scripts/main_test.py --test_dir ${test_dir} --stat_dir ${train_dir} --checkpoint_dir ${checkpoint_dir}  --save_dir ${save_dir} --model_type ${model_type} 
