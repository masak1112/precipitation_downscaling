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
#SBATCH --mail-user=b.gong@fz-juelich.de

module purge
module load Stages/2022 GCCcore/.11.2.0 dask/2021.9.1
module load PyTorch/1.11-CUDA-11.5
module load torchvision/0.12.0-CUDA-11.5
module load xarray/0.20.1
module load netcdf4-python/1.5.7
module load Cartopy/0.20.0
module load matplotlib/3.4.3

source ../sc_venv_template/venv/bin/activate

train_dir=/p/home/jusers/gong1/juwels/scratch_bing/datasets/precip_dataset/train
test_dir=/p/home/jusers/gong1/juwels/scratch_bing/datasets/precip_dataset/test
checkpoint_dir=../results/exp_20221218_diffusion_lr/684000_G.pth
save_dir=../outputs/exp_20221218_diffusion_lr/
model_type=diffusion

srun python ../main_scripts/main_test.py --test_dir ${test_dir} --checkpoint_dir ${checkpoint_dir} /
   --save_dir ${save_dir} --model_type ${model_type} --stat_dir ${train_dir}  /
    --patch_size ${patch_size} --window_size ${window_size} --upscale_swinIR ${upscale_swinIR} --timesteps 1000 # > vis.output
