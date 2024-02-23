#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=test-diffusion-out.%j
#SBATCH --error=test-diffusion-err.%j
#SBATCH --time=07:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=booster
#SBATCH --mail-type=ALL
#SBATCH --mail-user=b.gong@fz-juelich.de

module purge
module load Stages/2022 GCCcore/.11.2.0 dask/2021.9.1
module load PyTorch/1.11-CUDA-11.5
module load torchvision/0.12.0-CUDA-11.5
module load xarray/0.20.1
source ../sc_venv_template/venv/bin/activate

train_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_precipitation/precip_dataset/train
test_dir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5/downscaling_precipitation/precip_dataset/test #test
checkpoint=../results/exp_20231203_diffusion_t4_newscale2_only_rain_t02_yan_zscore_k095_top/159140_G.pth
#checkpoint=../results/exp_20230828_diffusion/3000_G.pth
save_dir=../results/exp_20231203_diffusion_t4_newscale2_only_rain_t02_yan_zscore_k095_top_test
model_type=diffusion

srun --overlap python ../main_scripts/main_test.py --test_dir ${test_dir} --stat_dir ${train_dir} --checkpoint ${checkpoint}  --save_dir ${save_dir} --model_type ${model_type} 
