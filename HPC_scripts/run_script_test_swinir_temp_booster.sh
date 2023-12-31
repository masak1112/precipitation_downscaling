#!/bin/bash -x
#SBATCH --account=deepacf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=postprocessing-out.%j
#SBATCH --error=postprocessing-err.%j
#SBATCH --time=01:00:00
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

# data-directories
# Note template uses Tier2-dataset. Adapt accordingly for other datasets.
datadir=/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_crea6/t2m/all_files/downscaling_tier2_test.nc 
model_basedir=/p/home/jusers/gong1/juwels/bing_folder/downscaling_maelstrom/HPC_scripts/model_base_directory
outdir=../results/exp_20230403_swinIR_booster_temp/postprocessing_116000
model_name=116000_G.pth
exp_name=swinIR
dataset=downscaling_tier2_test

# run job
python ../main_scripts/main_test_temp.py -data_dir ${datadir} -model_base_dir ${model_basedir} \
                                                                    -exp_name ${exp_name} -dataset ${dataset} -model_name ${model_name} \
                                                                    -output_base_dir ${outdir}

