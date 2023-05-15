## Precipitation downscaling by deep learning

This repository serves as the official PyTorch implementation for precipitation downscaling using deep learning techniques. Specifically, we investigate several neural networks: the Swin transformer (SwinIR) and a combination of U-Net with a Generative Adversarial Network (GAN) and a diffusion component for probabilistic downscaling. Our approach involves mapping short-range forecasts obtained from the Integrated Forecast System (IFS), which are provided on a regular spherical grid with a resolution of IFS=0.1°, to high-resolution observation radar data (RADKLIM) with a resolution of 0.01°. The mapping ratio between these resolutions is a factor of 10.

This repo can be used for temperature downscaling, which is the PyTorch version of the [repo](https://gitlab.jsc.fz-juelich.de/esde/machine-learning/downscaling_maelstrom)(TensorFLow version)

## Access the dataset
The dataset used in this work is currently not publicly available. If you are interested in accessing the dataset, please contact Bing Gong at b.gong@fz-juelich.de for further information and possible arrangements.


## Prerequisites
  * Linux or macOS
  * Python>=3.6
  * PyTorch


## Installation 

1. Clone this repo by typing the following command in your personal target dirctory:

```bash 
git clone https://gitlab.jsc.fz-juelich.de/esde/machine-learning/precipitation_downscaling.git
cd precipitation_downscaling
```

Now, we will setup the virtual enviornment. To setup envionrment on HPC systems, we recommend to use the tool provide from this [repo](https://gitlab.jsc.fz-juelich.de/kesselheim1/sc_venv_template).
2. Copy this repo under `precipitaiton_downscaling` directory

```bash 
git clone https://gitlab.jsc.fz-juelich.de/kesselheim1/sc_venv_template
cd sc_venv_template
```

3. Modify the `modules.sh` files to including the modules 

```bash 
module load Stages/2022 GCCcore/.11.2.0 dask/2021.9.1
module load PyTorch/1.11-CUDA-11.5
```
3. Modify the `requirement.txt` file to include all the required packages (see `precipitation_downscaling/env_setup/requirement.txt` file)

4. Install the packages by running the following command:

```bash 
source ./activate.sh
```

Now all the required packages will be installed under `venv` directory. 


4. Enter the `HPC_scripts` directory:

```bash 
cd ../../HPC_scripts
```

Under this directory, we have prepared several runscript templates that can be used to submit your jobs to the JUWELS Booster and HDF-ML systems, which are operated at the Jülich Supercomputing Center.

The runscript with name convention `run_script_<mode>_<model_name>_<variable>_<system>.sh`
*  `mode`: `train` or `test`, which are used for training or postprocessing.
*  `model_name`: architecture name. e.g., unet, swinir,swinunet, diffusion etc.
*  `variable`: `precip` or `temp`, which indicate precipitation or temperature as target variable.
*  `system`: `hdfml` or `booster`, which indicate  HDFML or JUWELS Booster systems.


Configure the argument for each runscript and submit it by running:

```bash 
sbatch `run_script_<mode>_<model_name>_<variable>_<system>.sh`
```

5. The prediction are generated by run `run_script_tes_<model_name>_<variable>_<system>`. The data are saved monthly in `netCDF` file.  The evaluation and analysis of the prediciton is realised by Jupyter Notebook (see example under JupyterNotebook directory)

6. The logging files for training are saved to the `wandb` directory under the output_dirctory that the user defined in the runscript. To sync the log to the wandb server, you need to activate your envionrment in terminal and run the following command:

```bash 
source ../../sc_venv_template/venv/bin/activate
wandb sync YOUR_WANDB_ID
```

## License and Acknowledgement

This project is realised under the MIT lisence.  This code based on the [swin Transformer](https://github.com/microsoft/Swin-Transformer), [swinIR](https://github.com/JingyunLiang/SwinIR) [swinUnet](https://github.com/HuCaoFighting/Swin-Unet), [SR3](https://iterative-refinement.github.io/). Please also follow their license. 
This project is supported by [MAELSTROM project](https://www.maelstrom-eurohpc.eu/), which is funded from the European High-Performance Computing Joint Undertaking (JU) under grant agreement No 955513

## Contributors

   * Bing Gong: b.gong@fz-juelich.de
   * Yan Ji: y.ji@fz-juelich.de
   * Maxim Bragilovski: maximbr@post.bgu.ac.il
   * Michael Langguth: m.langguth@fz-juelich.de