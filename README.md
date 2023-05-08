## Precipitation downscaling by deep learning

This repository serves as the official PyTorch implementation for precipitation downscaling using deep learning techniques. Specifically, we investigate several: the Swin transformer (SwinIR) and a combination of U-Net with a Generative Adversarial Network (GAN) and a diffusion component for probabilistic downscaling. Our approach involves mapping short-range forecasts obtained from the Integrated Forecast System (IFS), which are provided on a regular spherical grid with a resolution of IFS=0.1°, to high-resolution observation radar data (RADKLIM) with a resolution of 0.01°. The mapping ratio between these resolutions is a factor of 10.

## Access the dataset
The dataset used in this work is currently not publicly available. If you are interested in accessing the dataset, please contact Bing Gong at b.gong@fz-juelich.de for further information and possible arrangements.

## Installation 

1. Clone this repo by typing the following command in your personal target dirctory:

```bash 
git clone https://gitlab.jsc.fz-juelich.de/esde/machine-learning/precipitation_downscaling.git
```

Now, we will setup the virtual enviornment. Just make sure 'pip' is installed. 

2. Change into this subdirectory after cloning:

```bash 
cd env_setup
```

3. Ceate your virtual environment by running:

```bash 
./install_pkg.sh
```

All the required packages that in the `requirement.txt`  will be installed under `venv_booster`  directory. if you want to custermised your enviornment location. You can revise the `--target`  argument under in `install_pkg.sh` 

4. Enter the `HPC_scripts` directory

```bash 
cd ../HPC_scripts
```

Under this directory, we have prepared several runscript templates that can be used to submit your jobs to the JUWELS Booster and HDF-ML systems, which are operated at the Jülich Supercomputing Center.

The runscript with name convention `run_script_<mode>_<model_name>_<variable>_<system>.sh`
*  `mode`: `train` or `test`, which are used for training or postprocessing
*  `model_name`: architecture name. e.g., unet, swinir,swinunet, diffusion etc.
*  `variable`: `precip` or `temp`, which indicate precipitation or temperature as 
*  `system`: `hdfml` or `booster`, which indicate  HDFML or JUWELS Booster systems 


## License and Acknowledgement

This project is realised under the MIT lisence.  This code based on the [swin Transformer](https://github.com/microsoft/Swin-Transformer), [swinIR](https://github.com/JingyunLiang/SwinIR) [swinUnet](https://github.com/HuCaoFighting/Swin-Unet), [SR3](https://iterative-refinement.github.io/). Please also follow their license. 
This project is supported by [MAELSTROM project](https://www.maelstrom-eurohpc.eu/), which is funded from the European High-Performance Computing Joint Undertaking (JU) under grant agreement No 955513

## Contributors

   * Bing Gong: b.gong@fz-juelich.de
   * Yan Ji: y.ji@fz-juelich.de
   * Maxim Bragilovski: maximbr@post.bgu.ac.il
   * Michael Langguth: m.langguth@fz-juelich.de