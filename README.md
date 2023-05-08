## Precipitation downscaling by deep learning

This repository serves as the official PyTorch implementation for precipitation downscaling using deep learning techniques. Specifically, we investigate several: the Swin transformer (SwinIR) and a combination of U-Net with a Generative Adversarial Network (GAN) and a diffusion component for probabilistic downscaling. Our approach involves mapping short-range forecasts obtained from the Integrated Forecast System (IFS), which are provided on a regular spherical grid with a resolution of IFS=0.1°, to high-resolution observation radar data (RADKLIM) with a resolution of 0.01°. The mapping ratio between these resolutions is a factor of 10.

## Access the dataset
The dataset is not published yet. Please contact Bing Gong (b.gong@fz-juelich.de) for accessing the dataset.

## How to start 


## License and Acknowledgement

This project is realised under the MIT lisence.  This code based on the [swin Transformer](https://github.com/microsoft/Swin-Transformer), [swinIR](https://github.com/JingyunLiang/SwinIR) [swinUnet](https://github.com/HuCaoFighting/Swin-Unet), [SR3](https://iterative-refinement.github.io/). Please also follow their license. 
This project is supported by [MAELSTROM project](https://www.maelstrom-eurohpc.eu/), which is funded from the European High-Performance Computing Joint Undertaking (JU) under grant agreement No 955513

## Contributors

Bing Gong: (b.gong@fz-juelich.de)
