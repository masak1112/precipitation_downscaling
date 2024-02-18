# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)

# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2023-07-12"


def get_data_info(dataset_type, upscale=None, **kwargs):
        # Define parameters for datasets
    patch_size = 16
    if dataset_type=="precipitation":
        if "patch_size" in kwargs:
            patch_size = kwargs["patch_size"]
        print("The patch size for prepitation is", patch_size)
        n_channels = 8
        upscale=4
        img_size=[patch_size, patch_size]
    
    elif dataset_type == "precipitation_correction":
        n_channels = 1
        patch_size = 160
        img_size =  [patch_size,patch_size]

    elif dataset_type == "temperature":
        n_channels = 9
        upscale=1
        img_size=[96,120]
    else:
        raise NotImplementedError
    
    return n_channels, upscale, img_size