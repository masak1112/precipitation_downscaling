
# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)

# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2023-07-12"


import sys
sys.path.append('../')
from models.network_unet import UNet as unet
from models.network_swinir import SwinIR as swinIR
from models.network_vit import TransformerSR as vitSR
from models.network_swinunet_sys import SwinTransformerSys as swinUnet
from models.network_diffusion  import UNet_diff
from models.diffusion_utils import GaussianDiffusion
from models.network_critic import Discriminator as critic
from models.network_unet2 import UNetModel


def get_model(type_net, dataset_type, img_size, n_channels, upscale, **kwargs):
    netC = None
    # Define the models
    if type_net == "unet":

        netG = unet(n_channels = n_channels,dataset_type=dataset_type)
    elif type_net == "swinir":
        netG = swinIR(img_size=img_size,
                      patch_size=4,
                      in_chans=n_channels,
                      window_size=2,
                      upscale=upscale,
                      upsampler= "pixelshuffle",
                      dataset_type=dataset_type)
    
    elif type_net.lower()  == "vitsr":
        netG = vitSR(embed_dim =768)

    elif type_net == "swinunet":
        netG = swinUnet(img_size=img_size, 
                        patch_size=4, 
                        in_chans=n_channels,
                        num_classes=1, 
                        embed_dim=96, 
                        depths=[2, 2, 2],
                        depths_decoder=[1,2, 2], 
                        num_heads=[6, 6, 6],
                        window_size=5,
                        mlp_ratio=4, 
                        qkv_bias=True, 
                        qk_scale=None,
                        drop_rate=0., 
                        attn_drop_rate=0., 
                        drop_path_rate=0.1,
                        ape=False,
                        final_upsample="expand_first")

    elif type_net == "diffusion":
        # add one channel for the noise
        netG = UNet_diff(img_size=img_size[0],
                         n_channels=n_channels+1)
    elif type_net == "diffusion2":
        netG = UNetModel(image_size = img_size[0],
            in_channels=n_channels+1,
            model_channels= n_channels,
            out_channels=1,
            num_res_blocks=3,
            attention_resolutions=[40,10],
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False)

    elif type_net == "wgan":
        netG = unet(n_channels=n_channels, 
                    dataset_type=dataset_type)
        netC = critic((1, img_size[0], img_size[1]))

    else:
        raise NotImplementedError
    return netG, netC
