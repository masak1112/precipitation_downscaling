# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT


__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-11-06"


import math
import torch
from torch import nn,Tensor
from einops import rearrange
import random
import numpy as np
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"

def exists(x):
    return x is not None

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0

    #source of embedding https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/master/model/sr3_modules/unet.py
    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


### Building blocks
class Conv_Block(nn.Module):

    def __init__(self, in_channels:int = None, out_channels: int = None,
                 kernel_size: int = 3, bias=True, time_emb_dim=None):
        """
        The convolutional block consists of one convolutional layer, bach normalization and activation function
        :param in_channels : the number of input channels
        :param out_channels: the number of output channels
        :param kernel_size : the kernel size
        :param padding     : the techniques for padding, either 'same' or 'valid' or integer
        """
        super().__init__()

        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, in_channels))
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same", bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor, time_emb: Tensor = None)->Tensor:
        #print("input in Conv_block shape is", x.shape) #the fist  conv block shape is: ([16, 1, 160, 160])
        #print("time_embd in Conv_block shape is", time_emb.shape) #([16, 64)]
        condition = self.mlp(time_emb)
        condition = rearrange(condition, "b c -> b c 1 1")
        #print("x after condition", x.shape) # 16, 448, 40, 40
        x = x + condition 
        #print("x shape after add condition",  x.shape)
        return self.conv_block(x)



class Conv_Block_N(nn.Module):

    def __init__(self, in_channels:int = None, out_channels: int = None,
                 kernel_size: int = 3, time_emb_dim: int=None):
        """

        :param in_channels : the number of input channels
        :param out_channels: the number of output channels
        :param kernel_size : the kernel size
        :param padding     : the techniques for padding, either 'same' or 'valid' or integer
        :param n           : the number of convolutional block
        """
        super().__init__()
        self.block1 = Conv_Block(in_channels, out_channels, kernel_size=kernel_size,
                               bias=True, time_emb_dim = time_emb_dim)

        self.block2 = Conv_Block(out_channels, out_channels, kernel_size=kernel_size,
                               bias=True, time_emb_dim = time_emb_dim)

    def forward(self, x, time_emb: Tensor = None):
      
        h = self.block1(x, time_emb)
        h = self.block2(h, time_emb)
        return h


class Encoder_Block(nn.Module):
    """Downscaling with maxpool then convol block"""

    def __init__(self, in_channels: int = None, out_channels: int = None, 
                 kernel_maxpool: int = 2, time_emb_dim=None):
        """
        One complete encoder-block used in U-net.
        :param in_channels   : the number of input channels
        :param out_channels  : the number of ouput channels
        :param kernel_maxpool: the number of kernel size
        :param l_large       : flag for large encoder (n consecutive convolutional block)
        """
        super().__init__()

        self.layer1 = Conv_Block_N(in_channels, out_channels, time_emb_dim=time_emb_dim)

        self.maxpool_conv = nn.MaxPool2d(kernel_maxpool)

    def forward(self, x:Tensor, time_emb: Tensor = None)->Tensor:
        x = self.layer1(x, time_emb)
        e = self.maxpool_conv(x)
        return x, e


class Decode_Block(nn.Module):

    """Upscaling then double conv"""
    def __init__(self, in_channels: int=None, out_channels: int = None, kernel_size: int = 2,
                 stride_up: int = 2,  time_emb_dim=None):

        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size,
                                     stride=stride_up, padding = 0)
        
        self.conv = Conv_Block_N(in_channels, out_channels, kernel_size = kernel_size,
                                time_emb_dim = time_emb_dim)

    def forward(self, x1: Tensor, x2:Tensor, time_emb:Tensor = None)->Tensor:
        
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1) #16, 448, 40, 40
    
        return self.conv(x, time_emb)



class Conv_top(nn.Module):

    def __init__(self, in_channels:int = None, out_channels:int = None, kernel_size: int = 3, bias=True):
        """
        The convolutional block consists of one convolutional layer, bach normalization and activation function
        :param in_channels : the number of input channels
        :param kernel_size : the kernel size
        :param padding     : the techniques for padding, either 'same' or 'valid' or integer
        """
        super().__init__()


        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same", bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv_block2= nn.Sequential(
            nn.Conv2d(out_channels, out_channels*2, kernel_size=kernel_size, padding="same", bias=bias),
            nn.BatchNorm2d(out_channels*2),
            nn.ReLU(inplace=True)
        )


        self.conv_block3= nn.Sequential(
            nn.Conv2d(out_channels*2, out_channels*4, kernel_size=kernel_size, padding="same", bias=bias),
            nn.BatchNorm2d(out_channels*4),
            nn.ReLU(inplace=True)
        )

        self.maxpool_conv = nn.MaxPool2d(2)

    def forward(self, x: Tensor)->Tensor:
        x1 = self.conv_block1(x) #16, 8, 160, 160
        m1 = self.maxpool_conv(x1)
        x2 = self.conv_block2(m1)
        m2 = self.maxpool_conv(x2)
        x3 = self.conv_block3(m2)
        m3 = self.maxpool_conv(x3)
        
        return m3



#########Define Unet neural network for diffusion models ##############

class UNet_diff(nn.Module):
    def __init__(self, img_size: int, n_channels:int, channels_start: int = 56, with_time_emb: bool=True):
        """

        :param img_size        :  the dimension of the images
        :param n_channels      :  the channles of input images
        :param channels_start  :  the output channel number of the first convolution block
        :param with_time_emb   :  the time embedding
        """

        super(UNet_diff, self).__init__()
        # time embeddings
        # The time embedding dims should be the same as the model dims in order to sum up of the two
        if with_time_emb:
            time_dim = img_size * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(img_size),
                nn.Linear(img_size, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None
        

        """encoder """
        self.down1 = Encoder_Block(in_channels = n_channels, out_channels = channels_start, time_emb_dim = time_dim)
        self.down2 = Encoder_Block(in_channels = channels_start, out_channels = channels_start*2, time_emb_dim = time_dim)
        self.down3 = Encoder_Block(in_channels = channels_start*2, out_channels = channels_start*4, time_emb_dim = time_dim)

        """ bridge encoder <-> decoder """
        self.b1 = Conv_Block(channels_start*4, channels_start*8, time_emb_dim = time_dim)

        #Topography encoder 
        self.top = Conv_top(in_channels=1, out_channels=8, kernel_size=3, bias=True)

        """decoder """
        self.up1 = Decode_Block(in_channels = channels_start*8 + 32, out_channels = channels_start*4 +32 , time_emb_dim = time_dim)
        self.up2 = Decode_Block(in_channels = channels_start*4 + 32, out_channels = channels_start*2 +32, time_emb_dim = time_dim)
        self.up3 = Decode_Block(in_channels = channels_start*2 + 32, out_channels = channels_start + 32, time_emb_dim = time_dim)

        self.output = nn.Conv2d(channels_start+32, 1, kernel_size=1, bias=True)
        torch.nn.init.xavier_uniform(self.output .weight)


    def forward(self, x:Tensor,time: Tensor = torch.tensor(-1), topograhpy=None)->Tensor:
        
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        print("t in Unet is ",t)

        s1, e1 = self.down1(x, t)
        s2, e2 = self.down2(e1, t)
        s3, e3 = self.down3(e2, t)
        x4 = self.b1(e3, t)
        top = self.top(topograhpy)
        #add the topograph to the neural network
        x5 = torch.cat((x4, top), 1) #16，480， 20，20

        d1 = self.up1(x5, s3, t)
        d2 = self.up2(d1, s2, t)
        d3 = self.up3(d2, s1, t)
        output = self.output(d3)
        return output



