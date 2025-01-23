__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-07-13"

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from functools import reduce
from operator import __add__


class Conv2dSamePadding(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)


class Upsampling(nn.Module):

    def __init__(self, in_channels:int = None, out_channels: int = None,
                 kernel_size: int = 4, padding: int = 1, stride: int = 2,
                 upsampling: bool = True, sf: int = 10, mode: str = "bicubic"):
        super().__init__()
        """
        This block is used for transposed low-resolution to the same dim as high-resolution before performing UNet
        Note: The input data is assumed to be of the form minibatch x channels x [optional depth] x [optional height] x width.
        :param in_channels : the number of input variables  
        :param out_channels: the output channels for each ConvTranspose2D layer
        :param kernel_size : the kernel size
        :param padding     : the padding size
        :param stride      : the stride size
        :param sf          : the scaling factor (low-resolution to high resolution)
        :param upsampling  : use upsampling (True) to convert low to high resolution  or use transposed convolutional approach(False)
        """

        if upsampling:
            self.deconv_block = nn.Upsample(scale_factor = sf, mode = mode, align_corners = True)
        else:
            layers = [nn.ConvTranspose2d(in_channels, in_channels, kernel_size = kernel_size, stride = stride,
                                         padding = padding) for i in range(3)]
            layers.append(nn.ConvTranspose2d(in_channels, in_channels, kernel_size = kernel_size, stride = 1,
                                   padding = padding,  ))
            self.deconv_block = nn.Sequential(*layers)

    def forward(self, x:Tensor)->Tensor:
         return self.deconv_block(x)



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
        #add noise
        shape = x.shape
        torch.manual_seed(10)
        noise = torch.randn(shape).to(x.device)
        x0 = torch.cat([noise, x], dim=1)
        x1 = self.conv_block1(x0) #16, 8, 160, 160
        m1 = self.maxpool_conv(x1)
        x2 = self.conv_block2(m1)
        m2 = self.maxpool_conv(x2)
        x3 = self.conv_block3(m2)
        m3 = self.maxpool_conv(x3)
        
        return m3

class Conv_Block(nn.Module):

    def __init__(self, in_channels: int = None, out_channels: int = None,
                 kernel_size: int = 3, padding: str = "same", bias=True):
        """
        The convolutional block consists of one convolutional layer, bach normalization and activation function
        :param in_channels : the number of input channels
        :param out_channels: the number of output channels
        :param kernel_size : the kernel size
        :param padding     : the techniques for padding, either 'same' or 'valid' or integer
        """
        super().__init__()
        self.conv_block = nn.Sequential(
            Conv2dSamePadding(in_channels, out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_block(x)


class Conv_Block_N(nn.Module):

    def __init__(self, in_channels: int = None, out_channels: int = None,
                 kernel_size: int = 3, padding: str = "same", n: int = 2):
        """

        :param in_channels : the number of input channels
        :param out_channels: the number of output channels
        :param kernel_size : the kernel size
        :param padding     : the techniques for padding, either 'same' or 'valid' or integer
        :param n           : the number of convolutional block
        """
        super().__init__()
        n_layers = nn.ModuleList([Conv_Block(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)])
        for i in np.arange(n - 1):
            print("i", i)
            n_layers.append(Conv_Block(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True))

        self.conv_block_n = nn.Sequential(*n_layers)

    def forward(self, x):
        return self.conv_block_n(x)


class Encoder_Block(nn.Module):
    """Downscaling with maxpool then convol block"""

    def __init__(self, in_channels: int = None, out_channels: int = None, kernel_maxpool: int = 2,
                 l_large: bool = True):
        """
        One complete encoder-block used in U-net.
        :param in_channels   : the number of input channels
        :param out_channels  : the number of ouput channels
        :param kernel_maxpool: the number of kernel size
        :param l_large       : flag for large encoder (n consecutive convolutional block)
        """
        super().__init__()

        if l_large:
            self.layer1 = Conv_Block_N(in_channels, out_channels, n=8)
        else:
            self.layer1 = Conv_Block_N(in_channels, out_channels, n=1)

        self.maxpool_conv = nn.MaxPool2d(kernel_maxpool)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        e = self.maxpool_conv(x)
        return x, e


class Decode_Block(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int = None, out_channels: int = None, kernel_size: int = 2,
                 stride_up: int = 2, padding: str = "same"):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride_up, padding=0)
        self.conv = Conv_Block_N(in_channels, out_channels, n=2, kernel_size=(3, 3), padding=padding)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)  # 480
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, channels_start: int = 56, dataset_type: str = 'precipitation'):
        super(UNet, self).__init__()
        self.dataset_type = dataset_type
        
        self.upsampling = Upsampling(n_channels, channels_start,upsampling=True)

        """encoder """
        self.down1 = Encoder_Block(n_channels+1, channels_start)
        self.down2 = Encoder_Block(channels_start, channels_start * 2, l_large=False)
        self.down3 = Encoder_Block(channels_start * 2, channels_start * 4, l_large=False)

        """ bridge encoder <-> decoder """
        self.b1 = Conv_Block(channels_start * 4, channels_start * 8)
        
        #Topography encoder 
        self.top = Conv_top(in_channels=2, out_channels=8, kernel_size=3, bias=True)


        """decoder """
        # +++32
        self.up1 = Decode_Block(channels_start * 8, channels_start * 4)
        self.up2 = Decode_Block(channels_start * 4, channels_start * 2)
        self.up3 = Decode_Block(channels_start * 2, channels_start)
        self.output = nn.Conv2d(channels_start, 1, kernel_size=1, bias=True)
        torch.nn.init.xavier_uniform(self.output.weight)


    def forward(self, x: Tensor, topography: Tensor) -> Tensor:
        # x = x.cuda()
        # print("input shape",x.shape)
        # if self.dataset_type == 'precipitation':
        #     x = self.upsampling(x)
        #     print("x shape:",x.shape)
        # print("x shape:",x.shape)
        # remove top  
        # topography = nn.functional.interpolate(topography, scale_factor=1)
        top = nn.functional.interpolate(topography, scale_factor=0.1, mode='bicubic')
        x = torch.cat((x, top), 1)
        # print("x shape:",x.shape)

        s1, e1 = self.down1(x)
        # print("e1 shape:", e1.shape)
        s2, e2 = self.down2(e1)
        # print("e2 shape:", e2.shape)
        s3, e3 = self.down3(e2)
        # print("e3 shape:", e3.shape)
        x4 = self.b1(e3)  # -1,448,2,2
        # print("x4 shape:", x4.shape)
        # top = self.top(topography) # -1,32,2,2
        # print("top shape:",top.shape)
        # #add the topograph to the neural network

        # x5 = torch.cat((x4, top), 1) #-1，480， 20，20
        # print("x5 shape:",x5.shape)
        # remove top
        x5 = x4
        d1 = self.up1(x5, s3)
        # print("d1 shape:", d1.shape)
        d2 = self.up2(d1, s2)
        # print("d2 shape:", d2.shape)
        d3 = self.up3(d2, s1)
        # print("d3 shape:", d3.shape)
        output = self.output(d3) 
        if self.dataset_type == 'precipitation':
            output_last = self.upsampling(output)

        return output_last


# net = UNet(n_channels = 10)

# x = torch.rand((24,10,16,16))
# top = torch.rand((24,1,160,160))

# pred = net(x,top)
# print(pred.shape)
'''
--train_dir /cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/yzy/downscaling_precipitation/precip_dataset_new/train --val_dir /cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/yzy/downscaling_precipitation/precip_dataset_new/val/2017-01 --save_dir /cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/ShiXiSheng/yzy/zx_results/ex6 --model_type fc --epochs 50
'''