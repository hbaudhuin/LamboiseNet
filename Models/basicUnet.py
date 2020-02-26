from torch import *
import torch.nn as nn
import torch
from torch.nn import *
import torch.nn.functional as F
import numpy as np

"""Unet 
    """


class BasicUnet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(BasicUnet, self).__init__()

        self.name = "UNet"
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.input_layer = DoubleConvolutionLayer(n_channels, 64)
        self.downscaling_layer1 = Downscaling_layer(64, 128)
        self.downscaling_layer2 = Downscaling_layer(128, 256)
        self.downscaling_layer3 = Downscaling_layer(256, 512)
        self.bottleneck = Bottleneck(512, 1024, 512)
        self.upscaling_layer1 = ExpandingLayer(1024, 512, 256)
        self.upscaling_layer2 = ExpandingLayer(512, 256, 128)
        self.upscaling_layer3 = ExpandingLayer(256, 128, 64)
        self.output_layer = FinalLayer(128, 64, n_classes)



    def forward(self, x):
        down1 = self.input_layer(x)
        down2 = self.downscaling_layer1(down1)
        down3 = self.downscaling_layer2(down2)
        down4 = self.downscaling_layer3(down3)
        bottleneck = self.bottleneck(down4)
        concat = self.crop_and_cat(bottleneck, down4)
        up4 = self.upscaling_layer1(concat)
        concat2 = self.crop_and_cat(up4, down3)
        up3 = self.upscaling_layer2(concat2)
        concat3 = self.crop_and_cat(up3,down2)
        up2 =self.upscaling_layer3(concat3)
        concat4 = self.crop_and_cat(up2, down1)
        up1 = self.output_layer(concat4)
        return up1

    def crop_and_cat(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        #print('sizes',x1.size(),x2.size(),diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return torch.cat([x1, x2], 1)


class Downscaling_layer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Downscaling_layer,self).__init__()
        self.layer = nn.Sequential(nn.MaxPool2d(2),
                              DoubleConvolutionLayer(input_channels, output_channels))
    def forward(self, x):
        x = self.layer(x)
        return x


class DoubleConvolutionLayer(nn.Module):
    def __init__(self, n_channels_input, n_channels_output):
        super(DoubleConvolutionLayer, self).__init__()
        self.double_layer = nn.Sequential(nn.Conv2d(n_channels_input, n_channels_output, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(n_channels_output),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(n_channels_output, n_channels_output, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(n_channels_output),
                                          nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.double_layer(x)
        return x


class ExpandingLayer(nn.Module):
    def __init__(self, input_channels, middle_channels, output_channels):
        super(ExpandingLayer, self).__init__()
        self.conv = DoubleConvolutionLayer(input_channels,middle_channels )
        self.downscaling = nn.ConvTranspose2d(in_channels=middle_channels, out_channels=output_channels ,
                                              kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x1):
        x1 = self.conv(x1)
        x1 = self.downscaling(x1)
        return x1


class Bottleneck(nn.Module):
    def __init__(self, input_channels, middle_channels, output_channels) :
        super(Bottleneck, self).__init__()
        self.layer = nn.Sequential(nn.MaxPool2d(2),
                                   DoubleConvolutionLayer(input_channels, middle_channels),
                                   nn.ConvTranspose2d(in_channels=middle_channels, out_channels=output_channels,
                                                      kernel_size=3, stride=2, padding=1, output_padding=1))
    def forward(self, x):
        x = self.layer(x)
        return x

class FinalLayer(nn.Module) :
    def __init__(self, input_channels, middle_channels,output_channels) :
        super(FinalLayer, self).__init__()

        self.conv = nn.Sequential(DoubleConvolutionLayer(input_channels, middle_channels),
                                  nn.Conv2d(middle_channels, output_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(output_channels),
                                  nn.Sigmoid())
                                  #nn.ReLU(inplace=True))
    def forward(self, x) :
        x = self.conv(x)
        return x
