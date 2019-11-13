from torch import *
import torch.nn as nn
import torch
from torch.nn import *
import torch.nn.functional as F
"""Mini- Unet of 2 downscaling layers and 2 upscaling ones, 
    """


class BasicUnet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(BasicUnet, self).__init__()

        self.name = "UNet"
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.input_layer = DoubleConvolutionLayer(n_channels, 64)
        self.downscaling_layer1 = nn.Sequential(nn.MaxPool2d(2),
                                                DoubleConvolutionLayer(64, 128))
        self.downscaling_layer2 = nn.Sequential(nn.MaxPool2d(2),
                                                DoubleConvolutionLayer(128, 256))
        self.bottleneck =nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        # TODO add padding ?
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upscaling_layer1 = DoubleConvolutionLayer(256, 128)
        self.bottleneck2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,
                                             output_padding=1)
        self.up2 =nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upscaling_layer2 = DoubleConvolutionLayer(128, 64)
        self.output_layer = DoubleConvolutionLayer(64, n_classes)

    def forward(self, x):
        out0 = self.input_layer(x)
        out1 = self.downscaling_layer1(out0)
        out = self.downscaling_layer2(out1)
        bottleneck = self.bottleneck(out)
        out = self.up1(bottleneck)

        out = self.pad(out, out1)
        out = torch.cat([out, out1], dim=1)
        out = self.upscaling_layer1(out)

        out = self.bottleneck2(out)
        out = self.up2(out)
        out = self.pad(out, out0)
        out = torch.cat([out, out0], dim=1)
        out0 = self.upscaling_layer2(out)
        output = self.output_layer(out0)
        return output


    def pad(self, x1, x2) :

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
       # print('sizes', x1.size(), x2.size(), diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

       # print('sizes', x1.size(), x2.size(), diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2)
        return x1


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
