from torch import *
import torch.nn as nn
import torch
from torch.nn import *
import torch.nn.functional as F

class ResNet(nn.Module) :
    def __init__(self):
        super( ResNet, self).__init__()





class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResidualBlock).__init__()
        self.take_shortcut = input_channels == output_channels
        #simulates an identity block
        self.block = nn.Sequential()
        self.activation = nn.ReLU()
        self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        if self.take_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x


class ResNetResBlocks(ResidualBlock):
    def __init__(self, input_channels, output_channels, expansionCoef, downsampling, convType):
        super(input_channels,output_channels).__init__()
        expanded_channels = output_channels *expansionCoef
        self.take_shortcut = input_channels != expanded_channels
        self.shortcut = nn.Sequential(
                                nn.Conv2d(self.input_channels, expanded_channels, kernel_size=1,
                                          stride=self.downsampling, bias=False),
                                nn.BatchNorm2d(self.expanded_channels)) if self.take_shortcut else None

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(conv(in_channels, out_channels, *args, **kwargs),
                                nn.BatchNorm2d(out_channels))

class ResNetBasicBlocks(ResidualBlock) :
    def __init__(self, input_channels, output_channels):
        super().__init__(input_channels, output_channels)
        self.blocks = nn.Sequential(
            nn.Sequential(conv(input_channels, output_channels), nn.BatchNorm2d(output_channels)),
                            conv_bn(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
                            nn.ReLU,
                            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False)
        )
class ResNetBottleNeckBlock(ResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             nn.ReLU,
             conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             nn.ReLU,
             conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1))


