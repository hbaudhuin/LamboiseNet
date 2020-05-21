from torch import *
import torch.nn as nn
import torch
from torch.nn import *
import torch.nn.functional as F
import numpy as np

"""
This file contains the class defining the standard UNET and classes defining the buidling blocks of the UNET.
These building blocks are also used by the other Nets.
"""


class BasicUnet(nn.Module):
    """
    The class BasicUnet extends the class Module of pytorch which is used to define Neural Networks.
    We redefine two functions which define the architecture and the flow of data through the architecture.
    """

    def __init__(self, n_channels, n_classes):
        """
        Initialises a BasicUnet object by definint all the layers it contains. The weight are randomly initialised.
        The BasicUnet object will take n_channels as input channels and return n_classes feature maps.
        n_channels : number of input channels (number of 2D matrixes)
        n_classes : number of classes to be detected ( 2 here, bulding and no building)
        """
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
        """
        Describes the flow of data through the layers.
        :param x: the input to the neural net of size n_channels x 650 x 650
        :return: the output after x has passed through the Net of size n_classes x 650 x 650
        """
        down1 = self.input_layer(x)
        down2 = self.downscaling_layer1(down1)
        down3 = self.downscaling_layer2(down2)
        down4 = self.downscaling_layer3(down3)
        bottleneck = self.bottleneck(down4)
        concat = self.crop_and_cat(bottleneck, down4)
        up4 = self.upscaling_layer1(concat)
        concat2 = self.crop_and_cat(up4, down3)
        up3 = self.upscaling_layer2(concat2)
        concat3 = self.crop_and_cat(up3, down2)
        up2 = self.upscaling_layer3(concat3)
        concat4 = self.crop_and_cat(up2, down1)
        up1 = self.output_layer(concat4)
        return up1

    def crop_and_cat(self, x1, x2):
        """
        Helper function to concatenante x1 and x2 on the channels dimension and pad of one pixel if there is
        a size difference.
        :param x1, x2:  two matrixes with the same number of channels and up to 1 pixel of difference in size
        :return: a concatenanted version of x1 and x2  on the channels axis.
        """
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return torch.cat([x1, x2], 1)


#######################################
###### BUILDING MODULES ###############
#######################################

class DoubleConvolutionLayer(nn.Module):
    """
    The DoubleConvolutionLayer is an module extending torch Module. It is a module that contains a layer that applies
    Conv2d -> Batchnorm -> ReLU -> Conv2d -> Batchnorm -> ReLU.
    It changes the number of channels of the input but not it's size.
    """
    def __init__(self, n_channels_input, n_channels_output):
        """
        Initialises a DoubleConvolutionLayer object containing one layer that procedes to the operations described
        above sequentially.
        :param n_channels_input: number of channels in input
        :param n_channels_output: number of channels in output
        """
        super(DoubleConvolutionLayer, self).__init__()
        self.double_layer = nn.Sequential(nn.Conv2d(n_channels_input, n_channels_output, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(n_channels_output),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(n_channels_output, n_channels_output, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(n_channels_output),
                                          nn.ReLU(inplace=True))

    def forward(self, x):
        """
        Defines the flow of data in the DoubleConvolution object.
        :param x: the input data given through the layer with n_channels_input channels
        :return: x after passing through the layer with now n_channels_output channels.
        """
        x = self.double_layer(x)
        return x


class Downscaling_layer(nn.Module):
    """
    Defines a downscaling layer object extending the torch Module. A downscaling layer takes an input, passes it through
    a Maxpool Layer on the 2cn and third dimension to divide them by half and then through a DoubleConvolutionLayer.
    """
    def __init__(self, input_channels, output_channels):
        """
        Initialise a Downscaling_layer that takes input_channels as number of input channels and produces an output
        with the desired output_channels.
        We only define one layer which applies the maxPool and the DoubleConvolutionLayer sequentially.
        :param input_channels: number of input channels
        :param output_channels: number of output channels
        """
        super(Downscaling_layer, self).__init__()
        self.layer = nn.Sequential(nn.MaxPool2d(2),
                                   DoubleConvolutionLayer(input_channels, output_channels))

    def forward(self, x):
        """
        Describes the flow of data through the layer. Because there is only one layer, it simply goes through it and we
        returns the output.
        :param x: Input data given to the layer with input_channels channels.
        :return: x after it went through the layer with now output_channels channels.
        """
        x = self.layer(x)
        return x

class ExpandingLayer(nn.Module):
    """
    Expanding layer extends torch Modules. It defines the object Expanding layer which represents an expanding layer
    found on the expanding branch on the UNET.
    It applies a DoubleConvolutional layer which impact the number of channels and then a Transposed convolution to
    augment the size of the input.
    """
    def __init__(self, input_channels, middle_channels, output_channels):
        """
        Initialises a ExpandingLayer object. with two layers : one DoubleconvolutionalLayer and a transposed convolution
        one.
        :param input_channels: number of input channels
        :param middle_channels: number of channels after the doubleConvolutionalLayer
        :param output_channels: number of output channels after the transposed convolution and thus final number of
                channels
        """
        super(ExpandingLayer, self).__init__()
        self.conv = DoubleConvolutionLayer(input_channels, middle_channels)
        self.downscaling = nn.ConvTranspose2d(in_channels=middle_channels, out_channels=output_channels,
                                              kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x1):
        """
        Defines the flow of data through an ExpandingLayer. We first send it through the DoubleConvolution and then
         the transposed layer.
        :param x1: input matrix of the layer with input_channels channels
        :return: x1 after it went through the layer with now a size 2 times bigger and output_channels channels
        """
        x1 = self.conv(x1)
        x1 = self.downscaling(x1)
        return x1


class Bottleneck(nn.Module):
    """
    Defines a Bottleneck Module that extends torch Module. The bottleneck layer is the layer at the bottom of the UNET.
    It applies MaxPool -> DoubleConvolutionalLayer -> transposed convolution.
    """
    def __init__(self, input_channels, middle_channels, output_channels):
        """
        Initialises a BottleNeck object wich contains one layer applying the different operations sequentially.
        :param input_channels: number of channels of the input
        :param middle_channels: number of channels after the double convolution layer
        :param output_channels: number of channels after the transposed convolution
        """
        super(Bottleneck, self).__init__()
        self.layer = nn.Sequential(nn.MaxPool2d(2),
                                   DoubleConvolutionLayer(input_channels, middle_channels),
                                   nn.ConvTranspose2d(in_channels=middle_channels, out_channels=output_channels,
                                                      kernel_size=3, stride=2, padding=1, output_padding=1))

    def forward(self, x):
        """
        Defines the flow of x through the layer.
        :param x: input matrix given to the layer
        :return: x after passing through the layer
        """
        x = self.layer(x)
        return x


class FinalLayer(nn.Module):
    """
    The FinalLayer class extends torch Module. It applies DoubleconvolutionalLayer -> Conv2d -> Batchnorm -> sigmoid.
    It outputs the final two feature maps.
    """
    def __init__(self, input_channels, middle_channels, output_channels):
        """
        Inititalises a FinalLayer object containing one layer applying the operations described above sequentially.
        :param input_channels: number of channels of the input.
        :param middle_channels: number of channels after the DoubleConvolution
        :param output_channels: number of channels after the conv2d (here 2 because we have 2 classes)
        """
        super(FinalLayer, self).__init__()

        self.conv = nn.Sequential(DoubleConvolutionLayer(input_channels, middle_channels),
                                  nn.Conv2d(middle_channels, output_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(output_channels),
                                  nn.Sigmoid())

    def forward(self, x):
        """
        Defines the flow of x through the layer.
        :param x: input matrix given to the layer
        :return: x after passing through the layer
        """
        x = self.conv(x)
        return x
