import torch.nn as nn
import torch
import torch.nn.functional as F
from Models.basicUnet import Downscaling_layer, DoubleConvolutionLayer, FinalLayer


class lightUnetPlusPlus(nn.Module):
    """
    The class lightUnetPlusPlus extends torch Module. The main difference with the unetPlusPlus has fewer layers.
    """
    def __init__(self, n_channels, n_classes):
        """
        Initialises a lightUnetPlusPlus object.
        :param n_channels: number of channels in the input
        :param n_classes: number of classes to detect thus also the number of output feature maps
        """
        super(lightUnetPlusPlus, self).__init__()

        self.name = "light Unet++"

        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        basis_filter = 32

        filter_sizes = [basis_filter, basis_filter * 2, basis_filter * 4, basis_filter* 8, basis_filter * 16]

        self.layer0_0 = DoubleConvolutionLayer(n_channels, filter_sizes[0])
        self.layer1_0 = Downscaling_layer(filter_sizes[0], filter_sizes[1])
        self.layer2_0 = Downscaling_layer(filter_sizes[1], filter_sizes[2])
        self.layer3_0 = Downscaling_layer(filter_sizes[2], filter_sizes[3])
        self.layer3_1 = DoubleConvolutionLayer(filter_sizes[3], filter_sizes[3])
        self.layer0_2 = DoubleConvolutionLayer(filter_sizes[0] + filter_sizes[1], filter_sizes[0])
        self.layer1_2 = DoubleConvolutionLayer(filter_sizes[1] + filter_sizes[2], filter_sizes[1])
        self.layer2_2 = DoubleConvolutionLayer(filter_sizes[2] + filter_sizes[3], filter_sizes[2])
        self.layer0_3 = DoubleConvolutionLayer(filter_sizes[0] * 2 + filter_sizes[1], filter_sizes[0])
        self.layer1_3 = DoubleConvolutionLayer(filter_sizes[1] * 2 + filter_sizes[2], filter_sizes[1])
        self.layer0_4 = DoubleConvolutionLayer(filter_sizes[0] * 3 + filter_sizes[1], filter_sizes[0])
        self.final = FinalLayer(filter_sizes[0], n_classes,n_classes)

    def forward(self, input):
        """
        Describe the flow of the input given to the UnetPlusPlus.
        :param input: input matrix given to the unetPlusPlus
        :return: the input after passing through the Net.
        """
        x0_0 = self.layer0_0(input)

        x1_0 = self.layer1_0(x0_0)
        x2_0 = self.layer2_0(x1_0)
        x3_0 = self.layer3_0(x2_0)

        x3_1 = self.layer3_1(x3_0)

        x0_2 = self.layer0_2(self.multiple_cat([self.upsampling(x1_0),x0_0]))
        x1_2 = self.layer1_2(self.multiple_cat([self.upsampling(x2_0),x1_0]))
        x2_2 = self.layer2_2(self.multiple_cat([self.upsampling(x3_1),x2_0]))

        x0_3 = self.layer0_3(self.multiple_cat([x0_0, x0_2, self.upsampling(x1_2)]))
        x1_3 = self.layer1_3(self.multiple_cat([self.upsampling(x2_2), x1_0, x1_2]))

        x0_4 = self.layer0_4(self.multiple_cat([self.upsampling(x1_3), x0_0, x0_2, x0_3]))

        output = self.final(x0_4)

        return output

    def multiple_cat(self, array):
        """
        Helper function to concatenate an array of matrixes
        :array: array of matrixes to contatenate on the chanels axis
        """
        to_be_cat = array[0]
        for i in range(1, len(array)):
            to_be_cat = self.crop_and_cat(to_be_cat, array[i])
        return to_be_cat

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
