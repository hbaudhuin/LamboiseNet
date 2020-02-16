import torch.nn as nn
import torch
import torch.nn.functional as F
from Models.basicUnet import Downscaling_layer, ExpandingLayer, DoubleConvolutionLayer, FinalLayer, Bottleneck


class unetPlusPlus(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(unetPlusPlus, self).__init__()
        self.name = "UNet++"

        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        #filter_sizes = [64, 64 * 2, 64 * 4, 64 * 8, 64 * 16]
        filter_sizes = [64, 8, 8, 8, 8]

        self.layer0_0 = DoubleConvolutionLayer(n_channels, 64)

        self.layer1_0 = Downscaling_layer(filter_sizes[0], filter_sizes[1])
        self.layer2_0 = Downscaling_layer(filter_sizes[1], filter_sizes[2])
        self.layer3_0 = Downscaling_layer(filter_sizes[2], filter_sizes[3])
        self.layer4_0 = Downscaling_layer(filter_sizes[3], filter_sizes[4])

        self.layer0_1 = DoubleConvolutionLayer(filter_sizes[0] + filter_sizes[1], filter_sizes[0])
        self.layer1_1 = DoubleConvolutionLayer(filter_sizes[1] + filter_sizes[2], filter_sizes[1])
        self.layer2_1 = DoubleConvolutionLayer(filter_sizes[2] + filter_sizes[3], filter_sizes[2])
        self.layer3_1 = DoubleConvolutionLayer(filter_sizes[3] + filter_sizes[4], filter_sizes[3])

        self.layer0_2 = DoubleConvolutionLayer(filter_sizes[0] * 2 + filter_sizes[1], filter_sizes[0])
        self.layer1_2 = DoubleConvolutionLayer(filter_sizes[1] * 2 + filter_sizes[2], filter_sizes[1])
        self.layer2_2 = DoubleConvolutionLayer(filter_sizes[2] * 2 + filter_sizes[3], filter_sizes[2])

        self.layer0_3 = DoubleConvolutionLayer(filter_sizes[0] * 3 + filter_sizes[1], filter_sizes[0])
        self.layer1_3 = DoubleConvolutionLayer(filter_sizes[1] * 3 + filter_sizes[2], filter_sizes[1])

        self.layer0_4 = DoubleConvolutionLayer(filter_sizes[0] * 4 + filter_sizes[1], filter_sizes[0])

        self.final = FinalLayer(filter_sizes[0], n_classes, n_classes)

    def forward(self, input):

        x0_0 = self.layer0_0(input)

        x1_0 = self.layer1_0(x0_0)
        x2_0 = self.layer2_0(x1_0)
        x3_0 = self.layer3_0(x2_0)
        x4_0 = self.layer4_0(x3_0)

        x0_1 = self.layer0_1(self.multiple_cat([self.upsampling(x1_0),x0_0]))
        x1_1 = self.layer1_1(self.multiple_cat([ self.upsampling(x2_0),x1_0]))
        x2_1 = self.layer2_1(self.multiple_cat([x2_0, self.upsampling(x3_0)]))
        x3_1 = self.layer3_1(self.multiple_cat([x3_0, self.upsampling(x4_0)]))

        x0_2 = self.layer0_2(self.multiple_cat([x0_1, x0_0, self.upsampling(x1_1)]))
        x1_2 = self.layer1_2(self.multiple_cat([self.upsampling(x2_1),x1_0, x1_1]))
        x2_2 = self.layer2_2(self.multiple_cat([self.upsampling(x3_1), x2_1, x2_0]))

        x0_3 = self.layer0_3(self.multiple_cat([x0_0, x0_1, x0_2, self.upsampling(x1_2)]))
        x1_3 = self.layer1_3(self.multiple_cat([self.upsampling(x2_2), x1_0, x1_1, x1_2]))

        x0_4 = self.layer0_4(self.multiple_cat([self.upsampling(x1_3), x0_0, x0_1, x0_2, x0_3, ]))

        output = self.final(x0_4)

        return output

    def multiple_cat(self, array):
        to_be_cat = array[0]
        for i in range(1, len(array)):
            to_be_cat = self.crop_and_cat(to_be_cat, array[i])
        return to_be_cat

    def crop_and_cat(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        #print('sizes',x1.size(),x2.size(),diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return torch.cat([x1, x2], 1)
