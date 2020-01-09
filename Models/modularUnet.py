import torch.nn as nn
import torch
import torch.nn.functional as F
from Models.basicUnet import Downscaling_layer, ExpandingLayer,DoubleConvolutionLayer,FinalLayer,Bottleneck

class modularUnet(nn.Module):
    def __init__(self, n_channels, n_classes, depth):
        super(modularUnet,self).__init__()
        self.name = "UNet with depth of "+str(depth)
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.input_layer = DoubleConvolutionLayer(n_channels, 64)
        self.upscaling_layers = []
        self.downscaling_layers = []
        for i in range(1,depth):
            layer_depths = (2**(i-1))*64
            self.downscaling_layers.append(Downscaling_layer(layer_depths, layer_depths*2))
            self.upscaling_layers.append(ExpandingLayer(layer_depths*4, layer_depths*2, layer_depths))
        self.upscaling_layers.reverse()
        self.bottleneck = Bottleneck(64*(2**(depth-1)), 64*(2**depth), 64*(2**(depth-1)))
        self.output_layer = FinalLayer(128, 64, n_classes)

    def forward(self, x):
        downscaling_res = []
        x = self.input_layer(x)
        for layer in self.downscaling_layers :
            downscaling_res.append(x)
            x = layer(x)
        downscaling_res.append(x)
        bottleneck_result = self.bottleneck(x)
        cat = self.crop_and_cat(bottleneck_result, downscaling_res[-1])
        for i,layer in enumerate(self.upscaling_layers) :
            print(cat.shape)
            res = layer(cat)
            cat = self.crop_and_cat(res,downscaling_res[-(i+2)] )
        result = self.output_layer(cat)

        return result

    def crop_and_cat(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        #print('sizes',x1.size(),x2.size(),diffX // 2, diffX - diffX//2, diffY // 2, diffY - diffY//2)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return torch.cat([x1, x2], 1)
