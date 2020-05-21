import torch.nn as nn
import torch
import torch.nn.functional as F
from Models.basicUnet import Downscaling_layer, ExpandingLayer, DoubleConvolutionLayer, FinalLayer, Bottleneck


class modularUnet(nn.Module):
    """
    Describes a modularUnet extending torch Module. A modular Unet has the same architecture than a basicUnet except
    it has a modular depth given in the initialisation. It reuses the same basic building blocks. Only the number of
    channels and layers changes but not the type of layers.
    """
    def __init__(self, n_channels, n_classes, depth):
        """
        Initialyses a modularUnet object.
        :param n_channels: number of input channels of the input matrix given to the modularUnet
        :param n_classes: number of output feature maps desired.
        :param depth: depth desired of the modularUnet. Ex: if length is 2, the layer will have two 2 downscaling layers,
                a bottleneck, 2 expanding layers and an initial and final layer.
        """
        super(modularUnet,self).__init__()
        self.name = "UNet with depth of "+str(depth)
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.input_layer = DoubleConvolutionLayer(n_channels, 64)
        #We define a list of layers
        self.upscaling_layers = nn.ModuleList()
        self.upscaling_layers_tmp = []
        self.downscaling_layers = nn.ModuleList()

        #For each depth we define a downscaling layer and an Expanding layer
        for i in range(1,depth):
            layer_depths = (2**(i-1))*64
            self.downscaling_layers.append(Downscaling_layer(layer_depths, layer_depths*2))
            self.upscaling_layers_tmp.append(ExpandingLayer(layer_depths*4, layer_depths*2, layer_depths))

        # Custom list reverse
        for ilayer in range(len(self.upscaling_layers_tmp)):
            self.upscaling_layers.append(self.upscaling_layers_tmp[len(self.upscaling_layers_tmp)-1-ilayer])

        self.bottleneck = Bottleneck(64*(2**(depth-1)), 64*(2**depth), 64*(2**(depth-1)))
        self.output_layer = FinalLayer(128, 64, n_classes)

    def forward(self, x_):
        """
        Defines the flow of x through the Net. Like the Unet we start with the initial layer and downscaling of layers
        and finish with the upscaling layers and FinalLayer.
        :param x: input matrix given to the layer
        :return: x after passing through the layer
        """
        downscaling_res = []
        x = self.input_layer(x_)
        for layer in self.downscaling_layers:
            downscaling_res.append(x)
            x = layer(x)
        downscaling_res.append(x)
        bottleneck_result = self.bottleneck(x)
        cat = self.crop_and_cat(bottleneck_result, downscaling_res[-1])
        for i,layer in enumerate(self.upscaling_layers) :
            res = layer(cat)
            cat = self.crop_and_cat(res,downscaling_res[-(i+2)] )
        result = self.output_layer(cat)
        return result

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
