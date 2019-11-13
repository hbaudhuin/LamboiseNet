from torch import *

"""Mini- Unet of 2 downscaling layers and 2 upscaling ones, 
    """


class basicUnet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(basicUnet, self).__init__()

        self.name = "UNet"
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.input_layer = nn.Sequential(nn.Conv2d(n_channels, 64, kernels_size=3, padding=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLu(inplace=True),
                                         nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.downscaling_layer1 = nn.Sequential(nn.MaxPool2d(2),
                                                nn.Conv2d(64, 128, kernels_size=3, padding=1),
                                                nn.BatchNorm2d(128),
                                                nn.ReLu(inplace=True),
                                                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(128),
                                                nn.ReLU(inplace=True)
                                                )
        self.downscaling_layer2 = nn.Sequential(nn.MaxPool2d(2),
                                                nn.Conv2d(128, 256, kernels_size=3, padding=1),
                                                nn.BatchNorm2d(256),
                                                nn.ReLu(inplace=True),
                                                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(256),
                                                nn.ReLU(inplace=True)
                                                )
        # TODO add padding ?
        self.upscaling_layer1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                              nn.Conv2d(256, 64, kernels_size=3, padding=1),
                                              nn.BatchNorm2d(64),
                                              nn.ReLu(inplace=True),
                                              nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                              nn.BatchNorm2d(64),
                                              nn.ReLU(inplace=True)
                                              )
        self.upscaling_layer = nn.Sequential(nn.Conv2d(128, 64, kernels_size=3, padding=1),
                                             nn.BatchNorm2d(64),
                                             nn.ReLu(inplace=True),
                                             nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                             nn.BatchNorm2d(64),
                                             nn.ReLU(inplace=True))

        self.output_layer = nn.Sequential(nn.Conv2d(64, n_classes, kernels_size=3, padding=1),
                                          nn.BatchNorm2d(n_classes),
                                          nn.ReLu(inplace=True),
                                          nn.Conv2d(n_classes, n_classes, kernel_size=3, padding=1),
                                          nn.BatchNorm2d(n_classes),
                                          nn.ReLU(inplace=True)
                                          )

    def forward(self, x):
        out0 = self.input_layer(x)
        out1 = self.downscaling_layer1(out0)
        out = self.downscaling_layer2(out1)
        out= self.Upsample(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = cat([out, out1], dim = 1)
        out = self.upscaling_layer1(out)
        out= self.Upsample(out)
        out = cat([out,x], dim =1 )
        out = self.upscaling_layer2(out)
        output = self.output_layer(out)
        return output
