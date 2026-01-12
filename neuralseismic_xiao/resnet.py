import torch
from torch import nn, Tensor
from torch.nn.modules import padding
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torchvision.models.resnet import BasicBlock, conv1x1, conv3x3

class Resnet18(nn.Module):
    def __init__(self, in_channels, out_channels, layer_channels=(16, 32, 128, 128, 32, 16), norm_class=BatchNorm2d, activation_fn=None):
        """
        Tiny resnet18 with no downsample.
        You can see the original version by running:
            model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
        """
        super().__init__()
        self.layer_channels = layer_channels
        self.prelayer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.layer_channels[0], 
                kernel_size=(7, 7), padding=(3,3), bias=False), 
            norm_class(num_features=self.layer_channels[0]),
            ReLU(inplace=True),
            )

        self.mid_layers = nn.Sequential(*[
            nn.Sequential(
                BasicBlock(inplanes=self.layer_channels[i-1], planes=self.layer_channels[i], 
                    downsample=nn.Sequential(Conv2d(self.layer_channels[i-1], self.layer_channels[i], kernel_size=(1,1)),
                                            norm_class(self.layer_channels[i]))),
                BasicBlock(inplanes=self.layer_channels[i], planes=self.layer_channels[i])
            ) for i in range(1, len(self.layer_channels))
        ])
        
        self.endlayer = conv1x1(self.layer_channels[-1], out_planes=out_channels)

        
    def forward(self, x:Tensor):

        pre_x = self.prelayer(x)
 
        mid_x = self.mid_layers(pre_x)
        return self.endlayer(mid_x)
