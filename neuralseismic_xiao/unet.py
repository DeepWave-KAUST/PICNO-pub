""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from torch import nn

from .unet_parts import Down, Up, DoubleConv, OutConv


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, hidden_size=20):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, hidden_size)
        self.down1 = Down(hidden_size, hidden_size*2)
        self.down2 = Down(hidden_size*2, hidden_size*4)
        self.down3 = Down(hidden_size*4, hidden_size*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(hidden_size*8, hidden_size*16 // factor)
        self.up1 = Up(hidden_size*16, hidden_size*8 // factor, bilinear)
        self.up2 = Up(hidden_size*8, hidden_size*4 // factor, bilinear)
        self.up3 = Up(hidden_size*4, hidden_size*2 // factor, bilinear)
        self.up4 = Up(hidden_size*2, hidden_size, bilinear)
        self.outc = OutConv(hidden_size, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
