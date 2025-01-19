# model design: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
# code implementation from: https://amaarora.github.io/posts/2020-09-13-unet.html

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


""" Block consisting of 2x(convolution, batch normalization, relu) """
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)
        


""" Input Block, just the first Block"""
class InputBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = Block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x



""" Encoder Blocks after Input Block, which are (max pool, Block)"""
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), Block(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x

    
    
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:         # bilinear interpolation
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        self.conv = Block(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW (channels x height x width)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)   



class OutputBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = torch.sigmoid(self.conv(x))
        return x
    
  

""" Complete U-Net: InputBlock + EncoderBlocks + DecoderBlocks + OutputBlock """
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.inp = InputBlock(n_channels, 64)
        self.down1 = EncoderBlock(64, 128)
        self.down2 = EncoderBlock(128, 256)
        self.down3 = EncoderBlock(256, 512)
        self.down4 = EncoderBlock(512, 512)
        self.up1 = DecoderBlock(1024, 256, False)
        self.up2 = DecoderBlock(512, 128, False)
        self.up3 = DecoderBlock(256, 64, False)
        self.up4 = DecoderBlock(128, 64, False)
        self.out = OutputBlock(64, n_classes)

    def get_arch(self):
        return "UNet"
    
    def forward(self, x):
        x1 = self.inp(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x 