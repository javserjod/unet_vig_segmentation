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
        x = self.conv(x)
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
  
  
    
"""
"Encoder: contracting path, left side of the U-Net"
class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


"Decoder: expansive path, right side of the U-net"
class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], kernel_size=2, stride=2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)       # concat 
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):     # crop corresponding encoder output, according to current decoder instance dimensions
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


"Full U-Net, as proposed originally"
class UNet(nn.Module):
    def __init__(self, enc_chs=(3, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sz=(572,572)):
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.conv1x1     = nn.Conv2d(dec_chs[-1], num_class, kernel_size=1, stride=1)
        self.retain_dim  = retain_dim
        self.out_sz      = out_sz

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.conv1x1(out)
        
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz, mode='bilinear', align_corners=True)     # to match U-net output image size to input image size
    
        return out

"""