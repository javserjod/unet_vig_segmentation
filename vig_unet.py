import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from unet import Block, InputBlock, EncoderBlock, DecoderBlock, OutputBlock
from vig import TwoLayerNN, SimplePatchifier, ViGBlock, VGNN, ViGClassifier, normalize_to_range
    
''' U-Net where the output of a Vision GNN is concatenated before the Decoder path '''
class ViGUNet(nn.Module):
    def __init__(self, n_channels, n_classes, 
                 in_features, out_feature,
                 num_patches, num_ViGBlocks,
                 num_edges, head_num, 
                 patch_size, output_size):
        
        super().__init__()
        
        self.n_channels=n_channels
        self.n_classes=n_classes
        self.in_features=in_features
        self.out_feature = out_feature
        self.num_patches=num_patches
        self.num_ViGBlocks=num_ViGBlocks
        self.num_edges=num_edges
        self.head_num=head_num
        self.patch_size=patch_size
        self.output_size=output_size
        
        self.inp = InputBlock(self.n_channels, 64)
                
        self.down1 = EncoderBlock(64, 128)
        self.down2 = EncoderBlock(128, 256)
        self.down3 = EncoderBlock(256, 512)
        self.down4 = EncoderBlock(512, 1024)
        
        self.up1 = DecoderBlock(1024, 512, False)
        self.up2 = DecoderBlock(512, 256, False)
        self.up3 = DecoderBlock(256, 128, False)
        self.up4 = DecoderBlock(128, 64, False)
        
        #self.out = OutputBlock(64, self.n_classes)
        self.unet_out = nn.Conv2d(64, 1, kernel_size=1)       # no sigmoid, unlike output block
        
        self.vgnn = ViGClassifier(self.in_features, self.out_feature,
                              self.num_patches, self.num_ViGBlocks,
                              self.num_edges, self.head_num, 
                              self.patch_size, self.output_size)
        
        # Conv layers to transform fusion
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)   #############
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        
        # Output layer to return segmentation
        self.conv_out = nn.Conv2d(1, self.n_classes, kernel_size=1)
        self.act = nn.Sigmoid()


    def get_arch(self):
        return "Vig_UNet"    
    
    def forward(self, x):
        
        vgn_output = self.vgnn(x)     # original images through VGGN, get [B, C, H, W]
        
        x1 = self.inp(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)   
        x = self.up4(x, x1)
        
        x = self.unet_out(x)
        
        # Fusion of U-Net and ViG
        #x = torch.cat([x, vgn_output], dim=1)
        x = x * vgn_output
        
        # Conv layers to transform fusion
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Output layer to return segmentation
        x = self.conv_out(x)
        x = self.act(x)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x.to(device)
        
        return x 



# ''' U-Net decoder block, but accepting a concat before the upsampling ''' 
# class DecoderBlockViG(nn.Module):
#     def __init__(self, in_ch, out_ch, bilinear=True, concat_channels = 192):
#         super().__init__()
#         self.in_ch=in_ch
#         self.out_ch=out_ch
#         self.concat_channels=concat_channels
#         if bilinear:         # bilinear interpolation
#             self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(self.in_ch+self.concat_channels, self.in_ch // 2, kernel_size=2, stride=2)
#         self.conv = Block(self.in_ch, self.out_ch)

#     def forward(self, x1, x2, features_img=None):
#         #print(f"{features_img.shape=}")    #[4, 192, 14, 14]
#         features_img = F.interpolate(features_img, size=(x1.shape[2], x1.shape[3]), mode='bilinear', align_corners=False)   # resize hxw to x1 hxw
#         # print(f"{features_img.shape=}")    #[4, 192, 32, 32]
#         # print(f"{x1.shape=}")              #[4, 1024, 32, 32]
#         x1 = torch.cat([x1, features_img], dim=1)
#         #print(f"After concat: {x1.shape=}")   #[4, 1216, 32, 32]
        
#         x1 = self.up(x1)

#         # input is CHW (channels x height x width)
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)  

    
    