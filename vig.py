import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


''' FFN Module'''
class TwoLayerNN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.layer = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.layer(x) + x



''' Obtain the patches of the image '''
class SimplePatchifier(nn.Module):
    def __init__(self, patch_size=36):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).unfold(1, self.patch_size, self.patch_size)\
            .unfold(2, self.patch_size, self.patch_size).contiguous()\
            .view(B, -1, C, self.patch_size, self.patch_size)
        # now we have tensor [B batch size, N patches, C channels, H, W]
        return x


''' ViG Block: graph creation, Grapher module, FFN'''
class ViGBlock(nn.Module):
    def __init__(self, in_features, num_edges=9, head_num=1):
        super().__init__()
        self.k = num_edges
        self.num_edges = num_edges
        self.in_layer1 = TwoLayerNN(in_features)
        self.out_layer1 = TwoLayerNN(in_features)
        self.droppath1 = nn.Identity()  # DropPath(0)
        self.in_layer2 = TwoLayerNN(in_features, in_features*4)
        self.out_layer2 = TwoLayerNN(in_features, in_features*4)
        self.droppath2 = nn.Identity()  # DropPath(0)
        self.multi_head_fc = nn.Conv1d(
            in_features*2, in_features, 1, 1, groups=head_num)

    def forward(self, x):
        B, N, C = x.shape         # [B, N, C]

        sim = x @ x.transpose(-1, -2)               # sim is [B, N, N] where [b,i,j] is similitude between node i and node j in batch b
        graph = sim.topk(self.k, dim=-1).indices    # choose most similar nodes = neighbours   [B, N, k]  where k are the most similar nodes to node i in batch b
                                                    # it is like connecting the nodes with edges -> creating graph
        
        shortcut = x         # copy of x for residual connection
        
        # FEAT. TRANSF. AND ACTIVATION FUNC. TO AVOID OVER-SMOOTHING 
        x = self.in_layer1(x.view(B * N, -1)).view(B, N, -1)   # first transform of features
        
        # GRAPH CONVOLUTION -------------------------------------------------------------------------------------
        # aggregation
        neibor_features = x[torch.arange(
            B).unsqueeze(-1).expand(-1, N).unsqueeze(-1), graph]        # uses indices in graph to select the k neighbours' C features  -> [B, N, k, C]
        x = torch.stack(
            [x, (neibor_features - x.unsqueeze(-2)).amax(dim=-2)], dim=-1)    # pick max features of aggregation, then concat with x, creating tensor [B, N, C, 2] 

        # update
        # Multi-head
        x = self.multi_head_fc(x.view(B * N, -1, 1)).view(B, N, -1)
        # --------------------------------------------------------------------------------------------------------
        # MORE FEAT. TRANSF. AND ACTIVATION FUNC. TO AVOID OVER-SMOOTHING
        x = self.droppath1(self.out_layer1(
            F.gelu(x).view(B * N, -1)).view(B, N, -1))           # activation function to avoid layer collapse
        
        x = x + shortcut       # residual connection concatenation

        # FFN module. Z in paper (ecuation 7) -------------------------------------------------------------------
        x = self.droppath2(self.out_layer2(
            F.gelu(self.in_layer2(x.view(B * N, -1)))).view(B, N, -1)) + x
        
        return x    # tensor [batch_size, num_patches, out_feature]
    
    
class VGNN(nn.Module):
    # num_patches: number of nodes (patches)
    # num_ViGBlocks: depth of the network
    # num_edges: number of neighbours
    def __init__(self, in_features=3*16*16, out_feature=192, num_patches=196,
                 num_ViGBlocks=12, num_edges=9, head_num=1, patch_size=36):
        super().__init__()

        self.patchifier = SimplePatchifier(patch_size)    

        # self.patch_embedding = TwoLayerNN(in_features)
        self.patch_embedding = nn.Sequential(
            # six linear layers to extract features from patches
            nn.Linear(in_features, out_feature//2),
            nn.BatchNorm1d(out_feature//2),
            nn.GELU(),
            nn.Linear(out_feature//2, out_feature//4),
            nn.BatchNorm1d(out_feature//4),
            nn.GELU(),
            nn.Linear(out_feature//4, out_feature//8),
            nn.BatchNorm1d(out_feature//8),
            nn.GELU(),
            nn.Linear(out_feature//8, out_feature//4),
            nn.BatchNorm1d(out_feature//4),
            nn.GELU(),
            nn.Linear(out_feature//4, out_feature//2),
            nn.BatchNorm1d(out_feature//2),
            nn.GELU(),
            nn.Linear(out_feature//2, out_feature),
            nn.BatchNorm1d(out_feature)
        )
        self.pose_embedding = nn.Parameter(
            torch.rand(num_patches, out_feature))

        self.blocks = nn.Sequential(
            *[ViGBlock(out_feature, num_edges, head_num)
              for _ in range(num_ViGBlocks)])

    def forward(self, x):
        # GRAPH PROCESSING
        x = self.patchifier(x)        # get patches -> tensor [B batch size, N patches, C channels, H, W]
        
        B, N, C, H, W = x.shape
        x = self.patch_embedding(x.view(B * N, -1)).view(B, N, -1)    # get tensor [batch_size, num_patches, out_feature]
        
        x = x + self.pose_embedding     # positional encoding (sum random [0, 1) to every node x feature)

        # FEATURE TRANSFORM
        x = self.blocks(x)

        return x
    

class Classifier(nn.Module):
    def __init__(self, in_features=3*16*16, out_feature=192,
                 num_patches=196, num_ViGBlocks=16, hidden_layer=1024,
                 num_edges=9, head_num=1, n_classes=1, patch_size=36,
                 output_size=512):
        super().__init__()
        # VGNN as BACKBONE
        self.backbone = VGNN(in_features, out_feature,
                             num_patches, num_ViGBlocks,
                             num_edges, head_num, patch_size)
        
        self.num_patches = num_patches
        self.out_feature = out_feature        # numebr of featuers per patch
        self.output_size = output_size        # size of upsampled
        self.h = self.w = int(np.sqrt(self.num_patches))       # get h and w of image where each pixel is a feature

        
        # HEAD
        self.predictor = nn.Sequential(
            # nn.Linear(num_patches*out_feature, hidden_layer),
            # nn.BatchNorm1d(hidden_layer),
            # nn.GELU(),
            # nn.Linear(hidden_layer, num_patches),
            # nn.Sigmoid()  # Para obtener probabilidades entre 0 y 1
        )
        
        from unet import UNet
        self.mlp = nn.Linear(self.out_feature, self.out_feature)           # refine characteristics
        #self.patch_to_image = nn.Unfold(kernel_size=(1, 1))  # Simplificación
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(self.out_feature, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),  # Output 1 channel for mask
            #UNet(self.out_feature+3, n_classes)
        )
        
        self.unet = UNet(1+3, n_classes)
                
        

    def forward(self, x):
        
        orig_x = x.clone()    # copy of original input
        
        x = self.backbone(x)
        B, N, C = x.shape             # [batch size, number patches, features per patch]
        x = self.mlp(x)               # [batch_size, number patches, features_per_patch]
        
        x = x.view(B, C, self.h, self.w)     # [batch size, features per patch, h, w]
                                             # in each batch, now we have one image (h x w) for each feature calculated per patch
        
        x = F.interpolate(x, size=(self.output_size, self.output_size), mode="bilinear")     # increase each feature image from hxw to HxW
        
        x = normalize_to_range(x)
        
        # [batch size, features per patch, H, W]
        x = self.conv_blocks(x)    #process all feature image, return just one mask
        
        
        x = torch.cat((orig_x, x), dim=1)  # Concat along channels original images with their respective feature images
        
        x = self.unet(x)       # [B, 1 channel, H, W]
        
        return x  # Final shape [B, 1, H, W]   
        

    
    # def forward(self, x):
    #     features = self.backbone(x)  # [B, Number of patches, Features per patch (C)]
    #     B, N, C = features.shape
    #     H, W = 512, 512
    #     patch_size = H // int(N**0.5)  # 36, para 196 patches

    #     # Reorganiza patches en la estructura original
    #     # features = features.transpose(1, 2).contiguous().view(B, C, int(H/patch_size), int(W/patch_size))
    #     # upsampled = F.interpolate(features, size=(H, W), mode='bilinear', align_corners=False)

    #     # Aplica la cabeza convolucional para obtener un mapa de segmentación
    #     pred = self.predictor(features)
    #     return pred  # Salida de tamaño [B, 1, H, W]
    
    
    
    
    
    
    
def normalize_to_range(tensor, min_val=0.0, max_val=1.0):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized = (tensor - tensor_min) / (tensor_max - tensor_min)  # Escala al rango [0, 1]
    return normalized * (max_val - min_val) + min_val
