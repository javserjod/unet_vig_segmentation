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
        BN, C = x.shape
        #print(f"Before layers: {x.shape}")  # Debug print
        x = self.layer(x) + x
        #print(f"After layers: {x.shape}")  # Debug print
        return x



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
                
        return x   # [B batch size, N patches, C channels, h of patches, w of patches]


''' ViG Block: graph creation, Grapher module, FFN'''
class ViGBlock(nn.Module):
    def __init__(self, in_features, num_edges=9, head_num=1):
        super().__init__()
        self.in_features=in_features
        self.k = num_edges
        self.head_num=head_num
        
        self.in_layer1 = TwoLayerNN(self.in_features)
        self.out_layer1 = TwoLayerNN(self.in_features)
        self.droppath1 = nn.Identity()  # DropPath(0)
        self.in_layer2 = TwoLayerNN(self.in_features, self.in_features*4)
        self.out_layer2 = TwoLayerNN(self.in_features, self.in_features*4)
        self.droppath2 = nn.Identity()  # DropPath(0)
        self.multi_head_fc = nn.Conv1d(
            self.in_features*2, self.in_features, 1, 1, groups=self.head_num)

    def forward(self, x):
        B, N, C = x.shape         # [B, N, C]
        
        # Create the graph
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
        x = self.droppath2(self.out_layer2(F.gelu(self.in_layer2(
            x.view(B * N, -1)))).view(B, N, -1)) + x
        
        return x    # tensor [batch_size, num_patches, out_feature]
    
    
class VGNN(nn.Module):
    # num_patches: number of nodes (patches)
    # num_ViGBlocks: depth of the network
    # num_edges: number of neighbours
    def __init__(self, in_features, out_feature, 
                 num_patches, num_ViGBlocks, 
                 num_edges, head_num, 
                 patch_size):
        
        super().__init__()
        self.in_features=in_features
        self.out_feature = out_feature
        self.num_patches=num_patches
        self.num_ViGBlocks=num_ViGBlocks
        self.num_edges=num_edges
        self.head_num=head_num
        self.patch_size=patch_size

        self.patchifier = SimplePatchifier(self.patch_size)     # get [B batch size, N patches, C channels, H, W]

        # self.patch_embedding = TwoLayerNN(in_features)
        self.patch_embedding = nn.Sequential(
            # six linear layers to extract features from patches
            nn.Linear(self.in_features, self.out_feature//2),
            nn.BatchNorm1d(self.out_feature//2),
            nn.GELU(),
            nn.Linear(self.out_feature//2, self.out_feature//4),
            nn.BatchNorm1d(self.out_feature//4),
            nn.GELU(),
            nn.Linear(self.out_feature//4, self.out_feature//8),
            nn.BatchNorm1d(self.out_feature//8),
            nn.GELU(),
            nn.Linear(self.out_feature//8, self.out_feature//4),
            nn.BatchNorm1d(self.out_feature//4),
            nn.GELU(),
            nn.Linear(self.out_feature//4, self.out_feature//2),
            nn.BatchNorm1d(self.out_feature//2),
            nn.GELU(),
            nn.Linear(self.out_feature//2, self.out_feature),
            nn.BatchNorm1d(self.out_feature)
        )
        self.pose_embedding = nn.Parameter(
            torch.rand(self.num_patches, self.out_feature))

        # self.blocks = nn.Sequential(
        #     *[ViGBlock(self.out_feature, self.num_edges, self.head_num)
        #       for _ in range(self.num_ViGBlocks)])
        
        # increase num of edges linearly when advancing ViG blocks (stop at 18) as original article says
        self.blocks = nn.Sequential(
            *[ViGBlock(self.out_feature, min(self.num_edges + i, 18), self.head_num)
                for i in range(self.num_ViGBlocks)])

    def forward(self, x):
        # Pre-GRAPH PROCESSING
        x = self.patchifier(x)        # get patches -> # [B batch size, N patches, C channels, h of patches, w of patches]
        B, N, C, H, W = x.shape
        
        x = x.view(B*N, C * H * W)   # [B*N_patches, 3 channels*patch_height*patch_width]
        #print(f"Before patch_embedding: {x.shape}")  # Debug print
        x = self.patch_embedding(x)    # get tensor [batch_size, num_patches, out_feature]
        #print(f"After patch_embedding: {x.shape}")  # Debug print
         # Reshape back to [B, N, out_feature]
        x = x.view(B, N, -1)
        x = x + self.pose_embedding     # positional encoding (sum random [0, 1) to every node x feature)

        # FEATURE TRANSFORM (ViG blocks)
        x = self.blocks(x)

        return x    #[batch_size, num_patches, out_feature]
    

class ViGClassifier(nn.Module):
    def __init__(self, in_features, out_feature,
                 num_patches, num_ViGBlocks,
                 num_edges, head_num, 
                 patch_size, output_size):
        
        super().__init__()
        
        self.in_features=in_features
        self.out_feature = out_feature    # number of features per patch
        self.num_patches=num_patches
        self.num_ViGBlocks=num_ViGBlocks
        self.num_edges=num_edges
        self.head_num=head_num
        self.patch_size=patch_size
        self.output_size=output_size      # size of upsampled
        
        self.grid_size= int(np.sqrt(self.num_patches))     # if 196 patches in 512x512 img, then there are 14x14 patches
        
        # VGNN as BACKBONE
        self.backbone = VGNN(self.in_features, self.out_feature,
                             self.num_patches, self.num_ViGBlocks,
                             self.num_edges, self.head_num, 
                             self.patch_size)
        

        # HEAD
        # # DEFAULT PREDICTOR
        # self.predictor = nn.Sequential(
        #     nn.Linear(num_patches*out_feature, hidden_layer),
        #     nn.BatchNorm1d(hidden_layer),
        #     nn.GELU(),
        #     nn.Linear(hidden_layer, num_patches),
        #     nn.Sigmoid()  # probabilities between 0 and 1
        # )
        
        
        
        # Process Vision GNN C features maps
        self.conv1 = nn.Conv2d(self.out_feature, self.out_feature//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.out_feature//2, self.out_feature//4, kernel_size=3, padding=1)
        
        # Output layer, generate just 1 channel
        self.conv_out = nn.Conv2d(self.out_feature//4, 1, kernel_size=3, padding=1)
        self.act = nn.Sigmoid()   # only if vig alone
        
        
    def forward(self, x):

        features = self.backbone(x)  #[batch_size, num_patches, out_feature]
        B, N, C = features.shape

        #x = features.permute(0, 2, 1).contiguous()  # reshape to [batches, features, patches]
        

        x = features.view(B, C, self.grid_size, self.grid_size)
        #print(f"ViG: {x.shape=}")
        

        #Aplicar convoluciones
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Capa final para generar la imagen de salida
        x = self.conv_out(x)  # [B, 1, grid_size, grid_size]
        #print(f"{x.shape=}")   
        
        #x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        # [B, C, H, W]
    
        # mean channels
        #x = x.mean(dim=1, keepdim=True)  # [B, C, H, W] to [B, 1, H, W]
        
        
        x = F.interpolate(x, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)    #[B, 1, H, W]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x.to(device)
        
        x = self.act(x)
        #x = normalize_to_range(x)
        
        return x   #[B, 1, H, W]


    # # DEFAULT
    # def forward(self, x):
  
    #     features = self.backbone(x)  #[batch_size, num_patches, out_feature]
    #     B, N, C = features.shape
    #     x = self.predictor(features.view(B, -1))     # [B, N]
        
    #     # Reshape to spatial grid: [Batch, Grid_Height, Grid_Width]
    #     x = x.view(B, self.grid_size, self.grid_size)  # [Batch, number of patches in x, number of patches in y]
        
    #     # Upsample to target resolution: [Batch, 1, 512, 512]
    #     x = F.interpolate(x.unsqueeze(1), size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
    #     return x
    
    
    
    
    
def normalize_to_range(tensor, min_val=0.0, max_val=1.0):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized = (tensor - tensor_min) / (tensor_max - tensor_min)  # Escala al rango [0, 1]
    return normalized * (max_val - min_val) + min_val
