#reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Loss-Function-Reference-for-Keras-&-PyTorch


import torch.nn as nn
import torch.nn.functional as F


''' Dice loss function ''' 
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice
    
    
''' Loss function considering Dice loss and BCE loss '''    
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True, bce_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        
    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        Dice_BCE = self.bce_weight * BCE + dice_loss
        
        return Dice_BCE
    



import scipy.ndimage as ndi
from skimage.measure import label
''' Loss function considering Dice loss, BCE loss and difference between CC '''
class DiceBCELossWithTopology(nn.Module):
    def __init__(self, bce_weight=0.5, topology_weight=1.0, image_pixels=512*512, size_average=True):
        super(DiceBCELossWithTopology, self).__init__()
        self.bce_weight = bce_weight
        self.topology_weight = topology_weight
        self.image_pixels = image_pixels

    def compute_topological_difference(self, pred, target):
        """
        Calcula la diferencia en el número de componentes conexas
        entre la predicción y la etiqueta. 
        """
        pred = pred.detach().cpu().numpy()  # to numpy to use skimage
        target = target.detach().cpu().numpy()

        # label cc (images already binary)
        pred_labels = label(pred)
        #target_labels = label(target)

        # number of cc
        num_pred_components = len(set(pred_labels.flatten())) - 1  # exclude background
        #num_target_components = len(set(target_labels.flatten())) - 1
        num_target_components = 1

        # diference of cc
        return abs(num_pred_components - num_target_components)/self.image_pixels

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        # Calculamos la diferencia topológica
        topo_loss = self.compute_topological_difference(inputs, targets)
        
        # Ponderamos la BCE, Dice y la diferencia topológica
        Dice_BCE_Topology = BCE * self.bce_weight + dice_loss + topo_loss * self.topology_weight
        
        return Dice_BCE_Topology