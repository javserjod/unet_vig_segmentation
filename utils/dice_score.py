import torch

"Returns dice score given probability predictions, ground-truth and a threshold"
def dice_score(predictions, ground_truth, threshold=0.5):
    
    # binarize predictions using custom threshold
    bin_predictions = (predictions > threshold).float()
    
    # intersection between gt and predicted
    intersection = torch.sum(bin_predictions * ground_truth)
    
    # union
    union = torch.sum(bin_predictions) + torch.sum(ground_truth)
    
    # Dice score
    dice = (2. * intersection) / (union + 1e-6)  # +1e-6 to avoid div by zero
    
    return dice 