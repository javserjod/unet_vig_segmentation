import torch

def matthews_corr_coef(predictions, ground_truth, threshold=0.5):
    # binarize predictions using custom threshold
    bin_predictions = (predictions > threshold).float()
    
    # calculate true positives, true negatives, false positives and false negatives
    tp = torch.sum((ground_truth == 1) & (bin_predictions == 1))
    tn = torch.sum((ground_truth == 0) & (bin_predictions == 0))
    fp = torch.sum((ground_truth == 0) & (bin_predictions == 1))
    fn = torch.sum((ground_truth == 1) & (bin_predictions == 0))
    
    # mcc formula
    numerator = tp * tn - fp * fn
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
   
    if denominator == 0:    # avoid divide by zero exception
            return 0.0
        
    mcc = (numerator / denominator).item()
    
    return mcc