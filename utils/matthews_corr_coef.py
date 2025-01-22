import torch

def matthews_corr_coef(predictions, ground_truth, threshold=0.5):
        
    # binarize predictions using custom threshold
    bin_predictions = (predictions > threshold).float()

    # calculate true positives, true negatives, false positives and false negatives
    tp = torch.sum((ground_truth == 1) & (bin_predictions == 1)).float()
    tn = torch.sum((ground_truth == 0) & (bin_predictions == 0)).float()
    fp = torch.sum((ground_truth == 0) & (bin_predictions == 1)).float()
    fn = torch.sum((ground_truth == 1) & (bin_predictions == 0)).float()

    #print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    # mcc formula
    numerator = tp * tn - fp * fn
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    if denominator < 1e-8:  # avoid div by zero 
            return torch.tensor(0.0, device=predictions.device)

    mcc = (numerator / denominator)

    return mcc