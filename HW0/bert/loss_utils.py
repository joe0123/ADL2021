import torch

def smooth(labels, ratio):
    smoothed_one = torch.ones(labels.shape).to(labels.device) * (1 - ratio)
    smoothed_zero = torch.zeros(labels.shape).to(labels.device) * ratio
    return torch.where(labels == 1, smoothed_one, smoothed_zero) 
