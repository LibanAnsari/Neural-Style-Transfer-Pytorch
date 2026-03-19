import torch

def tv_loss(x):
    return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
           
           
# Total Variation Loss (Not in the original paper)
# It penalizes rapid intensity changes between neighboring pixels.
# Smoothens the image by acting as a regularizer.