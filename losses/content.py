import torch
from torch import nn
from torch.nn import functional as F

class ContentLoss(nn.Module):
    def __init__(self, target: torch.Tensor): # target is the original image activations through a single layer
        super().__init__()
        
        self.target = target.detach() # detach its value so its not a part of the graph
        self.loss = 0.0
        
    def forward(self, x):
        # x is the generated image
        self.loss = F.mse_loss(x, self.target)
    
        return x