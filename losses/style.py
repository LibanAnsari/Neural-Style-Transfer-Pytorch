import torch
from torch import nn

class StyleLoss(nn.Module):
    def __init__(self, target: torch.Tensor, weight = 0): # target is the original style layer
        super().__init__()
        
        _, c, h, w = target.shape
        self.N_l = c
        self.M_l = h * w
        
        self.wl = weight
        
        self.target = self.gram_matrix(target).detach() # A_l
        self.loss = 0.0
        
    def forward(self, x):
        # x is the generated image
        G = self.gram_matrix(x) # G_l
        
        E_l = torch.sum((G - self.target) ** 2)
        E_l = 1 / (4 * (self.N_l ** 2) * (self.M_l ** 2)) * E_l
        
        self.loss = self.wl * E_l
    
        return x
    
    def gram_matrix(self, x: torch.Tensor):
        _, c, h, w = x.shape
        
        features = x.view(c, h*w) # F_l ∈ R^{N_l × M_l}, N_l: no of channels, M_l: height * width
        
        G = features @ features.T # G_l ∈ R^{N_l × N_l}
        
        # Gram matrix is independen of image size
        return G
    