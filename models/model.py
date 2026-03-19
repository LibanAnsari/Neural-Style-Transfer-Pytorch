import torch
from torch import nn
from torchvision.models import vgg19, VGG19_Weights
from models.feature_extractor import VGGFeatures

vgg_weights = VGG19_Weights.DEFAULT

vgg = vgg19(weights=vgg_weights)

vgg = vgg.features # Removing the classifier head

# Freeze the model layers
for param in vgg.parameters():
    param.requires_grad = False
    
# Convert the MaxPool Layers into AvgPool layers as in the paper
for i, layer in enumerate(vgg):
    if(isinstance(layer, nn.MaxPool2d)):
        vgg[i] = nn.AvgPool2d(
            kernel_size=layer.kernel_size,
            padding=layer.padding,
            stride=layer.stride
        )
        
vgg_model = VGGFeatures(vgg=vgg).eval()