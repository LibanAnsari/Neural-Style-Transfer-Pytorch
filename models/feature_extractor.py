from torch import nn

class VGGFeatures(nn.Module):
    def __init__(self, vgg: nn.Sequential):
        super().__init__()
        
        self.vgg = vgg
        
        self.layer_names = {
            0  : 'conv1_1',
            5  : 'conv2_1',
            10 : 'conv3_1',
            19 : 'conv4_1',
            21 : 'conv4_2',
            28 : 'conv5_1',
        }
        
    def forward(self, x):
        # x is an Image
        features = {}
        for i, layer in enumerate(self.vgg): # Manually pass the image through every layer while storing thier outputs
            x = layer(x)
            if i in self.layer_names:
                features[self.layer_names[i]] = x
        return features