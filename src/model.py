import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class CNN(nn.Module):
    def __init__(self, pretrained=True):
        super(CNN, self).__init__()
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            self.network = resnet50(weights=weights)
        else:
            self.network = resnet50()
        
        #final layer should return a single logit: 0/negative -> real, highly positive -> fake
        in_feat = self.network.fc.in_features
        self.network.fc = nn.Linear(in_feat, 1) 

    def forward(self, x):
        return self.network(x) 
    
    def get_final_conv_layer(self):
        #gradCAM needs the final convolutional layer
        return self.network.layer4[-1]