import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.models as models

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
    

class DeepfakeViTWrapper(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepfakeViTWrapper, self).__init__()
        
        # 1. Load the Base Vision Transformer (16x16 patch size, 224x224 input image size)
        if pretrained:
            # Using modern weights initialization pattern
            weights = models.ViT_B_16_Weights.DEFAULT
            self.network = models.vit_b_16(weights=weights)
            print("🚀 Loaded Vision Transformer with ImageNet pre-trained weights.")
        else:
            self.network = models.vit_b_16(weights=None)
            print("🧱 Loaded raw Vision Transformer architecture (No pre-training).")
            
        # 2. Modify the classification head ('heads.head') for binary classification
        # The hidden feature dimension for vit_b_16 is 768.
        in_features = self.network.heads.head.in_features
        
        # Output 2 values: [Probability of Fake, Probability of Real]
        self.network.heads.head = nn.Linear(in_features, 2)

    def forward(self, x):
        # Simply pass input through the modified backbone network
        return self.network(x)