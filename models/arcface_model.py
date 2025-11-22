import torch.nn as nn
from torchvision import models

class ArcFaceResNetBackbone(nn.Module):
    def __init__(self, embedding_size=512):
        super(ArcFaceResNetBackbone, self).__init__()
        # Load ResNet50 Weights
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        self.net = models.resnet50(weights=weights)
        
        # Thay thế lớp FC cuối cùng của ResNet (2048 -> embedding_size)
        self.net.fc = nn.Linear(2048, embedding_size)

    def forward(self, x):
        return self.net(x)