import torch.nn as nn
from torchvision import models

class ShoeClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ShoeClassifier, self).__init__()g
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x):
        return self.resnet(x)
    
    def extract_features(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
            x = x.view(x.size(0), -1)  # Flatten
        return x
