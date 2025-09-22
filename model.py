import torch
import torch.nn as nn
import torchvision.models as models

class Disease_Classifier(nn.Module):
    def __init__(self, label, freeze_backbone=True):
        super().__init__()
        # Load pretrained ResNet18
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Optionally freeze everything
        if freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Replace classification head
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, label)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)
