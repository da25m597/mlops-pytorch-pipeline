# Fashion-MNIST Image classifier model

import torch.nn as nn
import torchvision.models as models

def get_model(architecture: str, num_classes: int =10, pretrained: bool=True) -> nn.Module:
    architecture = architecture.lower()

    if architecture != "resnet18":
        raise ValueError(f"Unsupported architecture: {architecture!r} (expected 'resnet18')")

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    # 1 channel stem
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
