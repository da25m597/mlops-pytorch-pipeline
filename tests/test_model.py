# Unit tests for src/model.py
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import get_model

def test_get_model_returns_module():
    model = get_model("resnet18", num_classes=10, pretrained=False)
    assert isinstance(model, nn.Module)

def test_get_model_adapts_stem_for_grayscale_input():
    model = get_model("resnet18", num_classes=10, pretrained=False)
    assert isinstance(model.conv1, nn.Conv2d)
    assert model.conv1.in_channels == 1
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, nn.Identity)

def test_get_model_sets_correct_output_classes():
    model = get_model("resnet18", num_classes=10, pretrained=False)
    assert model.fc.out_features == 10

    model_5 = get_model("resnet18", num_classes=5, pretrained=False)
    assert model_5.fc.out_features == 5

def test_get_model_rejects_unsupported_architecture():
    with pytest.raises(ValueError):
        get_model("vgg16", num_classes=10, pretrained=False)

def test_forward_pass_output_shape():
    model = get_model("resnet18", num_classes=10, pretrained=False)
    model.eval()
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 28, 28)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (batch_size, 10)
