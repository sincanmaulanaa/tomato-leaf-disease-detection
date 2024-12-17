# model.py
import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    model = models.alexnet(weights=None)  # Use weights=None instead of pretrained=False
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model