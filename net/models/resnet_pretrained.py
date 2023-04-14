import torch
import torch.nn as nn
import torchvision.models as models

from types import SimpleNamespace


class Resnet101_ImageNet(nn.Module):
    """
    A wrapper around a pre-trained ResNet101 network.
    """
    def __init__(self, cfg):
        super(Resnet101_ImageNet, self).__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # resnet = torch.hub.load('pytorch/vision', 'resnet101', pretrained=True).to(device)
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.features = nn.Sequential(*modules)
        self.fc = resnet.fc
        for p in self.features.parameters():
            p.requires_grad = False
        for p in self.fc.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            emb = self.features(x).squeeze()
            logits = self.fc(emb)
            emb = emb.float()

        extras = {}

        return SimpleNamespace(logits = logits, emb = emb, **extras)

class Resnet50_ImageNet(nn.Module):
    """
    A wrapper around a pre-trained ResNet101 network.
    """
    def __init__(self, cfg):
        super(Resnet50_ImageNet, self).__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # resnet = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True).to(device)
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.features = nn.Sequential(*modules)
        self.fc = resnet.fc
        for p in self.features.parameters():
            p.requires_grad = False
        for p in self.fc.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            emb = self.features(x).squeeze()
            logits = self.fc(emb)
            emb = emb.float()

        extras = {}

        return SimpleNamespace(logits = logits, emb = emb, **extras)

