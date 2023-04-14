import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

from types import SimpleNamespace

class VITB16_ImageNet(nn.Module):
    """
    A wrapper around a pre-trained vit-b16 network.
    """
    def __init__(self, cfg):
        super(VITB16_ImageNet, self).__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # vit = torch.hub.load('pytorch/vision', 'vit_b_16', pretrained=True).to(device)
        vit = models.vit_b_16(pretrained=True)
        # get node name from print(torchvision.models.feature_extraction.get_graph_node_names(vit))
        self.features = create_feature_extractor(vit, return_nodes=['getitem_5', 'heads.head'])
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

    def forward(self, x):
        with torch.no_grad():
            out_dict = self.features(x)
            emb = out_dict['getitem_5']
            logits = out_dict['heads.head']

        extras = {}

        return SimpleNamespace(logits = logits, emb = emb, **extras)


class VITL16_ImageNet(nn.Module):
    """
    A wrapper around a pre-trained vit-b16 network.
    """
    def __init__(self, cfg):
        super(VITL16_ImageNet, self).__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # vit = torch.hub.load('pytorch/vision', 'vit_b_16', pretrained=True).to(device)
        vit = models.vit_l_16(pretrained=True)
        # get node name from print(torchvision.models.feature_extraction.get_graph_node_names(vit))
        self.features = create_feature_extractor(vit, return_nodes=['getitem_5', 'heads.head'])
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()

    def forward(self, x):
        with torch.no_grad():
            out_dict = self.features(x)
            emb = out_dict['getitem_5']
            logits = out_dict['heads.head']

        extras = {}

        return SimpleNamespace(logits = logits, emb = emb, **extras)

