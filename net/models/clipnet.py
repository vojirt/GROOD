import torch
import torch.nn as nn
import clip
from types import SimpleNamespace

class CLIPNet(nn.Module):
    """
    A wrapper around a pre-trained CLIP network (https://github.com/openai/CLIP)
    """
    def __init__(self, cfg):
        super(CLIPNet, self).__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using CLIP variant {cfg.MODEL.CLIP_VARIANT}")
        self.model, self.preprocess = clip.load(cfg.MODEL.CLIP_VARIANT, device=device)

    def forward(self, x):
        with torch.no_grad():
            emb = self.model.encode_image(x)
            emb = emb.float()

        return SimpleNamespace(logits = None, emb = emb)
