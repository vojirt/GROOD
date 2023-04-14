import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(object):
    def __init__(self, cfg, **kwargs):
        self.nll_loss = nn.NLLLoss(reduction="mean")

    def __call__(self, res, target):
        log_softmax = F.log_softmax(res.logits, dim=-1)
        ce_loss = self.nll_loss(log_softmax, target.long())

        return ce_loss


class BCELoss(object):
    def __init__(self, cfg, **kwargs):
        self.bceloss = nn.BCELoss(reduction="mean")

    def __call__(self, res, target):
        bceloss = self.bceloss(res.decoded, target)

        return bceloss
