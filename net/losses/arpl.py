import torch.nn.functional as F

class ARPL(object):
    """
    Implementation of the ARPL loss from Adversarial Reciprocal Points Learning for Open Set Recognition (TPAMI'21) 
    adopted to our codebase, i.e.
        - plane_normals serves as a Reciprocal Points
        - res.logits = directional error (projection of emb to dir of plane_normals ~ reciprocal points)
    """
    def __init__(self, cfg, **kwargs):
        self.l = 0.1

    def __call__(self, res, target):
        de = (res.emb[:, None, :] - res.plane_normals[None, ...]).pow(2).mean(-1)
        ce_loss = F.cross_entropy(de - res.logits, target)

        de_b = (res.emb - res.plane_normals[target, ...]).pow(2).mean(-1)

        return ce_loss + self.l * F.relu(de_b - res.R + 1).mean() 

