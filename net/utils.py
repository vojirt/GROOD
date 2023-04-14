import torch
import torch.nn as nn

class LinearFixedNormals(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(LinearFixedNormals, self).__init__()

        # [C, E]
        # random initialization like in ARPL code 
        generated_planes = 0.1 * torch.randn(cfg.MODEL.NUM_CLASSES, cfg.MODEL.EMB_SIZE)
        self.plane_normals = nn.Parameter(generated_planes, requires_grad=True)

        # [1, C, 1]
        self.s = nn.Parameter(torch.ones((1, cfg.MODEL.NUM_CLASSES, 1), dtype=torch.float), requires_grad=False)

        # [1, C]
        self.bias = nn.Parameter(torch.zeros((1, cfg.MODEL.NUM_CLASSES), dtype=torch.float), requires_grad=False)

    def get_plane_normals(self):
        return self.s[0, ...] * self.plane_normals

    def forward(self, x):
        # [B, C] = [1, C, E] x [B, E, 1]
        # NOTE: same as nn.Linear with weight = s*plane_normals, bias = bias
        return torch.matmul(self.s * self.plane_normals[None, ...], x[..., None])[:, :, 0] + self.bias
