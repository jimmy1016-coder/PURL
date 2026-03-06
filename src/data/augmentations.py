import torch
import torch.nn as nn


class SpecNormalization(nn.Module):
    def forward(self, spec):
        spec = spec - torch.mean(spec, dim=-2)

        return spec
