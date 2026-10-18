import torch
import math
import torch.nn as nn


def swiglu_torch(x, w1, w2, w3):
    s1 = x @ w1.T
    SiLU = s1 * torch.sigmoid(s1)
    GLU = x @ w3.T
    product = SiLU * GLU
    swiglu = product @ w2.T
    return swiglu


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, w1=None, w2=None, w3=None, dff=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if not dff:
            dff = int(math.ceil(8 / 3 * self.d_model / 64) * 64)
        self.dff = dff
        self.device = device
        self.dtype = dtype
        if w1 is None and w2 is None and w3 is None:
            w1 = torch.empty(self.dff, self.d_model, device=self.device, dtype=self.dtype)
            w2 = torch.empty(self.d_model, self.dff, device=self.device, dtype=self.dtype)
            w3 = torch.empty(self.dff, self.d_model, device=self.device, dtype=self.dtype)
            self.w1 = torch.nn.Parameter(torch.nn.init.trunc_normal_(w1))
            self.w2 = torch.nn.Parameter(torch.nn.init.trunc_normal_(w2))
            self.w3 = torch.nn.Parameter(torch.nn.init.trunc_normal_(w3))
        else:
            self.w1 = nn.Parameter(w1)
            self.w2 = nn.Parameter(w2)
            self.w3 = nn.Parameter(w3)

    def forward(self, x:torch.Tensor):
        return swiglu_torch(x, self.w1, self.w2, self.w3)