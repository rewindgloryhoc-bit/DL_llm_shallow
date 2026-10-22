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
    def __init__(self, d_model, w1=None, w2=None, w3=None, dff=None, device=None, dtype=None, num_layers=1):
        super().__init__()
        self.d_model = d_model
        if not dff:
            dff = int(math.ceil(8 / 3 * self.d_model / 64) * 64)
        self.dff = dff
        self.device = device
        self.dtype = dtype
        if w1 is None and w2 is None and w3 is None:
            # w1 and w3 (the gate) read d_model and do not write the residual.
            # w2 reads d_ff and adds into the residual, so its std includes both fan-in and 2L.
            proj_std = self.d_model ** -0.5
            residual_std = (self.dff * 2 * num_layers) ** -0.5
            w1 = torch.empty(self.dff, self.d_model, dtype=self.dtype)
            w2 = torch.empty(self.d_model, self.dff, dtype=self.dtype)
            w3 = torch.empty(self.dff, self.d_model, dtype=self.dtype)
            w1 = torch.nn.init.trunc_normal_(w1, mean=0.0, std=proj_std, a=-3.0 * proj_std, b=3.0 * proj_std).to(self.device)
            w2 = torch.nn.init.trunc_normal_(w2, mean=0.0, std=residual_std, a=-3.0 * residual_std, b=3.0 * residual_std).to(self.device)
            w3 = torch.nn.init.trunc_normal_(w3, mean=0.0, std=proj_std, a=-3.0 * proj_std, b=3.0 * proj_std).to(self.device)

            self.w1 = torch.nn.Parameter(w1)
            self.w2 = torch.nn.Parameter(w2)
            self.w3 = torch.nn.Parameter(w3)
        else:
            self.w1 = nn.Parameter(w1)
            self.w2 = nn.Parameter(w2)
            self.w3 = nn.Parameter(w3)

    def forward(self, x:torch.Tensor):
        return swiglu_torch(x, self.w1, self.w2, self.w3)