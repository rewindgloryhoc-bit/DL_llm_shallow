import torch
import torch.nn as nn

class RMSnorm(torch.nn.Module):
    def __init__(self, d_model, gi=None, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        if gi is None:
            self.gi = nn.Parameter(torch.ones(d_model, device=self.device, dtype=self.dtype))
        else:
            self.gi = nn.Parameter(gi)

    def forward(self, x: torch.Tensor): #[seq, d_model]
        in_dtype = x.dtype
        x32p = x.to(torch.float32)
        normalizer = torch.rsqrt(torch.mean(x32p ** 2, dim=-1, keepdim=True) + self.eps)
        rmsn = (x32p * normalizer).to(in_dtype)
        normalized = rmsn * self.gi
        return normalized

