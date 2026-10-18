import torch
import torch.nn as nn


def rmsnorm_torch(x, gi, eps):
    x_fp32 = x.float()
    rms_scale = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True).float() + eps)
    return x * rms_scale.to(dtype=x.dtype) * gi.to(dtype=x.dtype)


class RMSnorm(torch.nn.Module):
    def __init__(self, d_model, gi=None, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        # Gain stays fp32 even when the rest of the model is bf16.
        self.dtype = torch.float32
        if gi is None:
            self.gi = nn.Parameter(torch.ones(d_model, device=self.device, dtype=torch.float32))
        else:
            self.gi = nn.Parameter(gi.detach().to(device=self.device, dtype=torch.float32))

    
    def forward(self, x: torch.Tensor): #[seq, d_model])
        return rmsnorm_torch(x, self.gi, self.eps)

