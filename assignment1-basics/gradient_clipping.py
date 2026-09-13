import torch
import math

def gradient_clipping(params, M, eps=1e-6):
    with torch.no_grad():
        global_norm = 0
        for p in params:
            if p.grad is not None:
                global_norm += torch.pow(p.grad, 2).sum().item()
        global_norm = math.sqrt(global_norm)

        if global_norm > M:
            scaling = M / (global_norm + eps)
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(scaling)