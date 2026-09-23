import torch
from collections.abc import Callable, Iterable
from typing import Optional
from cos_annealing import cos_annealing


class Adam_W(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr_min=1e-4,
        betas=None,
        eps=None,
        weight_decay=None,
        lr_max=1e-3,
        Tw=500,
        Tc=5000,
    ):
        if betas is None:
            betas = (0.9, 0.98)
        if eps is None:
            eps = 1e-8
        if weight_decay is None:
            weight_decay = 0.01

        defaults = {
            "lr": lr_min,  # current scheduled lr (updated each step)
            "lr_min": lr_min,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "lr_max": lr_max,
            "Tw": Tw,
            "Tc": Tc,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr_min = group["lr_min"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            lr_max = group["lr_max"]
            Tw = group["Tw"]
            Tc = group["Tc"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0) + 1 # Get iteration number from state, or 0.
                lr = cos_annealing(t, lr_min, lr_max, Tw, Tc)
                group["lr"] = lr
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                m = state["m"]
                v = state["v"]
                bias_n1 = beta1 ** t
                bias_n2 = beta2 ** t
                grad = p.grad.data
                momentum = ((beta1 * m) + (1 - beta1) * grad)
                momentum_p2 = ((beta2 * v) + (1 - beta2) * grad ** 2)
                unit = (momentum / (1 - bias_n1))  / (torch.sqrt(momentum_p2  / (1 - bias_n2)) + eps)
                p.data -= (lr * weight_decay * p.data) + (lr * unit)
                state["t"] = t
                state["m"] = momentum
                state["v"] = momentum_p2
        return loss


