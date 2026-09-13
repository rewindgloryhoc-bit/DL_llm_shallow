import torch
from collections.abc import Callable, Iterable
from typing import Optional
from cos_annealing import cos_annealing


class Adam_W(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-5, betas=None, eps=None, weight_decay=None):
        if betas is None:
            betas = (0.9, 0.999)
        if eps is None:
            eps = 1e-8
        if weight_decay is None:
            weight_decay = 0.01
        if lr is None:
            lr = 1e-3
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0) + 1 # Get iteration number from state, or 0.
                #note some arguments here are subjects to change
                lr = cos_annealing(t, lr, lr_max=3e-4, Tw=200, Tc=9800)
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


