from collections.abc import Callable
from typing import Optional
import math
import torch
from cos_annealing import scheduled_lr


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
        decay_steps=1200,
        schedule_mode="iter",
        Tw_sec=None,
        T_budget=None,
        decay_sec=None,
    ):
        if betas is None:
            betas = (0.9, 0.98)
        if eps is None:
            eps = 1e-8
        if weight_decay is None:
            weight_decay = 0.01

        defaults = {
            "lr": lr_min,
            "lr_min": lr_min,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "lr_max": lr_max,
            "Tw": Tw,
            "Tc": Tc,
            "decay_steps": decay_steps,
            "schedule_mode": schedule_mode,
            "Tw_sec": Tw_sec,
            "T_budget": T_budget,
            "decay_sec": decay_sec,
            "train_start_time": None,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            # Hyperparameters
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # Shared step counter (still used for Adam bias correction).
            t = group.get("t", 0) + 1
            group["t"] = t

            # iter → cos_annealing(t); time → cos_annealing_time(elapsed)
            lr = scheduled_lr(group, t)
            group["lr"] = lr

            bias_n1 = 1.0 - (beta1**t)
            bias_n2 = 1.0 - (beta2**t)

            step_size = lr * (math.sqrt(bias_n2) / bias_n1)
            eps_hat = eps * math.sqrt(bias_n2)

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                
                # Momentum states stay in FP32
                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data, dtype=torch.float32)
                    state["v"] = torch.zeros_like(p.data, dtype=torch.float32)

                m = state["m"]
                v = state["v"]
                grad_fp32 = p.grad.data.float()

                # Momentum updates in FP32 (in-place into optimizer state buffers)
                m.mul_(beta1).add_(grad_fp32, alpha=(1.0 - beta1))
                v.mul_(beta2).add_(grad_fp32 * grad_fp32, alpha=(1.0 - beta2))

                # Compute update unit
                unit = m / (torch.sqrt(v) + eps_hat)

                # Decoupled weight decay + Adam update (in-place on original parameter dtype)
                p.data.mul_(1.0 - lr * weight_decay).add_(unit.to(p.dtype), alpha=-step_size)

        return loss


