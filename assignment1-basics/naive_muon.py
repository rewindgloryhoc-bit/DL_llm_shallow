import torch
import torch.distributed as dist
from cos_annealing import scheduled_lr

#muon optimizer implemented from nanogpt from Github


def zeropower_via_newtonschulz5(G, steps: int):
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    #momentum = beta * momentum + (1 - beta) * grad
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1))**0.5
    return update



class Muon(torch.optim.Optimizer):
    def __init__(self, params, weight_decay=0.01, beta=0.95, lr_min=0.002, lr_max=0.03, Tw=350, Tc=5000, decay_steps=1200,
                 schedule_mode="iter", Tw_sec=None, T_budget=None, decay_sec=None):
        defaults = {
            "lr": lr_min,
            "lr_min": lr_min,
            "lr_max": lr_max,
            "Tw": Tw,
            "Tc": Tc,
            "decay_steps": decay_steps,
            "weight_decay": weight_decay,
            "beta": beta,
            "schedule_mode": schedule_mode,
            "Tw_sec": Tw_sec,
            "T_budget": T_budget,
            "decay_sec": decay_sec,
            "train_start_time": None,
            }

        params = list(params)
        assert len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # NanoGPT Muon shards updates across ranks; fall back to local updates
        # when torch.distributed is not initialized (e.g. dry-run / single GPU).
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        #note rank here refers to specific gpu index.
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            t = group.get("t", 0) + 1
            group["t"] = t
            lr = scheduled_lr(group, t)
            group["lr"] = lr

            if distributed:
                pad = (world_size - len(params) % world_size) % world_size
                #pad the params to the world size
                params_pad = params + [torch.empty_like(params[-1])] * pad
            else:
                params_pad = params

            #This for loop assigns a specific gpu with different weight tensors.
            for base_i in range(0, len(params_pad), world_size):
                local_i = base_i + rank
                if local_i < len(params):
                    p = params[local_i]
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["beta"])
                    p.mul_(1 - lr * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-lr)
                if distributed:
                    dist.all_gather(
                        params_pad[base_i:base_i + world_size],
                        params_pad[base_i + rank],
                    )

        return loss
