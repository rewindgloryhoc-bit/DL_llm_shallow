import math
import time


def cos_annealing(t, lr_min, lr_max, Tw, Tc, decay_steps=1200):
    T_decay_start = Tc - decay_steps
    if t < Tw:
        return t / Tw * lr_max
    elif Tw <= t <= T_decay_start:
        #gr = (t - Tw) / (Tc - Tw) #min set to 0 max set to 1
        #return lr_min + (0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * gr)))
        return lr_max
    elif T_decay_start <= t <= Tc:
        gr = (t - T_decay_start) / decay_steps  # Progress from 0.0 to 1.0
        return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * gr))
    else:
        return lr_min


def cos_annealing_time(elapsed, lr_min, lr_max, Tw_sec, T_budget, decay_sec):
    """Legacy pure wall-clock schedule (warmup + decay both in seconds)."""
    T_decay_start = T_budget - decay_sec
    if elapsed < Tw_sec:
        return elapsed / Tw_sec * lr_max
    elif Tw_sec <= elapsed <= T_decay_start:
        return lr_max
    elif T_decay_start <= elapsed <= T_budget:
        gr = (elapsed - T_decay_start) / decay_sec
        return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * gr))
    else:
        return lr_min


def cos_annealing_hybrid(t, elapsed, lr_min, lr_max, Tw, T_budget, decay_sec):
    """Hybrid schedule: fixed-step warmup, wall-clock cosine decay near budget end.
    - Warmup: first Tw optimizer steps (independent of time budget)
    - Decay: last decay_sec of T_budget (typically T_budget / 9)
    - Hold lr_max between warmup end and decay start
    """
    T_decay_start = T_budget - decay_sec
    if t < Tw:
        return (t / Tw) * lr_max
    if elapsed < T_decay_start:
        return lr_max
    if elapsed <= T_budget:
        gr = (elapsed - T_decay_start) / decay_sec
        return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * gr))
    return lr_min


def scheduled_lr(group, t):
    """Pick step-, hybrid-, or legacy time-based schedule from optimizer param group fields."""
    if group.get("schedule_mode") == "time":
        start = group.get("train_start_time")
        elapsed = 0.0 if start is None else time.time() - start
        # Prefer hybrid: step warmup + time decay (Tw_sec unused for scheduling).
        if group.get("Tw") is not None and group.get("decay_sec") is not None:
            return cos_annealing_hybrid(
                t,
                elapsed,
                group["lr_min"],
                group["lr_max"],
                group["Tw"],
                group["T_budget"],
                group["decay_sec"],
            )
        return cos_annealing_time(
            elapsed,
            group["lr_min"],
            group["lr_max"],
            group["Tw_sec"],
            group["T_budget"],
            group["decay_sec"],
        )
    return cos_annealing(
        t,
        group["lr_min"],
        group["lr_max"],
        group["Tw"],
        group["Tc"],
        group["decay_steps"],
    )
