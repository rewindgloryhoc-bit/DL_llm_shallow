import torch
from Adam_W import Adam_W
from naive_muon import Muon

OPTIMIZER_CHOICES = ("muon", "adamw")


def setup_optimizers(
    model,
    lr_min,
    lr_max,
    weight_decay,
    Tw,
    Tc,
    optimizer="muon",
    decay_steps=1200,
    schedule_mode="iter",
    Tw_sec=None,
    T_budget=None,
    decay_sec=None,
):
    """Returns (optimizer_muon, optimizer_adamw).

    optimizer="muon":  2D hidden matrices -> Muon, everything else -> AdamW.
    optimizer="adamw": every parameter -> AdamW; optimizer_muon is None.
    schedule_mode="time": LR from cos_annealing_time (needs Tw_sec, T_budget, decay_sec).
    """
    if optimizer not in OPTIMIZER_CHOICES:
        raise ValueError(f"setup_optimizers: optimizer must be one of {OPTIMIZER_CHOICES}, got {optimizer!r}")

    schedule_kwargs = dict(
        schedule_mode=schedule_mode,
        Tw_sec=Tw_sec,
        T_budget=T_budget,
        decay_sec=decay_sec,
    )

    if optimizer == "adamw":
        optimizer_adamw = Adam_W(
            [p for p in model.parameters() if p.requires_grad],
            lr_min=lr_min,
            lr_max=lr_max,
            weight_decay=weight_decay,
            Tw=Tw,
            Tc=Tc,
            decay_steps=decay_steps,
            **schedule_kwargs,
        )
        return None, optimizer_adamw

    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 1D (e.g. RMSNorm gains), embeddings, and LM head → AdamW
        # remaining 2D matrices (attn / FFN) → Muon
        if (
            param.ndim < 2
            or "embed" in name
            or "lm_head" in name
            or name.endswith("Wlm")
            or ".Wlm" in name
            or name == "Wlm"
        ):
            adamw_params.append(param)
        else:
            muon_params.append(param)

    if not adamw_params:
        raise ValueError("setup_optimizers: no AdamW parameters found")
    if not muon_params:
        raise ValueError("setup_optimizers: no Muon parameters found")

    optimizer_adamw = Adam_W(
        adamw_params,
        lr_min=lr_min,
        lr_max=lr_max,
        weight_decay=weight_decay,
        Tw=Tw,
        Tc=Tc,
        decay_steps=decay_steps,
        **schedule_kwargs,
    )

    optimizer_muon = Muon(
        muon_params,
        decay_steps=decay_steps,
        Tw=Tw,
        Tc=Tc,
        lr_min=lr_min,
        lr_max=lr_max,
        weight_decay=weight_decay,
        **schedule_kwargs,
    )

    return optimizer_muon, optimizer_adamw
