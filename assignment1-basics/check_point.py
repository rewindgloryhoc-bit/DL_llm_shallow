import torch


def _unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _normalize_state_dict(state_dict):
    """Strip torch.compile's `_orig_mod.` prefix if present."""
    prefix = "_orig_mod."
    if state_dict and all(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _as_optimizer_list(optimizer):
    if isinstance(optimizer, (list, tuple)):
        return list(optimizer)
    return [optimizer]


def save_checkpoint(model, optimizer, iteration, out):
    # Always persist the underlying module weights (no compile wrapper prefix).
    # `optimizer` may be a single optimizer or a list/tuple (Muon + AdamW).
    opts = _as_optimizer_list(optimizer)
    checkpoint = {
        "parameters": _unwrap_model(model).state_dict(),
        "optimizers": [opt.state_dict() for opt in opts],
        # keep legacy single-optimizer key for older loaders
        "optimizer": opts[0].state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    # Remap tensors onto the model's device (e.g. MPS checkpoint -> CUDA).
    map_device = next(model.parameters()).device
    checkpoint = torch.load(src, map_location=map_device, weights_only=False)
    state = _normalize_state_dict(checkpoint["parameters"])
    _unwrap_model(model).load_state_dict(state)

    opts = _as_optimizer_list(optimizer)
    if "optimizers" in checkpoint:
        saved = checkpoint["optimizers"]
        if len(saved) != len(opts):
            raise ValueError(
                f"checkpoint has {len(saved)} optimizers, but loader got {len(opts)}"
            )
        for opt, opt_state in zip(opts, saved):
            opt.load_state_dict(opt_state)
    else:
        # Legacy single-optimizer checkpoints
        opts[0].load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]
