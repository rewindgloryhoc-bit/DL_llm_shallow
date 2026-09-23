import torch


def _unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _normalize_state_dict(state_dict):
    """Strip torch.compile's `_orig_mod.` prefix if present."""
    prefix = "_orig_mod."
    if state_dict and all(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def save_checkpoint(model, optimizer, iteration, out):
    # Always persist the underlying module weights (no compile wrapper prefix).
    checkpoint = {
        "parameters": _unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    state = _normalize_state_dict(checkpoint["parameters"])
    _unwrap_model(model).load_state_dict(state)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
