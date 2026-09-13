import torch


def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "parameters" : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["parameters"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


