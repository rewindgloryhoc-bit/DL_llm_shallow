import torch


#i should be -1
def softmax(x: torch.Tensor, i):
    c = torch.max(x, dim=i, keepdim=True).values
    x_scaled = x - c
    summation = torch.sum(torch.exp(x_scaled), dim=i, keepdim=True)
    return torch.exp(x_scaled) / summation
