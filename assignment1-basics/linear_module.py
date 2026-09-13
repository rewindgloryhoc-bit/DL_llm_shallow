import torch

class LinearModule(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        W = torch.empty((out_features, in_features), device=device, dtype=dtype)
        W_tensor = torch.nn.init.trunintersc_normal_(W)
        self.weight = torch.nn.Parameter(W_tensor)

    #use transpose
    def forward(self, x:torch.Tensor):
        return x @ self.weight.T




