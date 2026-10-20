import torch


#i should be -1
def softmax(x: torch.Tensor, i):
    c = torch.max(x, dim=i, keepdim=True).values
    # One buffer: subtract, then exp in place. A second exp() would write the
    # whole map again. div_ on that buffer breaks exp's backward, so divide out of place.
    x_scaled = x - c
    x_scaled.exp_()
    summation = torch.sum(x_scaled, dim=i, keepdim=True)
    return x_scaled / summation
