import torch
import einops

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        L = torch.arange(self.max_seq_len, device=self.device, dtype=torch.float32)
        k = torch.arange(0, self.d_k, 2, device=self.device, dtype=torch.float32)
        w = 1 / (self.theta ** (k / self.d_k))
        s_theta = torch.outer(L, w)
        self.register_buffer("cos_cached",torch.cos(s_theta), persistent=False)
        self.register_buffer("sin_cached",torch.sin(s_theta), persistent=False)


    def forward(self, x, token_positions):
        #pairs = x.unflatten(dim=-1, sizes=(-1, 2)) #becomes [... seq_len, d/2, 2]
        pairs = einops.rearrange(x, "... seq (dK_half two) -> ... seq dK_half two", two=2)
        x1 = pairs[..., 0] #becomes [... seq_len, d/2]
        x2 = pairs[..., 1]

        cos = self.cos_cached[token_positions, :] #becomes [... some_seq, d]
        sin = self.sin_cached[token_positions, :]
        '''
        if cos.ndim < x.ndim:
            cos = cos.unsqueeze(-2)  # [batch, 1, seq, d_k // 2]
            sin = sin.unsqueeze(-2)
        '''
        xr1 = x1 * cos - x2 * sin
        xr2 = x2 * cos + x1 * sin
        x_out = torch.stack([xr1, xr2], dim=-1).flatten(start_dim=-2)
        return x_out

