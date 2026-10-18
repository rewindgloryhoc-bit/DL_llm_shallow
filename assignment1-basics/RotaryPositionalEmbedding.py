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

        #note both rotary matrix has shape with 2D excluding batch dimension
        cos = self.cos_cached[token_positions, :] #becomes [... some_seq, d/2]
        sin = self.sin_cached[token_positions, :]

        # Trig caches are fp32; rotate in fp32 then cast back to x's dtype (e.g. bf16).
        in_dtype = x.dtype
        x1f, x2f = x1.float(), x2.float()
        xr1 = x1f * cos - x2f * sin
        xr2 = x2f * cos + x1f * sin
        x_out = torch.stack([xr1, xr2], dim=-1).flatten(start_dim=-2)
        return x_out.to(in_dtype)

