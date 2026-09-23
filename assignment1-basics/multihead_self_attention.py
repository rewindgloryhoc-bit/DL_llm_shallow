import torch.nn as nn
import torch
from RotaryPositionalEmbedding import RotaryPositionalEmbedding as RoPE
import scaled_dot_product_attention as sca
import einops


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, max_seq, Wq, Wk, Wv, Wo, rope_theta, mask=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.Wq = nn.Parameter(Wq)
        self.Wk = nn.Parameter(Wk)
        self.Wv = nn.Parameter(Wv)
        self.Wo = nn.Parameter(Wo)
        self.device = device
        self.dtype = dtype
        self.theta = rope_theta
        self.max_seq = max_seq
        self.d_heads = self.d_model // self.num_heads
        self.RoPE = RoPE(self.theta, self.d_heads, self.max_seq, device=device)
        self.mask = mask

    #this function does not actually calculate attention score
    def forward(self, x, token_positions=None):
        if self.mask is None:
            mask_device = self.device if self.device is not None else x.device
            causal_mask = torch.tril(torch.ones(self.max_seq, self.max_seq, dtype=torch.bool, device=mask_device))
            self.mask = causal_mask
        seq_len = x.size(-2)
        mask = self.mask[:seq_len, :seq_len]
        q = x @ self.Wq.T
        k = x @ self.Wk.T
        v = x @ self.Wv.T
        query = torch.unflatten(q, dim=-1, sizes=[-1, self.d_heads]).transpose(-3, -2)
        #[... heads, seq, d_heads]
        key = torch.unflatten(k, dim=-1, sizes=[-1, self.d_heads]).transpose(-3, -2)
        value = torch.unflatten(v, dim=-1, sizes=[-1, self.d_heads]).transpose(-3, -2)

        if token_positions is None:
            tpq = torch.arange(query.size(-2), device=x.device)
            tpk = torch.arange(key.size(-2), device=x.device)
            qi = self.RoPE(query, tpq)
            ki = self.RoPE(key, tpk)

        else:
            qi = self.RoPE(query, token_positions)
            ki = self.RoPE(key, token_positions)

        attention = sca.scaled_attention(qi, ki, value, mask=mask)
        flat_att = einops.rearrange(attention, "... head seq hd -> ... seq (head hd)")
        projection = flat_att @ self.Wo.T
        return projection

