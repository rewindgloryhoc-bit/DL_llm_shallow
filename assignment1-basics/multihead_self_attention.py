import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
from RotaryPositionalEmbedding import RotaryPositionalEmbedding as RoPE
import mixture_RoPE as mr
import scaled_dot_product_attention as sdpa


''' 
note that this clsas takes in custom rope class, but only to be used 
as cos and sin matrices which serves mixture RoPE.
'''
class MultiHeadSelfAttention(torch.nn.Module):
    #(q, k, v, mask) -> attention output; subclasses may swap in another implementation
    attention_fn = staticmethod(sdpa.scaled_attention)

    def __init__(self, d_model, num_heads, max_seq, Wq, Wk, Wv, Wo, rope_theta, mask=None, rope=None, device=None, dtype=None):
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
        # Mixture rotates half-head; standard RoPE rotates full head dim.
        if rope is None:
            rope_dim = self.d_heads // 2 if mr.USE_MIXTURE_ROPE else self.d_heads
            rope = RoPE(self.theta, rope_dim, self.max_seq, device=device)
        self.RoPE = rope
        self.mask = mask

    #this function does not actually calculate attention score
    def forward(self, x, token_positions=None):
        if self.mask is None:
            mask_device = self.device if self.device is not None else x.device
            causal_mask = torch.tril(torch.ones(self.max_seq, self.max_seq, dtype=torch.bool, device=mask_device))
            self.mask = causal_mask
        seq_len = x.size(-2)
        mask = self.mask[:seq_len, :seq_len]
        query, key, value = self._project_qkv(x)

        if token_positions is None:
            token_positions = torch.arange(query.size(-2), device=x.device)

        # Attention map stays saved. Projections are recomputed on backward.
        rope_fn = mr.mixture_RoPE if mr.USE_MIXTURE_ROPE else mr.standard_RoPE
        attention = rope_fn(query, key, value, self.RoPE, token_positions, mask, self.attention_fn)
        return self._out_proj(attention)

    def _project_qkv(self, x):
        def project(x):
            q = x @ self.Wq.T
            k = x @ self.Wk.T
            v = x @ self.Wv.T
            query = torch.unflatten(q, dim=-1, sizes=[-1, self.d_heads]).transpose(-3, -2)
            key = torch.unflatten(k, dim=-1, sizes=[-1, self.d_heads]).transpose(-3, -2)
            value = torch.unflatten(v, dim=-1, sizes=[-1, self.d_heads]).transpose(-3, -2)
            return query, key, value

        if torch.is_grad_enabled() and x.requires_grad:
            return checkpoint(project, x, use_reentrant=False)
        return project(x)

    def _out_proj(self, attention):
        def project(attention):
            return attention @ self.Wo.T

        if torch.is_grad_enabled() and attention.requires_grad:
            return checkpoint(project, attention, use_reentrant=False)
        return project(attention)

