import torch
from torch.utils.checkpoint import checkpoint
import RMSnorm as rm
import multihead_self_attention as MHSA
from positionwise_feedforward import SwiGLU



class Sublayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, weights=None, rope=None, device=None, dtype=None, num_layers=1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype

        if weights is not None:
            Wq, Wk, Wv, Wo, Wg1, W1, W2, W3, Wg2 = weights.values()
            self.rms_norm1 = rm.RMSnorm(d_model, gi=Wg1, eps=1e-5, device=self.device, dtype=torch.float32)
            self.rms_norm2 = rm.RMSnorm(d_model, gi=Wg2, eps=1e-5, device=self.device, dtype=torch.float32)
            self.attn = MHSA.MultiHeadSelfAttention(
                d_model,
                num_heads,
                self.max_seq_len,
                Wq, Wk, Wv, Wo, self.theta,
                mask=None, rope=rope, device=self.device, dtype=self.dtype
            )
            self.FFN = SwiGLU(d_model, w1=W1, w2=W2, w3=W3, dff=d_ff, device=self.device, dtype=self.dtype)
        else:
            self.rms_norm1 = rm.RMSnorm(d_model, eps=1e-5, device=self.device, dtype=torch.float32)
            self.rms_norm2 = rm.RMSnorm(d_model, eps=1e-5, device=self.device, dtype=torch.float32)
            # Q, K, V: std = 1/sqrt(d_in) so a unit-variance input stays unit variance.
            # Wo writes the residual, so std = 1/sqrt(d_model * 2L) and each block adds ~1/(2L).
            proj_std = d_model ** -0.5
            residual_std = (d_model * 2 * num_layers) ** -0.5
            Wq = torch.empty(d_model, d_model, dtype=self.dtype)
            Wk = torch.empty(d_model, d_model, dtype=self.dtype)
            Wv = torch.empty(d_model, d_model, dtype=self.dtype)
            Wo = torch.empty(d_model, d_model, dtype=self.dtype)
            torch.nn.init.trunc_normal_(Wq, mean=0.0, std=proj_std, a=-3.0 * proj_std, b=3.0 * proj_std)
            torch.nn.init.trunc_normal_(Wk, mean=0.0, std=proj_std, a=-3.0 * proj_std, b=3.0 * proj_std)
            torch.nn.init.trunc_normal_(Wv, mean=0.0, std=proj_std, a=-3.0 * proj_std, b=3.0 * proj_std)
            torch.nn.init.trunc_normal_(Wo, mean=0.0, std=residual_std, a=-3.0 * residual_std, b=3.0 * residual_std)
            Wq = Wq.to(self.device)
            Wk = Wk.to(self.device)
            Wv = Wv.to(self.device)
            Wo = Wo.to(self.device)
            self.attn = MHSA.MultiHeadSelfAttention(
                d_model,
                num_heads,
                self.max_seq_len,
                Wq, Wk, Wv, Wo, self.theta,
                mask=None, rope=rope, device=self.device, dtype=self.dtype
            )
            self.FFN = SwiGLU(d_model, dff=d_ff, device=self.device, dtype=self.dtype, num_layers=num_layers)

    def _pre_attn(self, x):
        return self.rms_norm1(x)

    def _ffn_block(self, res1):
        return self.FFN(self.rms_norm2(res1))

    def forward(self, x):
        # Recompute norms and SwiGLU. Attention, called below, keeps its S^2 map.
        if torch.is_grad_enabled() and x.requires_grad:
            norm_x1 = checkpoint(self._pre_attn, x, use_reentrant=False)
        else:
            norm_x1 = self._pre_attn(x)
        out_1 = self.attn(norm_x1)
        res1 = x + out_1
        if torch.is_grad_enabled() and res1.requires_grad:
            ffn_out = checkpoint(self._ffn_block, res1, use_reentrant=False)
        else:
            ffn_out = self._ffn_block(res1)
        return res1 + ffn_out
