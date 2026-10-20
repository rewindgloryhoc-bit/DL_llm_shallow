import torch
from torch.utils.checkpoint import checkpoint
import RMSnorm as rm
import multihead_self_attention as MHSA
from positionwise_feedforward import SwiGLU



class Sublayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, weights=None, rope=None, device=None, dtype=None):
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
            #do the math of truncation on cpu directly instead of using torch.empty on gpu.
            Wq = torch.empty(d_model, d_model, dtype=self.dtype)
            Wk = torch.empty(d_model, d_model, dtype=self.dtype)
            Wv = torch.empty(d_model, d_model, dtype=self.dtype)
            Wo = torch.empty(d_model, d_model, dtype=self.dtype)
            torch.nn.init.trunc_normal_(Wq)
            torch.nn.init.trunc_normal_(Wk)
            torch.nn.init.trunc_normal_(Wv)
            torch.nn.init.trunc_normal_(Wo)
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
            self.FFN = SwiGLU(d_model, dff=d_ff, device=self.device, dtype=self.dtype)

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
