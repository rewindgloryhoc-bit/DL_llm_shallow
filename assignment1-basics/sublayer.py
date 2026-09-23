import torch
import RMSnorm as rm
import multihead_self_attention as MHSA
from positionwise_feedforward import SwiGLU



class Sublayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, weights=None, device=None, dtype=None):
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
            self.rms_norm1 = rm.RMSnorm(d_model, gi=Wg1, eps=1e-5, device=self.device, dtype=self.dtype)
            self.rms_norm2 = rm.RMSnorm(d_model, gi=Wg2, eps=1e-5, device=self.device, dtype=self.dtype)
            self.attn = MHSA.MultiHeadSelfAttention(
                d_model,
                num_heads,
                self.max_seq_len,
                Wq, Wk, Wv, Wo, self.theta,
                mask=None, device=self.device, dtype=self.dtype
            )
            self.FFN = SwiGLU(d_model, w1=W1, w2=W2, w3=W3, dff=d_ff, device=self.device, dtype=self.dtype)
        else:
            self.rms_norm1 = rm.RMSnorm(d_model, eps=1e-5, device=self.device, dtype=self.dtype)
            self.rms_norm2 = rm.RMSnorm(d_model, eps=1e-5, device=self.device, dtype=self.dtype)
            Wq = torch.empty(d_model, d_model, device=self.device, dtype=self.dtype)
            Wk = torch.empty(d_model, d_model, device=self.device, dtype=self.dtype)
            Wv = torch.empty(d_model, d_model, device=self.device, dtype=self.dtype)
            Wo = torch.empty(d_model, d_model, device=self.device, dtype=self.dtype)
            torch.nn.init.trunc_normal_(Wq)
            torch.nn.init.trunc_normal_(Wk)
            torch.nn.init.trunc_normal_(Wv)
            torch.nn.init.trunc_normal_(Wo)
            self.attn = MHSA.MultiHeadSelfAttention(
                d_model,
                num_heads,
                self.max_seq_len,
                Wq, Wk, Wv, Wo, self.theta,
                mask=None, device=self.device, dtype=self.dtype
            )
            self.FFN = SwiGLU(d_model, dff=d_ff, device=self.device, dtype=self.dtype)

    def forward(self, x):
        norm_x1 = self.rms_norm1(x)
        out_1 = self.attn(norm_x1)
        res1 = x + out_1
        norm_x2 = self.rms_norm2(res1)
        x_out = res1 + self.FFN(norm_x2)
        return x_out
