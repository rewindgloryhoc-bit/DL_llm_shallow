from RMSnorm import RMSnorm
import torch
from sublayer import Sublayer
from embedding import Embedding
'''
    out = x + Multi-head_attention(RMSnorm(x)), with RoPE
    x ~ [batch sequence h_d]
'''

class Full_llm(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta, num_layers, vocab_size, weights=None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.d_head = self.d_model // self.num_heads
        self.device = device
        self.dtype = dtype

        self.embedding = Embedding(vocab_size, d_model, device=self.device, dtype=self.dtype)

        self.layers = torch.nn.ModuleList([
            Sublayer(
                self.d_model,
                self.num_heads,
                self.d_ff,
                self.max_seq_len,
                self.theta,
                weights,
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(num_layers)
        ])

        self.Wrmf = RMSnorm(self.d_model, gi=None, eps=1e-5, device=self.device, dtype=self.dtype)
        # Shape matches adapters' lm_head.weight: (vocab_size, d_model)
        Wf = torch.empty(vocab_size, self.d_model, dtype=self.dtype, device=self.device)
        Wf = torch.nn.init.trunc_normal_(Wf)
        self.Wlm = torch.nn.Parameter(Wf, requires_grad=True)


    #note x here refers to encoded prompt.
    def forward(self, x):
        x = self.embedding(x)
        for sublayer in self.layers:
            x = sublayer(x)
        final_x = self.Wrmf(x)
        logits = final_x @ self.Wlm.T
        return logits
