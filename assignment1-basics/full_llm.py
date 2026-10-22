from RMSnorm import RMSnorm
import torch
from sublayer import Sublayer
from embedding import Embedding
from RotaryPositionalEmbedding import RotaryPositionalEmbedding
from mixture_RoPE import USE_MIXTURE_ROPE
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

        # Shared RoPE cache: half-head for mixture, full head for standard RoPE.
        rope_dim = self.d_head // 2 if USE_MIXTURE_ROPE else self.d_head
        self.rope = RotaryPositionalEmbedding(self.theta, rope_dim, self.max_seq_len, device=self.device)

        self.layers = torch.nn.ModuleList([
            Sublayer(
                self.d_model,
                self.num_heads,
                self.d_ff,
                self.max_seq_len,
                self.theta,
                weights,
                rope=self.rope,
                device=self.device,
                dtype=self.dtype,
                num_layers=num_layers,
            )
            for _ in range(num_layers)
        ])

        # Final RMSNorm gain stays fp32 regardless of model dtype.
        self.Wrmf = RMSnorm(self.d_model, gi=None, eps=1e-5, device=self.device, dtype=torch.float32)
        # Zero head: logits start at 0, so the first loss is ln(vocab_size).
        Wf = torch.zeros(vocab_size, self.d_model, dtype=self.dtype, device=self.device)
        self.Wlm = torch.nn.Parameter(Wf, requires_grad=True)


    #note x here refers to encoded prompt.
    # Attention maps stay saved. Norms, QKV projections, and SwiGLU are recomputed.
    def forward(self, x, targets=None, return_logits=True, grad_accum_steps=1, mfu=None):
        self.mfu = mfu
        x = self.embedding(x)
        for sublayer in self.layers:
            x = sublayer(x)
        final_x = self.Wrmf(x)
        
        if targets is not None:
            # We compute loss here to allow chunking and saving memory
            from cross_entropy import cross_entropy
            # If not chunking, we could just do it normally, but we implement chunking over batch dimension.
            B, S, d_model = final_x.shape
            chunk_size = 32 # number of batch rows per chunk (256/32 = 8 chunks). Adjust as needed to balance overhead/memory.
            
            total_loss = torch.zeros((), device=final_x.device, dtype=final_x.dtype)
            
            # we need to be careful with backwards. if we just backward the chunk loss, it'll walk the whole graph
            # every chunk. so we detach final_x, do the backward on final_x chunks, then do one big backward
            # on the original final_x using the accumulated gradients.
            if final_x.requires_grad:
                final_x_detached = final_x.detach().requires_grad_(True)
            else:
                final_x_detached = final_x
            
            for i in range(0, B, chunk_size):
                x_chunk = final_x_detached[i:i+chunk_size]
                target_chunk = targets[i:i+chunk_size]
                
                # Compute logits for just this chunk
                logits_chunk = x_chunk @ self.Wlm.T
                
                # Compute loss for this chunk. cross_entropy handles the internal operations.
                loss_chunk = cross_entropy(logits_chunk, target_chunk)
                # scale chunk loss so the sum equals the mean over the whole batch
                loss_chunk = loss_chunk * (x_chunk.shape[0] / B)
                total_loss += loss_chunk.detach()
                
                if final_x.requires_grad:
                    # Apply grad accum step scaling right here to the chunk loss
                    scaled_loss_chunk = loss_chunk / grad_accum_steps
                    scaled_loss_chunk.backward()
            
            if final_x.requires_grad:
                if self.mfu is not None:
                    # Signal that forward pass is done (transformer + chunked final logit/loss)
                    self.mfu.record_end_fwd()
                    
                final_x.backward(final_x_detached.grad)
                
                if self.mfu is not None:
                    # Signal that backward pass is done
                    self.mfu.record_end_bwd()
                    
            total_loss.requires_grad_(True)
            if return_logits:
                return total_loss, None
            return total_loss

        logits = final_x @ self.Wlm.T
        return logits
