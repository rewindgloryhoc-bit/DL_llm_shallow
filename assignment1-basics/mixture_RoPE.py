import torch
import scaled_dot_product_attention as sdpa
import einops


def mixture_RoPE(q, k, v, rope, token_positions, mask, attention_fn=sdpa.scaled_attention):
    """Mixture attention over head dim halves.

    q, k, v: [..., heads, seq, d_head]
    - First half of Q/K: apply RoPE, then scaled attention with V
    - Second half of Q/K: shift K by one along seq, then scaled attention with V
    - Return sum of the two attention outputs, heads flattened
    """
    # Contiguous half-split along d_head (not interleaved pair unflatten).
    d_half = q.size(-1) // 2
    q_rope, q_shift = q[..., :d_half], q[..., d_half:]
    k_rope, k_raw = k[..., :d_half], k[..., d_half:]

    q_rope = rope(q_rope, token_positions)
    k_rope = rope(k_rope, token_positions)

    # Key at position t uses features from t-1; position 0 has no predecessor.
    k_shift = torch.roll(k_raw, shifts=1, dims=-2)
    k_shift[..., 0, :] = 0.0

    rope_attn = attention_fn(q_rope, k_rope, v, mask)
    shift_attn = attention_fn(q_shift, k_shift, v, mask)
    attention = rope_attn + shift_attn
    return einops.rearrange(attention, "... head seq hd -> ... seq (head hd)")
