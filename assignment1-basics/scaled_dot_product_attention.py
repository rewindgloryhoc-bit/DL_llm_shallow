import einops
from softmax import softmax
import math


'''
extra attention of final einsum notation, can mess up whole MHA and transformer block 
without throwing runtime error
dangerous einsum diagonal indexing, avoid [... x x] notation which extracts the diagonal
'''

#note mask has dim (seq_len, seq_len)
def scaled_attention(query, keys, values, mask):
    d_k = query.size(-1)
    attention = query @ keys.transpose(-2, -1)
    attention_scaled = attention / math.sqrt(d_k)
    attention_masked = attention_scaled.masked_fill(~mask, float("-inf"))
    # Keep model dtype (bf16 on CUDA) — avoid materializing a full fp32 [B,H,S,S] map.
    normalized_attention = softmax(attention_masked, -1)
    attention_score = einops.einsum(
        normalized_attention, values, "batch ... q k, batch ... k dv -> batch ... q dv"
    ) #previously [batch ... seq seq, batch ... seq d_model -> seq d_model]
    return attention_score
