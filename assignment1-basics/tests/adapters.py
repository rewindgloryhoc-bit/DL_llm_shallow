from __future__ import annotations
from tokenizer_parallel import tokenization
import tokenizer_encode
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
import linear_module
import embedding
from RMSnorm import RMSnorm
from positionwise_feedforward import SwiGLU
from RotaryPositionalEmbedding import RotaryPositionalEmbedding as RoPE
from softmax import softmax
import scaled_dot_product_attention as sc
from multihead_self_attention import MultiHeadSelfAttention
import einops
from full_llm import Sublayer
from cross_entropy import cross_entropy
from Adam_W import Adam_W
from cos_annealing import cos_annealing
from gradient_clipping import gradient_clipping
import numpy as np
from check_point import save_checkpoint, load_checkpoint




torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    weight_tensor = linear_module.LinearModule(d_in, d_out, device=weights.device, dtype=weights.dtype)
    weight = {"weight": weights}
    weight_tensor.load_state_dict(weight)
    D = weight_tensor.forward(in_features)
    return D



def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    embedding_tensor = embedding.Embedding(vocab_size, d_model, device=weights.device, dtype=weights.dtype)
    d_embed = {"embedding": weights}
    embedding_tensor.load_state_dict(d_embed)
    vectors = embedding_tensor.forward(token_ids)
    return vectors



def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    swiglu = SwiGLU(d_model,
                    w1=w1_weight,
                    w2=w2_weight,
                    w3=w3_weight,
                    dff=d_ff, device=in_features.device, dtype=in_features.dtype)
    return swiglu(in_features)



def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    return sc.scaled_attention(Q, K, V, mask=mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    q, k, v, m = MHSA.MHA(
        d_model, num_heads, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features
    )
    attention = sc.scaled_attention(q, k, v, m)
    attention_flattened = einops.rearrange(
        attention, "... num_head seq d_head -> ... seq (num_head d_head)"
    )
    return attention_flattened @ o_proj_weight.T



def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    #respect with size [... heads, seq, d_heads]

    attention = MultiHeadSelfAttention(d_model,
                                       num_heads,
                                       max_seq_len,
                                       q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
                                       theta, mask=None, device=in_features.device, dtype=in_features.dtype)
    return attention(in_features)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:

    rotation = RoPE(theta, d_k, max_seq_len, device=in_query_or_key.device)
    return rotation.forward(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    assigned_weights = {
        "Wq" : weights["attn.q_proj.weight"],
        "Wk" : weights["attn.k_proj.weight"],
        "Wv" : weights["attn.v_proj.weight"],
        "Wo" : weights["attn.output_proj.weight"],
        "Wg1" : weights["ln1.weight"],
        "W1" : weights["ffn.w1.weight"],
        "W2" : weights["ffn.w2.weight"],
        "W3" : weights["ffn.w3.weight"],
        "Wg2" : weights["ln2.weight"],
    }
    tranformer_block = Sublayer(
        d_model,
        num_heads, d_ff, max_seq_len, theta, assigned_weights,
        device=in_features.device, dtype=in_features.dtype
    )
    return tranformer_block(in_features)



def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """

    W_e = weights["token_embeddings.weight"]
    x = W_e[in_indices]

    for layers in range(0, num_layers):
        assigned_weights = {
            "Wq" : weights[f"layers.{layers}.attn.q_proj.weight"],
            "Wk" : weights[f"layers.{layers}.attn.k_proj.weight"],
            "Wv" : weights[f"layers.{layers}.attn.v_proj.weight"],
            "Wo" : weights[f"layers.{layers}.attn.output_proj.weight"],
            "Wg1" : weights[f"layers.{layers}.ln1.weight"],
            "W1" : weights[f"layers.{layers}.ffn.w1.weight"],
            "W2" : weights[f"layers.{layers}.ffn.w2.weight"],
            "W3" : weights[f"layers.{layers}.ffn.w3.weight"],
            "Wg2" : weights[f"layers.{layers}.ln2.weight"],
        }
        transformer_layer = Sublayer(
        d_model,
        num_heads, d_ff, context_length, rope_theta, assigned_weights,
        device=x.device, dtype=x.dtype
    )
        x = transformer_layer(x)


    #layers succeed
    Wf = weights["ln_final.weight"]
    Wg3 = weights["lm_head.weight"]
    final_norm = RMSnorm(d_model, gi=Wf, eps=1e-5, device=x.device, dtype=x.dtype)
    x_normf = final_norm(x)
    logits = x_normf @ Wg3.T
    return logits



def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    normalizer = RMSnorm.RMSnorm(d_model, gi=weights, eps=1e-5, device=in_features.device, dtype=in_features.dtype)
    return normalizer.forward(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_idx = len(dataset) - context_length - 1
    start_indices = np.random.randint(0, max_idx + 1, size=batch_size)
    raw_x = [dataset[k: k + context_length] for k in start_indices]
    raw_y = [dataset[k + 1: k + 1 + context_length] for k in start_indices]
    x = torch.as_tensor(np.array(raw_x), device=device)
    y = torch.as_tensor(np.array(raw_y), device=device)
    return x, y



def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    gradient_clipping(parameters, max_l2_norm)



def get_adamw_cls() -> Any:
    return Adam_W

def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    return cos_annealing(it, min_learning_rate, max_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
   save_checkpoint(model, optimizer, iteration, out)

def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:

    return load_checkpoint(src, model, optimizer)



def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    try:
        token_encoder = tokenizer_encode.Tokenizer(vocab, merges, special_tokens)
        return token_encoder
    except:
        raise NotImplementedError


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    try:
        vocab, merges = tokenization(vocab_size, input_path, special_tokens)
        return vocab, merges
    except:
        raise NotImplementedError
