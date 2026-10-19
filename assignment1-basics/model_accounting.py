
import torch
from full_llm import Full_llm

# --- mirrored from llm_train.main() / resolve_dataset("opentext") ---
D_MODEL = 1024
NUM_HEADS = 16
NUM_LAYERS = 12
MAX_SEQ = 1024
BATCH_SIZE = 256
GRAD_ACCUM_STEPS = 1
VOCAB_SIZE = 32000  # opentext
THETA = 10000.0
D_FF = int((8 / 3 * D_MODEL) // 128 * 128)  # same as llm_train.d_ff_from_d_model
EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM_STEPS


# small helper function to calculate parameters given current llm architecture
def count_parameters(model):
    total = 0
    embedding = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if name.startswith("embedding."):
            embedding += n
    return total, total - embedding


# Non-attention FLOPs for one training iteration: 6 * N_non_embed * (B * S)
# (2ND forward + 4ND backward).
def flops_non_attention_per_step(model, batch_size, seq_len):
    _, N = count_parameters(model)  # exclude embedding / token tables
    return 6 * N * (batch_size * seq_len)


def flops_attention_per_token(num_layers, seq_len, d_model):
    return 12 * num_layers * seq_len * d_model


def flops_attention(num_layers, seq_len, d_model, batch_size):
    num_tokens = batch_size * seq_len
    return flops_attention_per_token(num_layers, seq_len, d_model) * num_tokens


# Hypothetical length of a training run until we wire real step counts.
HYPOTHETICAL_TRAIN_STEPS = 4000


def flops_total_training(
    model,
    batch_size,
    seq_len,
    num_layers,
    d_model,
    num_steps=HYPOTHETICAL_TRAIN_STEPS,
):
    flops_per_step = flops_non_attention_per_step(model, batch_size, seq_len) + flops_attention(
        num_layers, seq_len, d_model, batch_size
    )
    return flops_per_step * num_steps


# Scaling-law fit for compute-optimal training tokens: N*(C) = A * C^α
SCALING_A = 0.29
SCALING_ALPHA = 0.53
def optimal_training_tokens(compute_flops, A=SCALING_A, alpha=SCALING_ALPHA):
    if compute_flops <= 0:
        raise ValueError(f"compute_flops must be positive, got {compute_flops}")
    return A * (compute_flops ** alpha)


def actual_training_tokens(batch_size, seq_len, num_steps=HYPOTHETICAL_TRAIN_STEPS):
    return batch_size * seq_len * num_steps


def analyze_vram_with_checkpointing(model, batch_size, seq_len, num_layers, d_model, d_ff, num_heads):
    """Calculates max VRAM storage with activation checkpointing recomputation.
    Total VRAM = Model Parameters + Optimizer States + Gradients + Activations + Workspace Overhead
    """
    P_total, _ = count_parameters(model)
    bytes_params = 2 * P_total
    bytes_opt = 8 * P_total
    bytes_grads = 4 * P_total
    
    #Activations (with checkpointing)
    bytes_saved_activations = 2 * num_layers * batch_size * seq_len * d_model
    # Transient_Layer_Bytes: memory footprint inside a single layer's computation 
    # (e.g. Q/K/V tensors, attention out, SwiGLU intermediates in bf16).
    transient_qkv_attn = 4 * (batch_size * seq_len * d_model) * 2  # Q, K, V, and Attn_Out
    attn_matrix_bytes = 2 * batch_size * num_heads * (seq_len ** 2) # QK^T without FlashAttention
    transient_swiglu = 2 * (batch_size * seq_len * d_ff) * 2       # SiLU and GLU up-projections
    bytes_transient = transient_qkv_attn + attn_matrix_bytes + transient_swiglu
    bytes_activations = bytes_saved_activations + bytes_transient
    bytes_overhead = 2 * 1024**3  # Assumes 2 GB overhead
    total_vram = bytes_params + bytes_opt + bytes_grads + bytes_activations + bytes_overhead
    
    return {
        "bytes_params": bytes_params,
        "bytes_opt": bytes_opt,
        "bytes_grads": bytes_grads,
        "bytes_activations": bytes_activations,
        "bytes_overhead": bytes_overhead,
        "total_vram": total_vram,
    }


def build_current_model(device="cpu", dtype=torch.float32):
    """Instantiate Full_llm with the same shape knobs as llm_train.main()."""
    return Full_llm(
        D_MODEL,
        NUM_HEADS,
        D_FF,
        MAX_SEQ,
        THETA,
        NUM_LAYERS,
        VOCAB_SIZE,
        weights=None,
        device=device,
        dtype=dtype,
    ).to(device)


def report_current_setup(num_steps=HYPOTHETICAL_TRAIN_STEPS):
    """Print accounting for the llm_train.main() configuration."""
    model = build_current_model()
    total, non_embed = count_parameters(model)
    flops_non_attn = flops_non_attention_per_step(model, EFFECTIVE_BATCH, MAX_SEQ)
    flops_attn = flops_attention(NUM_LAYERS, MAX_SEQ, D_MODEL, EFFECTIVE_BATCH)
    flops_step = flops_non_attn + flops_attn
    flops_run = flops_total_training(
        model, EFFECTIVE_BATCH, MAX_SEQ, NUM_LAYERS, D_MODEL, num_steps=num_steps
    )
    tokens_actual = actual_training_tokens(EFFECTIVE_BATCH, MAX_SEQ, num_steps=num_steps)
    tokens_opt = optimal_training_tokens(flops_run)
    vram_stats = analyze_vram_with_checkpointing(model, BATCH_SIZE, MAX_SEQ, NUM_LAYERS, D_MODEL, D_FF, NUM_HEADS)

    print("config (from llm_train.main):")
    print(
        f"  d_model={D_MODEL} heads={NUM_HEADS} layers={NUM_LAYERS} d_ff={D_FF}\n"
        f"  vocab={VOCAB_SIZE} seq={MAX_SEQ} batch={BATCH_SIZE} accum={GRAD_ACCUM_STEPS}\n"
        f"  effective_batch={EFFECTIVE_BATCH} steps={num_steps}"
    )
    print(f"total parameters:            {total:,}")
    print(f"parameters excl. embedding:  {non_embed:,}")
    print(f"FLOPs non-attn / step:       {flops_non_attn:.6e}")
    print(f"FLOPs attention / step:      {flops_attn:.6e}")
    print(f"FLOPs / step:                {flops_step:.6e}")
    print(f"FLOPs total ({num_steps} steps): {flops_run:.6e}")
    print(f"actual training tokens:      {tokens_actual:,}")
    print(f"optimal training tokens:     {tokens_opt:.6e}")
    
    print("\n== Peak VRAM Estimation (with Checkpointing) ==")
    print(f"Model Parameters (bf16):     {vram_stats['bytes_params'] / 1024**3:5.2f} GB")
    print(f"Optimizer States (fp32):     {vram_stats['bytes_opt'] / 1024**3:5.2f} GB")
    print(f"Gradients (fp32):            {vram_stats['bytes_grads'] / 1024**3:5.2f} GB")
    print(f"Activations (peak):          {vram_stats['bytes_activations'] / 1024**3:5.2f} GB")
    print(f"Workspace Overhead:          {vram_stats['bytes_overhead'] / 1024**3:5.2f} GB")
    print(f"Total VRAM:                  {vram_stats['total_vram'] / 1024**3:5.2f} GB")
    
    return {
        "total_parameters": total,
        "parameters_excluding_embedding": non_embed,
        "flops_non_attention_per_step": flops_non_attn,
        "flops_attention_per_step": flops_attn,
        "flops_per_step": flops_step,
        "flops_total_training": flops_run,
        "actual_training_tokens": tokens_actual,
        "optimal_training_tokens": tokens_opt,
        "vram_stats": vram_stats,
    }


if __name__ == "__main__":
    report_current_setup()
