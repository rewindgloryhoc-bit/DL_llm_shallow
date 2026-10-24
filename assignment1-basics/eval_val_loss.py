
import gc
import numpy as np
import torch
from check_point import _normalize_state_dict, _unwrap_model
from full_llm import Full_llm
from get_batch import _get_memmap
from llm_train import (
    BASE_DIR,
    checkpoint_path,
    pick_device,
    pick_dtype,
    resolve_dataset,
)
from tokenizer_encode import Encoder


SPECIAL_TOKENS = ["<|endoftext|>"]
_DATASET = resolve_dataset("opentext")
VALID_PATH = BASE_DIR / "owt_valid.txt"
ENCODED_PATH = BASE_DIR / "token_data" / "encoded_owt_valid.bin"
VOCAB_PATH = _DATASET["vocab_path"]
MERGES_PATH = _DATASET["merges_path"]
NUM_HEADS = 16
MAX_SEQ = 512
THETA = 10000.0
BATCH_SIZE = 32
LOG_EVERY = 50


# Load model weights on CPU and drop optimizer state so it is not kept in memory.
def load_weights(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = _normalize_state_dict(checkpoint["parameters"])
    iteration = checkpoint.get("iteration")
    del checkpoint
    gc.collect()
    return state, iteration


# Encode owt_valid.txt to a uint16 memmap, reusing the file if it already exists.
def encode_validation(valid_path, encoded_path, vocab_path, merges_path):
    if not valid_path.is_file():
        raise FileNotFoundError(f"validation text not found: {valid_path}")
    if not vocab_path.is_file() or not merges_path.is_file():
        raise FileNotFoundError(
            f"tokenizer files not found: {vocab_path} and {merges_path}"
        )
    encoder = Encoder.from_files(vocab_path, merges_path, SPECIAL_TOKENS)
    encoder.encode(valid_path, encoded_path)
    if not encoded_path.is_file():
        raise FileNotFoundError(f"encoding did not write {encoded_path}")


# Read vocab size, d_model, d_ff, and layer count from the checkpoint tensors.
def arch_from_state(state):
    try:
        vocab_size, d_model = state["embedding.embedding"].shape
        d_ff = state["layers.0.FFN.w1"].shape[0]
    except KeyError as exc:
        raise KeyError(
            "checkpoint is missing embedding.embedding or layers.0.FFN.w1"
        ) from exc
    num_layers = 0
    while f"layers.{num_layers}.FFN.w1" in state:
        num_layers += 1
    if num_layers < 1:
        raise KeyError("checkpoint has no transformer layers")
    lm_head = state["Wlm"].shape
    if lm_head != (vocab_size, d_model):
        raise ValueError(f"Wlm shape {tuple(lm_head)} != {(vocab_size, d_model)}")
    return {
        "vocab_size": int(vocab_size),
        "d_model": int(d_model),
        "d_ff": int(d_ff),
        "num_layers": int(num_layers),
    }


# Count non-overlapping sequences that can be extracted from the token memmap.
def window_count(n_tokens, context_length):
    return max(0, (n_tokens - 1) // context_length)


#this is only used for full evluation post trianing.
def iter_windows(encoded_path, batch_size, context_length, device):
    tokens = _get_memmap(encoded_path)
    n_windows = window_count(len(tokens), context_length)
    if n_windows == 0:
        raise ValueError(
            f"{encoded_path} has {len(tokens)} tokens, need at least {context_length + 1}"
        )
    x_buf = np.empty((batch_size, context_length), dtype=np.int64)
    y_buf = np.empty((batch_size, context_length), dtype=np.int64)
    for start in range(0, n_windows, batch_size):
        rows = min(batch_size, n_windows - start)
        for i in range(rows):
            k = (start + i) * context_length
            x_buf[i] = tokens[k : k + context_length]
            y_buf[i] = tokens[k + 1 : k + 1 + context_length]
        # Blocking copy: the numpy buffers are reused on the next batch.
        x = torch.from_numpy(x_buf[:rows]).to(device)
        y = torch.from_numpy(y_buf[:rows]).to(device)
        yield x, y


#this function build a small validation set during training for quick eval. 
#this small subset has fixed number of sequences.
#note n_sequences is the total number of sequneces within a microbatch. 
def load_fixed_subset(encoded_path, n_sequences, context_length, device):
    tokens = _get_memmap(encoded_path)
    n_windows = window_count(len(tokens), context_length)
    if n_windows == 0:
        raise ValueError(
            f"{encoded_path} has {len(tokens)} tokens, need at least {context_length + 1}"
        )
    #extract sparse sequences across the file. 
    window_ids = np.linspace(0, n_windows - 1, n_sequences).astype(np.int64)
    x = np.empty((n_sequences, context_length), dtype=np.int64)
    y = np.empty((n_sequences, context_length), dtype=np.int64)
    for i, w in enumerate(window_ids):
        k = int(w) * context_length
        x[i] = tokens[k : k + context_length]
        y[i] = tokens[k + 1 : k + 1 + context_length]
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


# Score a preloaded subset in batches and return the mean loss.
def quick_eval(model, x_all, y_all, batch_size):
    was_training = model.training
    model.eval()
    loss_sum = torch.zeros((), device=x_all.device, dtype=torch.float32)
    with torch.no_grad():
        #iterate over the microbatch to compute loss score.
        for i in range(0, x_all.shape[0], batch_size):
            x = x_all[i : i + batch_size]
            y = y_all[i : i + batch_size]
            loss = model(x, targets=y, return_logits=False)
            loss_sum += loss.float() * x.shape[0]
    if was_training:
        model.train()
    return (loss_sum / x_all.shape[0]).item()


# Run the model with no gradients and return the token-weighted mean loss.
def evaluate(model, encoded_path, batch_size, context_length, device):
    model.eval()
    loss_sum = 0.0
    token_count = 0
    n_batches = 0
    with torch.inference_mode():
        #note this iterates over the yielded microbatches, not the full set. 
        for x, y in iter_windows(encoded_path, batch_size, context_length, device):
            loss = model(x, targets=y, return_logits=False)
            n_tok = x.shape[0] * x.shape[1]
            loss_sum += loss.item() * n_tok
            token_count += n_tok
            n_batches += 1
            del x, y, loss
            #log the progress every LOG_EVERY(50) batches.
            if n_batches % LOG_EVERY == 0:
                print(
                    f"batch {n_batches}  tokens={token_count:,}  "
                    f"running_loss={loss_sum / token_count:.4f}",
                    flush=True,
                )
    if token_count == 0:
        raise RuntimeError("validation produced no tokens")
    return loss_sum / token_count, token_count, n_batches




# Load the checkpoint, score the full validation set, and return the mean loss.
def run_full_eval(num_heads=NUM_HEADS, max_seq=MAX_SEQ, theta=THETA, batch_size=BATCH_SIZE):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    encode_validation(VALID_PATH, ENCODED_PATH, VOCAB_PATH, MERGES_PATH)

    state, iteration = load_weights(checkpoint_path)
    arch = arch_from_state(state)

    device = pick_device()
    dtype = pick_dtype(device)
    print(
        f"checkpoint={checkpoint_path.name} step={iteration} device={device} dtype={dtype}\n"
        f"  d_model={arch['d_model']} heads={num_heads} layers={arch['num_layers']} "
        f"d_ff={arch['d_ff']} vocab={arch['vocab_size']} seq={max_seq} batch={batch_size}"
    )

    model = Full_llm(
        arch["d_model"],
        num_heads,
        arch["d_ff"],
        max_seq,
        theta,
        arch["num_layers"],
        arch["vocab_size"],
        weights=None,
        device=device,
        dtype=dtype,
    ).to(device)
    _unwrap_model(model).load_state_dict(state)
    del state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mean_loss, token_count, n_batches = evaluate(
        model, ENCODED_PATH, batch_size, max_seq, device
    )
    n_tokens = len(_get_memmap(ENCODED_PATH))
    n_windows = window_count(n_tokens, max_seq)
    scored_windows = token_count // max_seq
    print(
        f"validation loss: {mean_loss:.4f}\n"
        f"tokens scored:   {token_count:,}\n"
        f"windows:         {scored_windows:,} / {n_windows:,}\n"
        f"batches:         {n_batches:,}"
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return mean_loss


if __name__ == "__main__":
    run_full_eval()
