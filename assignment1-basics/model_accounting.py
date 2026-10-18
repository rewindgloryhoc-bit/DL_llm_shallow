import argparse
import re
from collections import defaultdict

import torch
from torch.utils.flop_counter import FlopCounterMode

from full_llm import Full_llm
from fused_optimizer import setup_optimizers, OPTIMIZER_CHOICES
from cross_entropy import cross_entropy

'''
Measured (not derived) model accounting: parameter counts, memory, and FLOPs
for one training step of Full_llm. Useful as ground truth when checking
hand-derived scaling formulas.

measure_flops / measure_memory run a real optimizer step, so they mutate the
model's weights; pass a freshly built model (see build_model).

    python model_accounting.py --d-model 512 --num-heads 16 --num-layers 4 \
        --d-ff 1344 --vocab-size 10000 --seq 256 --batch 64 --optimizer adamw
'''


def build_model(d_model, num_heads, num_layers, d_ff, vocab_size, max_seq, theta=10000.0,
                optimizer="muon", device="cpu", dtype=torch.float32):
    model = Full_llm(
        d_model, num_heads, d_ff, max_seq, theta, num_layers, vocab_size,
        weights=None, device=device, dtype=dtype,
    ).to(device)
    optimizer_muon, optimizer_adamw = setup_optimizers(
        model, lr_min=1e-4, lr_max=1e-3, weight_decay=0.01, Tw=1, Tc=10, optimizer=optimizer,
    )
    optimizers = [opt for opt in (optimizer_muon, optimizer_adamw) if opt is not None]
    return model, optimizers


def _component(name):
    # "layers.3.attn.Wq" -> "attn.Wq"; "embedding.embedding" -> "embedding.embedding"
    return re.sub(r"^layers\.\d+\.", "", name)


def count_parameters(model):
    """Returns total/trainable counts plus a per-component breakdown summed over layers."""
    total = 0
    trainable = 0
    by_component = defaultdict(int)
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        by_component[_component(name)] += n
    return {
        "total": total,
        "trainable": trainable,
        "by_component": dict(by_component),
    }


def _random_batch(batch_size, seq_len, vocab_size, device):
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    return x, y


def _flops_of(fn):
    counter = FlopCounterMode(display=False)
    with counter:
        out = fn()
    return out, counter.get_total_flops(), counter.get_flop_counts()


def measure_flops(model, optimizers, batch_size, seq_len, vocab_size):
    """FLOPs of one training step, split into forward / backward / optimizer step.

    Counts matmul-class aten ops only (mm, bmm, addmm, sdpa, ...); elementwise ops
    such as softmax, RMSnorm, RoPE and AdamW's update are not counted. Custom Triton
    kernels are invisible to the counter, so run this on the plain PyTorch path.
    """
    device = next(model.parameters()).device
    x, y = _random_batch(batch_size, seq_len, vocab_size, device)
    model.train()
    for opt in optimizers:
        opt.zero_grad()

    loss, forward_flops, forward_by_module = _flops_of(lambda: cross_entropy(model(x), y))
    _, backward_flops, _ = _flops_of(loss.backward)
    _, step_flops, _ = _flops_of(lambda: [opt.step() for opt in optimizers])

    top_level = {
        name: sum(counts.values())
        for name, counts in forward_by_module.items()
        if name != "Global"
    }
    return {
        "forward": forward_flops,
        "backward": backward_flops,
        "optimizer_step": step_flops,
        "total": forward_flops + backward_flops + step_flops,
        "forward_by_module": top_level,
    }


def _tensor_bytes(t):
    return t.numel() * t.element_size()


def measure_memory(model, optimizers, batch_size, seq_len, vocab_size):
    """Bytes for one training step, split into the four standard buckets.

    - parameters / gradients / optimizer_state: summed from the live tensors.
    - activations: unique tensors saved for backward during the forward pass
      (parameter storages excluded), i.e. what must stay resident until backward.
    - device_peak: allocator peak on CUDA; on MPS the max of samples taken after
      each phase (no peak API); None on CPU.
    """
    device = next(model.parameters()).device
    x, y = _random_batch(batch_size, seq_len, vocab_size, device)
    model.train()
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    mps_samples = []

    def sample_mps():
        if device.type == "mps":
            torch.mps.synchronize()
            mps_samples.append(torch.mps.current_allocated_memory())

    param_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
    saved = {}

    def pack(t):
        storage = t.untyped_storage()
        ptr = storage.data_ptr()
        if ptr not in param_storages:
            saved[ptr] = storage.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = cross_entropy(model(x), y)
    sample_mps()
    loss.backward()
    sample_mps()
    for opt in optimizers:
        opt.step()
    sample_mps()

    parameters = sum(_tensor_bytes(p) for p in model.parameters())
    gradients = sum(_tensor_bytes(p.grad) for p in model.parameters() if p.grad is not None)
    optimizer_state = sum(
        _tensor_bytes(v)
        for opt in optimizers
        for state in opt.state.values()
        for v in state.values()
        if torch.is_tensor(v)
    )
    activations = sum(saved.values())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        device_peak = torch.cuda.max_memory_allocated(device)
    elif device.type == "mps":
        device_peak = max(mps_samples)
    else:
        device_peak = None

    return {
        "parameters": parameters,
        "gradients": gradients,
        "optimizer_state": optimizer_state,
        "activations": activations,
        "total": parameters + gradients + optimizer_state + activations,
        "device_peak": device_peak,
    }


def _fmt_bytes(n):
    if n is None:
        return "n/a"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:,.2f} {unit}"
        n /= 1024


def _fmt_flops(n):
    return f"{n:,} ({n / 1e12:.4f} TFLOPs)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--optimizer", "-o", choices=OPTIMIZER_CHOICES, default="muon")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = dict(
        d_model=args.d_model, num_heads=args.num_heads, num_layers=args.num_layers,
        d_ff=args.d_ff, vocab_size=args.vocab_size, max_seq=args.seq,
        optimizer=args.optimizer, device=device,
    )
    print(f"config: {config}, batch={args.batch}, seq={args.seq}")

    model, _ = build_model(**config)
    params = count_parameters(model)
    print("\n== parameters ==")
    print(f"total: {params['total']:,}  trainable: {params['trainable']:,}")
    for name, n in sorted(params["by_component"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:<28} {n:>14,}")

    model, optimizers = build_model(**config)
    flops = measure_flops(model, optimizers, args.batch, args.seq, args.vocab_size)
    print("\n== FLOPs (one training step, matmul-class ops) ==")
    for key in ("forward", "backward", "optimizer_step", "total"):
        print(f"{key:<16} {_fmt_flops(flops[key])}")
    print("forward by module:")
    for name, n in sorted(flops["forward_by_module"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:<28} {n:>20,}")

    model, optimizers = build_model(**config)
    memory = measure_memory(model, optimizers, args.batch, args.seq, args.vocab_size)
    print("\n== memory (one training step) ==")
    for key in ("parameters", "gradients", "optimizer_state", "activations", "total", "device_peak"):
        print(f"{key:<16} {_fmt_bytes(memory[key])}")


if __name__ == "__main__":
    main()
