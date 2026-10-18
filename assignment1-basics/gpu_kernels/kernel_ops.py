import torch
import scaled_dot_product_attention as sdpa
from RMSnorm import rmsnorm_torch
from positionwise_feedforward import swiglu_torch

try:
    from .RMSnorm_kernel import launch_RMSnorm
    from .FFN_kernel import launch_FFN
    from .flash_attention import launch_flash_attention
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

#set False to force the PyTorch path everywhere (e.g. to compare against the kernels)
USE_KERNELS = True


def kernels_enabled(*tensors):
    return USE_KERNELS and HAS_TRITON and all(t.is_cuda for t in tensors)


def _is_pow2(n):
    return n >= 16 and (n & (n - 1)) == 0


#PyTorch references: used off-GPU, and recomputed in backward since the kernels are forward-only
rmsnorm_ref = rmsnorm_torch
swiglu_ref = swiglu_torch


def causal_attention_ref(q, k, v):
    seq_len = q.size(-2)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device))
    return sdpa.scaled_attention(q, k, v, mask)


def _recompute_grads(ref_fn, inputs, needs_grad, grad_out):
    with torch.enable_grad():
        detached = [t.detach().requires_grad_(need) for t, need in zip(inputs, needs_grad)]
        out = ref_fn(*detached)
        wanted = [t for t in detached if t.requires_grad]
        grads = iter(torch.autograd.grad(out, wanted, grad_out)) if wanted else iter(())
    return [next(grads) if t.requires_grad else None for t in detached]


class _RMSnormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gi, eps):
        ctx.eps = eps
        ctx.save_for_backward(x, gi)
        x3 = x.reshape(1, -1, x.size(-1))
        return launch_RMSnorm(x3, gi, eps).reshape(x.shape)

    @staticmethod
    def backward(ctx, grad_out):
        x, gi = ctx.saved_tensors
        ref = lambda x_, g_: rmsnorm_ref(x_, g_, ctx.eps)
        grad_x, grad_gi = _recompute_grads(ref, (x, gi), ctx.needs_input_grad[:2], grad_out)
        return grad_x, grad_gi, None


class _SwiGLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, w3):
        ctx.save_for_backward(x, w1, w2, w3)
        x3 = x.reshape(-1, x.size(-2), x.size(-1))
        #module weights are [dff, d] / [d, dff]; the kernel wants x @ W layout, strides handle the .T
        return launch_FFN(x3, w1.T, w2.T, w3.T).reshape(*x.shape[:-1], w2.size(0))

    @staticmethod
    def backward(ctx, grad_out):
        return tuple(_recompute_grads(swiglu_ref, ctx.saved_tensors, ctx.needs_input_grad, grad_out))


class _CausalFlashAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        ctx.save_for_backward(q, k, v)
        return launch_flash_attention(q, k, v, causal=True)

    @staticmethod
    def backward(ctx, grad_out):
        return tuple(_recompute_grads(causal_attention_ref, ctx.saved_tensors, ctx.needs_input_grad, grad_out))


def rmsnorm(x, gi, eps):
    if kernels_enabled(x, gi):
        return _RMSnormFn.apply(x, gi, eps)
    return rmsnorm_ref(x, gi, eps)


def swiglu(x, w1, w2, w3):
    if kernels_enabled(x, w1, w2, w3) and x.dim() >= 2:
        return _SwiGLUFn.apply(x, w1, w2, w3)
    return swiglu_ref(x, w1, w2, w3)


def causal_attention(q, k, v, mask):
    """q, k: [B, H, S, D_QK], v: [B, H, S, D_V]. Kernel path assumes `mask` is causal."""
    use_kernel = (
        kernels_enabled(q, k, v)
        and q.dim() == 4
        and _is_pow2(q.size(-1))  #tl.dot needs >= 16, tl.arange needs a power of 2
        and _is_pow2(v.size(-1))
    )
    if use_kernel:
        return _CausalFlashAttnFn.apply(q, k, v)
    return sdpa.scaled_attention(q, k, v, mask)
