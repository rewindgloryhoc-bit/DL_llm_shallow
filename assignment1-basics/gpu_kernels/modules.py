from RMSnorm import RMSnorm
from positionwise_feedforward import SwiGLU
from multihead_self_attention import MultiHeadSelfAttention
from . import kernel_ops


#Same parameters and state_dict keys as the baseline modules; only forward dispatches to Triton.
class KernelRMSnorm(RMSnorm):
    def forward(self, x):
        return kernel_ops.rmsnorm(x, self.gi, self.eps)


class KernelSwiGLU(SwiGLU):
    def forward(self, x):
        return kernel_ops.swiglu(x, self.w1, self.w2, self.w3)


class KernelMultiHeadSelfAttention(MultiHeadSelfAttention):
    attention_fn = staticmethod(kernel_ops.causal_attention)


_KERNEL_CLASSES = {
    RMSnorm: KernelRMSnorm,
    SwiGLU: KernelSwiGLU,
    MultiHeadSelfAttention: KernelMultiHeadSelfAttention,
}


def enable_kernels(model):
    """Swap baseline submodules to their Triton-backed versions in place; returns the model.

    Parameters are untouched, so checkpoints stay interchangeable with the baseline.
    Off CUDA (or without Triton) the swapped modules fall back to the baseline PyTorch math.
    """
    for module in model.modules():
        kernel_cls = _KERNEL_CLASSES.get(type(module))
        if kernel_cls is not None:
            module.__class__ = kernel_cls
    return model
