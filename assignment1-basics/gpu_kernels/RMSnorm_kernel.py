import torch
import triton
import triton.language as tl




def launch_RMSnorm(x, gi, eps):
    assert x.is_cuda and gi.is_cuda
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert gi.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert eps > 0

    B, S, D = x.shape
    #whole row of d_model handled by one program; tl.arange needs a power of 2
    BLOCK_D = triton.next_power_of_2(D)
    out = torch.empty_like(x)
    grid = (S, B)
    _RMSnorm_kernel[grid](
        x, gi, out, eps,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        D, BLOCK_D)
    return out


@triton.jit
def _RMSnorm_kernel(
    x_ptr, gi_ptr, out_ptr, eps,
    stride_xb, stride_xs, stride_xd,
    stride_ob, stride_os, stride_od,
    D,
    BLOCK_D: tl.constexpr):

    pid_s = tl.program_id(0)#sequence index
    pid_b = tl.program_id(1)#batch index

    offs_d = tl.arange(0, BLOCK_D) #offsets along d_model within this token
    mask = offs_d < D

    x_ptrs = x_ptr + pid_b * stride_xb + pid_s * stride_xs + offs_d * stride_xd
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    gi = tl.load(gi_ptr + offs_d, mask=mask, other=0.0).to(tl.float32)

    rms_scale = tl.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    out = x * rms_scale * gi

    out_ptrs = out_ptr + pid_b * stride_ob + pid_s * stride_os + offs_d * stride_od
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=mask)
