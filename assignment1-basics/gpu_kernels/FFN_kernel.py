import torch
import triton
import triton.language as tl


#x: [B, S, D_M]; W1, W3: [D_M, dff]; W2: [dff, D_M] (pass transposed views of nn.Module weights)
def launch_FFN(x, W1, W2, W3):
    assert x.is_cuda and W1.is_cuda and W2.is_cuda and W3.is_cuda

    B, S, D_M = x.shape
    _, dff = W1.shape
    Out = torch.zeros((B, S, D_M), device=x.device, dtype=torch.float32)

    BLOCK_M = 32
    BLOCK_F = 64
    D_Hm = 64 #"heads" dim on d_model

    grid = (triton.cdiv(S, BLOCK_M), triton.cdiv(dff, BLOCK_F), B)


    _FFN_kernel[grid](x, W1, W2, W3, Out,
                      x.stride(0), x.stride(1), x.stride(2), #b s d
                      W1.stride(0), W1.stride(1), #d dff
                      W2.stride(0), W2.stride(1), 
                      W3.stride(0), W3.stride(1), 
                      Out.stride(0), Out.stride(1), Out.stride(2), #b s d
                      S, D_M, dff,
                      D_Hm, BLOCK_M, BLOCK_F)
    return Out.to(x.dtype)


@triton.jit
def _FFN_kernel(
    x_ptr, W1_ptr, W2_ptr, W3_ptr, Out_ptr,
    stride_xb, stride_xs, stride_xd,
    stride_w1d, stride_w1df,
    stride_w2df, stride_w2d,
    stride_w3d, stride_w3df,
    stride_ob, stride_os, stride_od,
    S, D_M, DFF,
    D_Hm:tl.constexpr, BLOCK_M:tl.constexpr, BLOCK_F:tl.constexpr):
    
    row_tile = tl.program_id(0) #row tile idx
    col_tile = tl.program_id(1)
    start_b = tl.program_id(2) # Batch index

    b_idx = x_ptr + start_b * stride_xb
    offs_m = row_tile * BLOCK_M + tl.arange(0, BLOCK_M) #tile idx along seq dim
    offs_f = col_tile * BLOCK_F + tl.arange(0, BLOCK_F) #tile idx along dff dim
    dhm_idx_base = tl.arange(0, D_Hm)  # Constant base range
    mask_m = offs_m < S
    mask_f = offs_f < DFF #mask off dff tiles that don't fit in this block.

    #allocate a tile of out note not whole row or col.
    #x@W1 and x@W3 are accumulated separately: SiLU is nonlinear, so it can only be
    #applied once the full d_model reduction is done
    acc_1 = tl.zeros([BLOCK_M, BLOCK_F], dtype=tl.float32)
    acc_3 = tl.zeros([BLOCK_M, BLOCK_F], dtype=tl.float32)

    #sweep of row column dot which sets up intermediate grid
    for i in range(0, D_M, D_Hm):
        #sweeps horizontally over row tiles in X. avoid dynamic idx in compile time.
        dhm_idx = i + dhm_idx_base
        mask_k = dhm_idx < D_M #mask off d_model tiles that don't fit in this block.

        x_off = offs_m[:, None] * stride_xs + dhm_idx[None, :] * stride_xd
        w1_off = dhm_idx[:, None] * stride_w1d + offs_f[None, :] * stride_w1df
        w3_off = dhm_idx[:, None] * stride_w3d + offs_f[None, :] * stride_w3df

        w1 = tl.load(W1_ptr + w1_off, mask=mask_k[:, None] & mask_f[None, :], other=0.0)
        w3 = tl.load(W3_ptr + w3_off, mask=mask_k[:, None] & mask_f[None, :], other=0.0)
        x = tl.load(b_idx + x_off, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        acc_1 += tl.dot(x, w1)
        acc_3 += tl.dot(x, w3)
    h = (acc_1 * tl.sigmoid(acc_1)) * acc_3
   
    #second sweep of W2
    for j in range(0, D_M, D_Hm):
        dhm_idx = j + dhm_idx_base
        mask_n = dhm_idx < D_M

        out_off = start_b * stride_ob + offs_m[:, None] * stride_os + dhm_idx[None, :] * stride_od
        w2_off = offs_f[:, None] * stride_w2df + dhm_idx[None, :] * stride_w2d
        w2 = tl.load(W2_ptr + w2_off, mask=mask_f[:, None] & mask_n[None, :], other=0.0)

        out_tile = tl.dot(h.to(w2.dtype), w2)
        tl.atomic_add(Out_ptr + out_off, out_tile, mask=mask_m[:, None] & mask_n[None, :])
