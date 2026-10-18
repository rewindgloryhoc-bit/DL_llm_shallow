import torch
import triton
import triton.language as tl



#note q and k here should really be tensor transformed by RoPE already. 
#see separate kernel for RoPE. 
#Q, K: [Z, H, N_CTX, D_QK], V: [Z, H, N_CTX, D_V]; D_QK and D_V may differ (mixture RoPE uses d_head/2 for QK)
def launch_flash_attention(Q, K, V, causal=True):
    assert Q.is_cuda and K.is_cuda and V.is_cuda

    Z, H, N_CTX, D_QK = Q.shape
    D_V = V.shape[-1]
    Out = torch.empty((Z, H, N_CTX, D_V), device=Q.device, dtype=Q.dtype)

    BLOCK_M = 32     
    BLOCK_N = 32     

    grid = (triton.cdiv(N_CTX, BLOCK_M), H, Z)

    #launch kernel, distribute grids across GPU.
    _flash_attn_fwd_kernel[grid](
        Q, K, V, Out,
        
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        
        Z, H, N_CTX,
        1.0 / (D_QK ** 0.5),
        
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DQK=D_QK,
        BLOCK_DV=D_V,
        IS_CAUSAL=causal,
    )
    return Out



@triton.jit
def _flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    Z, H, N_CTX,
    sm_scale,
    BLOCK_M: tl.constexpr,  # Query tile size (e.g., 32)
    BLOCK_N: tl.constexpr,  # Key/Value tile size (e.g., 32)
    BLOCK_DQK: tl.constexpr, # head dim of q/k
    BLOCK_DV: tl.constexpr,  # head dim of v
    IS_CAUSAL: tl.constexpr,
):
    # 1. Map program IDs to Batch, Head, and Query Tile
    start_m = tl.program_id(0) # Query tile index
    off_h   = tl.program_id(1) # Head index
    off_z   = tl.program_id(2) # Batch index

    #everything is indexed into corresponding batch, head locations first.
    Q_ptr += off_z * stride_qb + off_h * stride_qh
    K_ptr += off_z * stride_kb + off_h * stride_kh
    V_ptr += off_z * stride_vb + off_h * stride_vh
    O_ptr = Out_ptr + off_z * stride_ob + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) 
    offs_n = tl.arange(0, BLOCK_N)                      
    offs_dqk = tl.arange(0, BLOCK_DQK)
    offs_dv = tl.arange(0, BLOCK_DV)

    #in regular tensor stride_qd is just 1, in case of tranpose transpose it will be BLOCK_DMODEL
    q_ptrs = Q_ptr + (offs_m[:, None] * stride_qs + offs_dqk[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") # Running Max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  #li is running sum
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)   # Running Output Accumulator

    #causal: keys past the last query of this tile are fully masked, so stop early
    if IS_CAUSAL:
        hi = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M)
    else:
        hi = N_CTX

    #for loop sweeps over seq dimension per 1 head
    for start_n in range(0, hi, BLOCK_N):
        cols = start_n + offs_n #arbitary tile indices
        #note that k is transposed and v is not
        k_ptrs = K_ptr + (cols[None, :] * stride_ks + offs_dqk[:, None] * stride_kd)
        v_ptrs = V_ptr + (cols[:, None] * stride_vs + offs_dv[None, :] * stride_vd)

        k = tl.load(k_ptrs, mask=cols[None, :] < N_CTX, other=0.0)
        v = tl.load(v_ptrs, mask=cols[:, None] < N_CTX, other=0.0)

        qk = tl.dot(q, k) * sm_scale
        #padded keys load as 0, which would still get exp(0 - m) weight; mask them to -inf
        valid = cols[None, :] < N_CTX #this check is redundant if casual mask is on.
        if IS_CAUSAL:
            valid = valid & (cols[None, :] <= offs_m[:, None]) #bool mask
        qk = tl.where(valid, qk, float("-inf"))

        m_ij = tl.max(qk, 1) #max per row dimension                      
        m_next = tl.maximum(m_i, m_ij)              

        alpha = tl.exp(m_i - m_next)#alpha adjust scaling on logits using running max
        p = tl.exp(qk - m_next[:, None])
        l_i = l_i * alpha + tl.sum(p, 1) #rescales running sum using new max
        acc = acc * alpha[:, None] #acc is not normalized with sum yet
        acc += tl.dot(p.to(v.dtype), v) #think in row form for matrix multiplication
        m_i = m_next
    acc = acc / l_i[:, None]

    o_ptrs = O_ptr + (offs_m[:, None] * stride_os + offs_dv[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=offs_m[:, None] < N_CTX)
