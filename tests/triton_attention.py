import torch
import torch.nn.functional as F
from utils.model_utils import protein_to_wavefunc
import triton
import triton.language as tl

def main():

    device = torch.device("cuda")

    batch, N, d_model = 1, 4, 8
    nheads = 2
    assert d_model%2==0 and d_model%nheads==0
    d_k = d_model // nheads
    min_wl, max_wl, base = 3.7, 20, 20

    coords = max_wl * torch.tensor([[[n,n,n] for n in range(N)] for b in range(batch)], dtype=torch.float64, device=device) # batch x N x 3
    wf = protein_to_wavefunc(coords, d_model, min_wl, max_wl, base, mask=None) # batch x N x d_model
    mask = torch.rand((batch, N), device=coords.device) > 1 # batch x N

    Q_proj = torch.randn((nheads, d_model, d_k), device=wf.device, dtype=torch.float64) # nhead x d_model x d_k 
    K_proj = torch.randn((nheads, d_model, d_k), device=wf.device, dtype=torch.float64) # nhead x d_model x d_k 
    V_proj = torch.randn((nheads, d_model, d_k), device=wf.device, dtype=torch.float64) # nhead x d_model x d_k 

    Q = torch.matmul(wf[:, None, :, :], Q_proj[None, :, :, :]) # batch x 1 x N x d_k @ 1 x nhead x d_model x d_k -> batch x nhead x N x d_k
    K = torch.matmul(wf[:, None, :, :], K_proj[None, :, :, :]) # batch x 1 x N x d_k @ 1 x nhead x d_model x d_k -> batch x nhead x N x d_k
    V = torch.matmul(wf[:, None, :, :], V_proj[None, :, :, :]) # batch x 1 x N x d_k @ 1 x nhead x d_model x d_k -> batch x nhead x N x d_k

    spreads = min_wl + (torch.logspace(0, 1, nheads, base, dtype=torch.float64, device=coords.device) - 1) / (base-1) * (max_wl-min_wl) # nheads,

    torch_out = torch_attn(Q, K, V, coords, spreads, mask=mask) # batch x N x d_model

    print(f"{torch_out=}, {torch_out.shape=}")

    triton_out = torch_attn(Q, K, V, coords, spreads, mask=mask) # batch x N x d_model

    print(f"{triton_out=}, {triton_out.shape=}")

def _triton_attn_kernel(
    O_ptr,
    stride_O_Z,
    stride_O_N,
    stride_O_D,

    Q_ptr,
    stride_Q_Z,
    stride_Q_H,
    stride_Q_N,
    stride_Q_D,
    
    K_ptr,
    stride_K_Z,
    stride_K_H,
    stride_K_N,
    stride_K_D,
    
    V_ptr,
    stride_V_Z,
    stride_V_H,
    stride_V_N,
    stride_V_D,

    coords_ptr,
    stride_coords_Z,
    stride_coords_N,
    stride_coords_S,

    spread_ptr,
    stride_spread_H,

    l_ptr,
    stride_l_Z,
    stride_l_H,
    stride_l_N,

    m_ptr,
    stride_m_Z,
    stride_m_H,
    stride_m_N,

    mask_ptr,
    stride_mask_Z,
    stride_mask_N,

    tot_N:tl.constexpr,
    tot_Z:tl.constexpr,
    nheads:tl.constexpr,
    d_k:tl.constexpr,

    BLOCK_Br:tl.constexpr,
    BLOCK_Bc:tl.constexpr,
    BLOCK_ZH:tl.constexpr,
):
    # rows of Q (computed in parallel)
    Br = tl.program_id(0)*BLOCK_Br + tl.arange(0, BLOCK_Br)[None, :, None]

    # batch and heads
    ZH = tl.program_id(1)*BLOCK_ZH + tl.arange(0, BLOCK_ZH)[:, None, None]
    Z = ZH // nheads
    H = ZH % nheads
    
    # d_k
    D = tl.arange(0, d_k)[None, None, :]

    # initialize Q row pointer
    Qr_ptr = Q_ptr + (Z*stride_Q_Z) + (H*stride_Q_H) + (Br*stride_Q_N) + (D*stride_Q_D) # ZH x Br x D

    # initialize O row pointer
    Or_ptr = O_ptr + (Z*stride_O_Z) + (Br*stride_O_N) + ((D + (d_k*H))*stride_O_D ) # ZH x Br x D

    # initialize l row pointer
    lr_ptr = tl.reshape(l_ptr + (Z*stride_l_Z) + (H*stride_l_H) + (Br*stride_l_N), (BLOCK_ZH, BLOCK_Br)) # ZH x Br
    
    # initialize m row pointer
    mr_ptr = tl.reshape(m_ptr + (Z*stride_m_Z) + (H*stride_m_H) + (Br*stride_m_N), (BLOCK_ZH, BLOCK_Br)) # ZH x Br

    # initialize_masks
    Br_mask_ptr = mask_ptr + (Z*stride_mask_Z) + (H*0) + (Br*stride_mask_N) + (D*0) # ZH x Br x D
    Br_mask = (Z < tot_Z) & (H < nheads) & (Br < tot_N) & (D < d_k) # ZH x Br x D
    Br_mask = tl.load(Br_mask_ptr, mask=Br_mask, other=0).to(tl.int1)

    lmr_mask_ptr = tl.reshape(mask_ptr + (Z*stride_mask_Z) + (H*0) + (Br*stride_mask_N), (BLOCK_ZH, BLOCK_Br)) # ZH x Br
    lmr_mask = tl.reshape((Z < tot_Z) & (H < nheads) & (Br < tot_N), (BLOCK_ZH, BLOCK_Br)) # ZH x Br
    lmr_mask = tl.load(lmr_mask_ptr, mask=lmr_mask, other=0).to(tl.int1)

    # load Q, O, l, m
    Qr = tl.load(Qr_ptr, mask=Br_mask, other=0) # ZH x Br x D
    Or = tl.load(Or_ptr, mask=Br_mask, other=0) # ZH x Br x D
    lr = tl.load(lr_ptr, mask=lmr_mask, other=0) # ZH x Br 
    mr = tl.load(mr_ptr, mask=lmr_mask, other=0) # ZH x Br 

    # columns of K and V (computed in series, in the for loop)
    Bc = tl.arange(0, BLOCK_Bc)[None, :, None] # 1 x Bc x 1
    for c in range(triton.cdiv(N, BLOCK_Bc)):

        # advance the block pointer
        Bc = Bc + c*BLOCK_Bc

        # load KV column mask
        KVc_mask_ptr = mask_ptr + (Z*stride_mask_Z) + (H*0) + (Bc*stride_mask_N) + (D*0) # ZH x Bc x D
        KVc_mask = (Z < tot_Z) & (H < nheads) & (Bc < tot_N) & (D < d_k) # ZH x Bc x D
        KVc_mask = tl.load(KVc_mask_ptr, mask=KVc_mask, other=0).to(tl.int1)

        # initialize KV column pointers
        Kc_ptr = K_ptr + (Z*stride_K_Z) + (H*stride_K_H) + (Bc*stride_K_N) + (D*stride_K_D) # ZH x Bc x D
        Vc_ptr = V_ptr + (Z*stride_V_Z) + (H*stride_V_H) + (Bc*stride_V_N) + (D*stride_V_D) # ZH x Bc x D

        # load K and V 
        Kc = tl.load(Kc_ptr, mask=KVc_mask, other=0) # ZH x Bc x D
        Vc = tl.load(Vc_ptr, mask=KVc_mask, other=0) # ZH x Bc x D

        # transpose K
        KcT = tl.permute(Kc, (0, 2, 1)) # ZH x D x Bc

        # attn
        Src = tl.dot(Qr, KcT) / tl.sqrt(d_k) # ZH x Br x Bc
        
        # ----------------------------------------------------------------------
        # skip gaussian scaling for now, ensure attention works normally first
        # ----------------------------------------------------------------------

        # compute attention mask
        Src_Br_mask = tl.sum(Br_mask.to(tl.int16), axis=-1) > 0 # ZH x Br x 1
        Src_Bc_mask = tl.sum(Bc_mask.to(tl.int16), axis=-1) > 0 # ZH x Bc x 1
        Src_Bc_mask = tl.permute(Src_Bc_mask, (0,2,1)) # ZH x 1 x Bc
        Src_mask = Src_Br_mask & Src_Bc_mask # ZH x Br x Bc

        # mask attn logits
        Src = tl.where(Src_mask, Src, float("-inf")) # ZH x Br x Bc

        # compute this block's m
        mrc = tl.max(Src, axis=2) # ZH x Br

        # compute this blocks P
        Prc = tl.exp(Src - mrc) # ZH x Br x Bc

        # compute this block's l
        lrc = tl.sum(Prc, axis=2) # ZH x Br

        # join mr and mrc along final axis, then reduces along that axis to find max between the two 
        # (intermediate is ZH x Br x 2 )
        mr_new = tl.max(tl.join(mr, mrc), axis=-1) # ZH x Br

        # compute new lr for this block
        lr_new = tl.exp(mr - mr_new) * lr + tl.exp(mrc - mr_new) * lrc # ZH x Br

        # ----------------------------------------------------------------------
        # skip RNG for dropout for now
        # ----------------------------------------------------------------------

        # update output
        lr_new_inv = (1 / tl.where(lr_new==0, float("inf"), lr_new))[:, :, None]
        Or_term1 = (lr * tl.exp(mr - mr_new))[:, :, None] * Or
        Or_term2 = tl.exp(mrc - mr_new)[:, :, None] * tl.dot(Prc, Vc)
        Or = lr_new_inv * (Or_term1 + Or_term2)

        # update statistics
        lr = lr_new
        mr = mr_new

    # once loop through all columns for assigned rows, store the output
    tl.store(Or_ptr, Or, mask=Br_mask)
    tl.store(lr_ptr, lr, mask=lmr_mask)
    tl.store(mr_ptr, mr, mask=lmr_mask)

def triton_attn(Q, K, V, coords, spreads, mask=None, dist_factor=3.0):
    
    assert (Q.shape == K.shape) and (K.shape == V.shape), f"Q, K, and V projection shapes must match, but got {Q.shape=}, {K.shape=}, {V.shape=}"
    batch, nheads, N, d_k = Q.shape
    d_model = nheads*d_k

    assert d_model % 2 == 0, f"d_model must be divisible by 2, not {d_model=}"
    assert coords.dim() == 3 and coords.size(2) == 3, f"coordinates must be of shape (batch, N, 3), not {coords.shape}" 

    assert spreads.size(0) == nheads, f"number of spreads must be equal to nheads, not {spreads.size(0)=} and {nheads=}"
    assert torch.all(spreads != 0), f"spreads must be a tensor of non-zero floats, not {spreads}"

    mask = torch.zeros(batch, N, device=Q.device) if mask is None else mask # batch x N
    out = torch.zeros(batch, N, d_model, device=Q.device) # batch x N x d_model
    l = torch.zeros(batch, nheads, N, device=Q.device) # batch x nheads x N
    m = torch.zeros(batch, nheads, N, device=Q.device) + float("-inf") # batch x nheads x N

    # available bytes in SRAM, assume 128 KB for now, but plan to make it dynamic by checking SRAM otf
    M = 128 * 1024

    BLOCK_ZH = 1 # number of batch, head combos per block
    BLOCK_Br = triton.cdiv(M, 4*d_k*BLOCK_ZH) # available bytes per block / bytes per (Q_row,batch,head combination) = (Q_row,batch,head combinations) per block
    BLOCK_Bc = min(BLOCK_Br, d_k) # number of KV columns per block (all Bc blocks processed sequentially within a Z x H x Br block)

    grid_Br = triton.cdiv(N, BLOCK_Br) # rows of Q (independant in flash attn)
    grid_ZH = triton.cdiv(batch*nhead, BLOCK_ZH) # batch,head

    grid = (grid_Br, grid_ZH)

    _triton_attn_kernel[grid](  out, out.stride(0), out.stride(1), out.stride(2), # batch x N x d_model
                                Q, Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), # batch x nhead x N x d_k
                                K, K.stride(0), K.stride(1), K.stride(2), K.stride(3), # batch x nhead x N x d_k
                                V, V.stride(0), V.stride(1), V.stride(2), V.stride(3), # batch x nhead x N x d_k
                                coords, coords.stride(0), coords.stride(1), coords.stride(2), # batch x N x 3
                                spreads, spreads.stride(0), # nhead, 
                                l, l.stride(0), l.stride(1), l.stride(2), # batch x nhead x N
                                m, m.stride(0), m.stride(1), m.stride(2), # batch x nhead x N
                                mask, mask.stride(0), mask.stride(1), # batch x N
                                N, batch, nheads, d_k,
                                BLOCK_Br, BLOCK_Bc, BLOCK_ZH
                            )

    return out


def torch_attn(wf, coords, nheads, Q_proj, K_proj, V_proj, spreads, mask=None, dist_factor=3.0):

    batch, N, d_model = wf.shape
    assert d_model % 2 == 0, f"d_model must be divisible by 2, not {d_model=}"
    assert d_model % nheads == 0, f"d_model must be divisible by number of heads, not {d_model=}, {nheads=}, {d_model/nheads=}"
    assert coords.dim() == 3 and coords.size(2) == 3, f"coordinates must be of shape (batch, N, 3), not {coords.shape}" 
    d_k = d_model // nheads

    Q_nheads, Q_d_model, Q_d_k = Q_proj.shape
    K_nheads, K_d_model, K_d_k = K_proj.shape
    V_nheads, V_d_model, V_d_k = V_proj.shape

    assert all( (q==k) and (k==v)
                for q, k, v in [
                                    [Q_nheads, K_nheads, V_nheads], 
                                    [Q_d_model, K_d_model, V_d_model], 
                                    [Q_d_k, K_d_k, V_d_k] 
                                ]
            ), f"Q, K, and V projection shapes must match, but got {Q.shape=}, {K.shape=}, {V.shape=}"

    shape_err =     f"Q, K, V projection shapes must match feature shapes and specified parameters. Note Q/K/V shapes are "\
                    f"(nhead, d_model, d_k), but have {Q_proj.shape=}, {K_proj.shape=}, {V_proj.shape=}, with "\
                    f"{d_k=}, {nheads=}, {d_model=}"
    assert Q_nheads == nheads, shape_err
    assert Q_d_model == d_model, shape_err
    assert Q_d_k == d_k, shape_err

    assert spreads.size(0) == nheads, f"number of spreads must be equal to nheads, not {spreads.size(0)=} and {nheads=}"
    assert torch.all(spreads != 0), f"spreads must be a tensor of non-zero floats, not {spreads}"
    mask = torch.zeros(batch, N) if mask is None else mask # batch x N

    Q = torch.matmul(wf[:, None, :, :], Q_proj[None, :, :, :]) # batch x nheads x N x d_k
    K = torch.matmul(wf[:, None, :, :], K_proj[None, :, :, :]) # batch x nheads x N x d_k
    V = torch.matmul(wf[:, None, :, :], V_proj[None, :, :, :]) # batch x nheads x N x d_k

    dists = torch.sqrt(torch.sum(coords[:, :, None, :] - coords[:, None, :, :], dim=-1).abs()) # batch x N x N

    # prepare for broadcasting with target shape of batch x nheads x N x N
    dists = dists[:, None, :, :]
    spreads = spreads[None, :, None, None]

    # clamp distances less than the spreads to the spread
    close_mask = dists < spreads
    dists = torch.where(
        close_mask,
        torch.clamp(dists, min=spreads),
        dists
    )

    # clamp distances > than dist_fac * spreads to dist_fac*spreads
    # will be masked out anyway, this is just for numerical stability 
    far_mask = dists > (dist_factor * spreads)
    dists = torch.where(
        far_mask,
        torch.clamp(dists, max=dist_factor*spreads),
        dists
    )

    print(f"{dists=}")

    # compute rbf
    rbf = torch.exp( (-dists**2) / (2*spreads**2) ) # batch x nheads x N x N
    attn = torch.matmul(Q, K.transpose(-2,-1)) / torch.sqrt(torch.tensor(d_k)) # batch x nheads x N x N

    print(f"{rbf}")
    print(f"{attn}")

    rbf = torch.where(attn < 0, 1/rbf, rbf)
    # attn = attn * rbf
    attn = attn.masked_fill(mask[:, None, :, None],  float("-inf")) # | far_mask,
    attn = F.softmax(attn, dim=-1)

    print(f"{attn}")

    out = torch.matmul(attn, V) # batch x nheads x N x d_k
    
    print(f"{out=}")
    out = out.view(batch, N, d_model) # batch x N x d_model
    print(f"{out=}")

    return out



if __name__ == '__main__':
    main()