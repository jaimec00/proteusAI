import torch
import torch.nn.functional as F
from utils.model_utils import protein_to_wavefunc
import triton
import triton.language as tl
from wf_triton_test import calculate_error

def main():

    device = torch.device("cuda")

    torch.manual_seed(37)

    batch, N, d_model = 1, 128, 128
    nheads = 1
    assert d_model%2==0 and d_model%nheads==0
    d_k = d_model // nheads
    min_wl, max_wl, base = 3.7, 20, 20

    coords = max_wl * torch.randn((batch, N, 3), dtype=torch.float64, device=device) # batch x N x 3
    # wf = protein_to_wavefunc(coords, d_model, min_wl, max_wl, base, mask=None).to(torch.float32) # batch x N x d_model
    wf = torch.randn((batch, N, d_model), dtype=torch.float32, device=device) # batch x N x d_model
    mask = torch.rand((batch, N), device=coords.device) > 1 # batch x N

    Q_proj = torch.randn((nheads, d_model, d_k), device=wf.device, dtype=torch.float32) # nhead x d_model x d_k 
    K_proj = torch.randn((nheads, d_model, d_k), device=wf.device, dtype=torch.float32) # nhead x d_model x d_k 
    V_proj = torch.randn((nheads, d_model, d_k), device=wf.device, dtype=torch.float32) # nhead x d_model x d_k 

    # print(f"{Q_proj.mean().item()=}{Q_proj.std().item()=}\n{K_proj.mean().item()=}{K_proj.std().item()=}\n{V_proj.mean().item()=}{V_proj.std().item()=}\n{wf.mean().item()=}{wf.std().item()=}")

    Q = torch.matmul(wf[:, None, :, :], Q_proj[None, :, :, :]) # batch x 1 x N x d_model @ 1 x nhead x d_model x d_k -> batch x nhead x N x d_k
    K = torch.matmul(wf[:, None, :, :], K_proj[None, :, :, :]) # batch x 1 x N x d_model @ 1 x nhead x d_model x d_k -> batch x nhead x N x d_k
    V = torch.matmul(wf[:, None, :, :], V_proj[None, :, :, :]) # batch x 1 x N x d_model @ 1 x nhead x d_model x d_k -> batch x nhead x N x d_k

    # getting numerical instability with exponential computations, so will apply layer norm after projections in real model, but for
    # now just initialize a normal distribution to test
    Q = torch.normal(mean=0, std=1, size=(batch, nheads, N, d_k), device=wf.device, dtype=torch.float32) # nhead x d_model x d_k 
    K = torch.normal(mean=0, std=1, size=(batch, nheads, N, d_k), device=wf.device, dtype=torch.float32) # nhead x d_model x d_k 
    V = torch.normal(mean=0, std=1, size=(batch, nheads, N, d_k), device=wf.device, dtype=torch.float32) # nhead x d_model x d_k 

    # print(f"{Q.mean().item()=}{Q.std().item()=}\n{K.mean().item()=}{K.std().item()=}\n{V.mean().item()=}{V.std().item()=}")

    # raise ValueError

    spreads = min_wl + (torch.logspace(0, 1, nheads, base, dtype=torch.float32, device=coords.device) - 1) / (base-1) * (max_wl-min_wl) # nheads,

    torch_out = torch_attn(Q, K, V, coords, spreads, mask=mask) # batch x N x d_model

    print(f"{torch_out=}, {torch_out.shape=}")

    triton_out = triton_attn.forward(Q, K, V, coords, spreads, mask=mask) # batch x N x d_model

    print(f"{triton_out=}, {triton_out.shape=}")

    print(torch.allclose(torch_out, triton_out))

    print(f"error: {calculate_error(torch_out, triton_out)}%")

@triton.jit
def _triton_attn_kernel(
    O_ptr,
    stride_O_Z,
    stride_O_H,
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

    L_ptr,
    stride_L_Z,
    stride_L_H,
    stride_L_N,

    mask_ptr,
    stride_mask_Z,
    stride_mask_N,

    tot_N:tl.constexpr,
    tot_Z:tl.constexpr,
    nheads:tl.constexpr,
    d_k:tl.constexpr,
    min_d_k:tl.constexpr,

    BLOCK_I:tl.constexpr,
    BLOCK_J:tl.constexpr,
):
    # get block info
    start_I = tl.program_id(0)
    start_ZH = tl.program_id(1) # note that exactly batch*nheads launched along y axis, so no need to mask this
    start_Z = (start_ZH // nheads).to(tl.int64)
    start_H = (start_ZH % nheads).to(tl.int64)

    I_offs = (start_I.to(tl.int64)*BLOCK_I).to(tl.int32)

    # create Q, K, V, and O block pointers
    Qi_block_ptr = tl.make_block_ptr( # N x d_k
        base=Q_ptr + (start_Z*stride_Q_Z) + (start_H*stride_Q_H),
        shape=(tot_N, d_k),
        strides=(stride_Q_N, stride_Q_D),
        offsets=(I_offs, 0),
        block_shape=(BLOCK_I, min_d_k),
        order=(0, 1)
    )

    # transpose k when loading directly by flipping N and D, as well as the Nstride and Dstride
    KjT_block_ptr = tl.make_block_ptr( # d_k x N
        base=K_ptr + (start_Z*stride_K_Z) + (start_H*stride_K_H),
        shape=(d_k, tot_N),
        strides=(stride_K_D, stride_K_N),
        offsets=(0, 0),
        block_shape=(min_d_k, BLOCK_J),
        order=(1, 0)
    )

    Vj_block_ptr = tl.make_block_ptr( # N x d_k
        base=V_ptr + (start_Z*stride_V_Z) + (start_H*stride_V_H),
        shape=(tot_N, d_k),
        strides=(stride_V_N, stride_V_D),
        offsets=(0, 0),
        block_shape=(BLOCK_J, min_d_k),
        order=(0, 1)
    )

    # load the Qi block first, out of bounds values are 0
    Qi_block = tl.load(Qi_block_ptr, boundary_check=(0,1), padding_option="zero") # N x d_k

    # initialize output and statistics block
    Oi_block = tl.zeros_like(Qi_block)
    li_block = tl.zeros((BLOCK_I, ), dtype=tl.float32)
    mi_block = tl.zeros_like(li_block) - float("inf") 

    # make an identity so can perform mat mults from linear arrays easily
    identity_block_i = tl.where(
        tl.arange(0, BLOCK_I)[:, None] == tl.arange(0, BLOCK_I)[None, :],
        1.0,
        0.0
    )

    for j in tl.range(0, triton.cdiv(tot_N, BLOCK_J), 1):

        KjT_block = tl.load(KjT_block_ptr, boundary_check=(0,1), padding_option="zero") # d_k x N
        Vj_block = tl.load(Vj_block_ptr, boundary_check=(0,1), padding_option="zero") # N x d_k

        Sij = tl.dot(Qi_block, KjT_block) / tl.sqrt(d_k*1.0) # N x N

        tl.device_print("QKT/sqrt(dk)=", Sij)

        # join mi and Sij_max to get N x 2, then compute max along axis 1 to get mij of shape N,
        mij = tl.max(tl.join(mi_block, tl.max(Sij, axis=1)), axis=1) # N, 

        Pij = tl.exp(Sij - mij[:, None]) # N x N

        lij = tl.exp(mi_block - mij)*li_block + tl.sum(Pij, axis=1) # N, 

        diag_exp_inv = tl.where(
            (mi_block - mij)==float("-inf"), 
            0.0, 
            1 / tl.exp(mi_block - mij)
        )[:, None] * identity_block_i # N x N 

        Oi_block = tl.dot(diag_exp_inv, Oi_block) + tl.dot(Pij, Vj_block) # N x d_k

        li_block = lij
        mi_block = mij

        KjT_block_ptr = tl.advance(KjT_block_ptr, (0, BLOCK_J))
        Vj_block_ptr = tl.advance(Vj_block_ptr, (BLOCK_J, 0))
    
    diag_li = identity_block_i*li_block[:, None]
    diag_li_inv = tl.where(diag_li==0, 0.0, 1/diag_li)
    Oi_block = tl.dot(diag_li_inv, Oi_block)
    Li_block = mi_block + tl.log(li_block)

    Oi_block_ptr = tl.make_block_ptr( # N x d_k
        base=O_ptr + (start_Z*stride_O_Z) + (start_H*stride_O_H),
        shape=(tot_N, d_k),
        strides=(stride_O_N, stride_O_D),
        offsets=(I_offs, 0),
        block_shape=(BLOCK_I, min_d_k),
        order=(0, 1)
    )

    Li_block_ptr = tl.make_block_ptr( # N,
        base=L_ptr + (start_Z*stride_L_Z) + (start_H*stride_L_H),
        shape=(tot_N, ),
        strides=(stride_L_N, ),
        offsets=(I_offs, ),
        block_shape=(BLOCK_I, ),
        order=(0, )
    )

    tl.store(Oi_block_ptr, Oi_block, boundary_check=(0,1))
    tl.store(Li_block_ptr, Li_block, boundary_check=(0,))

class triton_attn(torch.autograd.Function):

    @staticmethod
    def forward(Q, K, V, coords, spreads, mask=None, dist_factor=3.0):
        
        assert (Q.shape == K.shape) and (K.shape == V.shape), f"Q, K, and V projection shapes must match, but got {Q.shape=}, {K.shape=}, {V.shape=}"
        batch, nheads, N, d_k = Q.shape
        d_model = nheads*d_k

        assert d_model % 2 == 0, f"d_model must be divisible by 2, not {d_model=}"
        assert coords.dim() == 3 and coords.size(2) == 3, f"coordinates must be of shape (batch, N, 3), not {coords.shape}" 

        assert spreads.size(0) == nheads, f"number of spreads must be equal to nheads, not {spreads.size(0)=} and {nheads=}"
        assert torch.all(spreads != 0), f"spreads must be a tensor of non-zero floats, not {spreads}"

        mask = (torch.ones(batch, N, device=Q.device) if mask is None else ~mask).contiguous() # batch x N
        out = torch.zeros(batch, nheads, N, d_k, device=Q.device).contiguous() # batch x N x d_model
        L = torch.zeros(batch, nheads, N, device=Q.device).contiguous() # batch x nheads x N

        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()
        coords = coords.contiguous()
        spreads = spreads.contiguous()

        # available bytes in SRAM, assume 128 KB for now, but plan to make it dynamic by checking SRAM otf
        M = 128 * 1024

        BLOCK_I = 16 #triton.cdiv(M, 4*d_k*BLOCK_ZH) # available bytes per block / bytes per (Q_row,batch,head combination) = (Q_row,batch,head combinations) per block
        BLOCK_J = 32 #BLOCK_I #min(BLOCK_Br, d_k) # number of KV columns per block (all Bc blocks processed sequentially within a Z x H x Br block)

        grid = lambda args: (   triton.cdiv(args["tot_N"], args["BLOCK_I"]), 
                                args["tot_Z"]*args["nheads"],
                                1)

        _triton_attn_kernel[grid](  out, out.stride(0), out.stride(1), out.stride(2), out.stride(3), # batch x nheads x N x d_k
                                    Q, Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), # batch x nhead x N x d_k
                                    K, K.stride(0), K.stride(1), K.stride(2), K.stride(3), # batch x nhead x N x d_k
                                    V, V.stride(0), V.stride(1), V.stride(2), V.stride(3), # batch x nhead x N x d_k
                                    coords, coords.stride(0), coords.stride(1), coords.stride(2), # batch x N x 3
                                    spreads, spreads.stride(0), # nhead, 
                                    L, L.stride(0), L.stride(1), L.stride(2), # batch x nhead x N
                                    mask, mask.stride(0), mask.stride(1), # batch x N
                                    N, batch, nheads, d_k, max(d_k, 16),
                                    BLOCK_I, BLOCK_J
                                )

        return out


def torch_attn(Q, K, V, coords, spreads, mask=None, dist_factor=3.0):

    assert (Q.shape == K.shape) and (K.shape == V.shape), f"Q, K, and V projection shapes must match, but got {Q.shape=}, {K.shape=}, {V.shape=}"
    batch, nheads, N, d_k = Q.shape
    
    d_model = d_k * nheads
    assert d_model % 2 == 0, f"d_model must be divisible by 2, not {d_model=}"
    
    assert coords.dim() == 3 and coords.size(2) == 3, f"coordinates must be of shape (batch, N, 3), not {coords.shape}" 

    assert spreads.size(0) == nheads, f"number of spreads must be equal to nheads, not {spreads.size(0)=} and {nheads=}"
    assert torch.all(spreads != 0), f"spreads must be a tensor of non-zero floats, not {spreads}"
    mask = torch.zeros(batch, N) if mask is None else mask # batch x N

    S = torch.matmul(Q, K.transpose(-2,-1)) / (d_k**0.5) # batch x nheads x N x N
    S = S.masked_fill(mask[:, None, :, None] | mask[:, None, None, :],  float("-inf")) 
    P = torch.softmax(S, dim=-1)
    out = torch.matmul(P, V) # batch x nheads x N x d_k
    
    # out = out.view(batch, N, d_model) # batch x N x d_model

    return out


if __name__ == '__main__':
    main()