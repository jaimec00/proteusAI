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

    coords = max_wl * torch.tensor([[[n,n,n] for n in range(N)] for b in range(batch)], dtype=torch.float64, device=device)
    wf = protein_to_wavefunc(coords, d_model, min_wl, max_wl, base, mask=None)
    mask = torch.rand((batch, N), device=coords.device) > 1

    Q_proj = torch.randn((nheads, d_model, d_k), device=wf.device, dtype=torch.float64)
    K_proj = torch.randn((nheads, d_model, d_k), device=wf.device, dtype=torch.float64)
    V_proj = torch.randn((nheads, d_model, d_k), device=wf.device, dtype=torch.float64)

    spreads = min_wl + (torch.logspace(0, 1, nheads, base, dtype=torch.float64, device=coords.device) - 1) / (base-1) * (max_wl-min_wl)

    torch_out = torch_attn(wf, coords, nheads, Q_proj, K_proj, V_proj, spreads, mask=mask)

    print(torch_out, torch_out.shape)

    

def _triton_attn_kernel(
    output_ptr,
    stride_output_B,
    stride_output_N,
    stride_output_D,

    wf_ptr,
    stride_ptr_B,
    stride_ptr_QK,
    stride_ptr_V,

    coords_ptr,
    stride_coords_B,
    stride_coords_N,
    stride_coords_S,

    spread_ptr,
    stride_spread_B,
    stride_spread_H,

    Q_ptr,
    stride_Q_B,
    stride_Q_N,
    stride_Q_D,
    
    K_ptr,
    stride_K_B,
    stride_K_N,
    stride_K_D,
    
    V_ptr,    
    stride_V_B,
    stride_V_N,
    stride_V_D,

    BLOCK_B:tl.constexpr,
    BLOCK_QK:tl.constexpr,
    BLOCK_V:tl.constexpr
):
    pass

def triton_attn(wf, coords, nheads, Q_proj, K_proj, V_proj, spreads, mask=None, dist_factor=3.0):
    
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

    out = torch.zeros_like(wf)

    # BLOCK_ = 
    # BLOCK_ = 
    # BLOCK_ = 

    # grid_
    # grid_
    # grid_

    # grid = (grid_, grid_, grid_)

    # _triton_attn_kernel[grid](  out, out.stride(0), out.stride(1), out.stride(2),
    #                             wf, wf.stride(0), wf.stride(1), wf.stride(2)
    #                             coords, coords.stride(0), coords.stride(1), coords.stride(2)
    #                             spreads, spreads.stride(0), spreads.stride(1)

    # )


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

    # compute rbf
    rbf = torch.exp( (-dists**2) / (2*spreads**2) ) # batch x nheads x N x N
    attn = torch.matmul(Q, K.transpose(-2,-1)) / torch.sqrt(torch.tensor(d_k)) # batch x nheads x N x N

    rbf = torch.where(attn < 0, 1/rbf, rbf)
    attn = attn * rbf
    attn = attn.masked_fill(mask[:, None, :, None] | far_mask, float("-inf"))
    attn = F.softmax(attn, dim=-1)

    out = torch.matmul(attn, V) # batch x nheads x N x d_k
    out = out.view(batch, N, d_model) # batch x N x d_model

    return out



if __name__ == '__main__':
    main()