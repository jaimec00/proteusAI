
import torch

def get_coords(coords, chain_idxs=None): 
    '''
    utility method to get Cb coords
    '''

    # in both cases, ca and cb coords are batch x N x 3
    if coords.dim() == 3: # Ca only model
        if chain_idxs is None:
            raise ValueError("chain_idxs must be provided for Ca only model")
        coords_alpha, coords_beta = get_Cb_from_Ca(coords, chain_idxs)
    elif coords.dim() == 4: # backbone model
        coords_alpha, coords_beta = get_Cb_from_BB(coords)
    else:
        raise ValueError(f"invalid input size for coordinates, expected (batch,N,3) for Ca only model or (batch,N,3,3) for backbone model, but got {coords.shape=}")
    
    return coords_alpha, coords_beta
    
def get_Cb_from_Ca(coordsA, chain_idxs):
    '''
    compute beta carbon coords (not absolute, just relative to Ca). used for anisotropic wf embedding, in testing
    approximates N and C as being on the line connecting adjacent Ca, with ideal bond distances
    uses the same experimentally determined constants as PMPNN to compute linear combination of b1, b2, and b3

    chain_idxs is a list of lists of lists, like:
        [ 
            [ 
                [sample1_chain1_start, sample1_chain1_stop], 
                [sample1_chain2_start, sample1_chain2_stop] 
            ], 
            [
                [sample2_chain1_start, sample2_chain1_stop], 
                [sample2_chain2_start, sample2_chain2_stop] 
            ]
        ]
    
    flattens the indexes to perform efficient batched computation of virtual Cb coordinates
    '''

    batch, N, space = coordsA.shape

    # default cb position is 0,0,0
    coordsB = torch.zeros_like(coordsA)

    # create flattened lists of idxs
    batch_idxs_flat, start_idxs_flat, end_idxs_flat = [], [], []
    for sample_idx, sample in enumerate(chain_idxs):
        for start_idx, stop_idx in sample:
            batch_idxs_flat.append(sample_idx)
            start_idxs_flat.append(start_idx)
            end_idxs_flat.append(stop_idx)

    # convert to flattened tensors and reshape
    batch_idxs_flat = torch.tensor(batch_idxs_flat).unsqueeze(0).unsqueeze(2) # 1 x num_chains x 1
    start_idxs_flat = torch.tensor(start_idxs_flat).unsqueeze(0).unsqueeze(2)  # 1 x num_chains x 1
    end_idxs_flat = torch.tensor(end_idxs_flat).unsqueeze(0).unsqueeze(2) # 1 x num_chains x 1

    # create Z x num_chains x N boolean tensor, where True is the positions corresponding to each chain within its batch
    seq_idxs = torch.arange(N).unsqueeze(0).unsqueeze(0) # 1 x 1 x N
    batch_idxs = torch.arange(batch).unsqueeze(1).unsqueeze(2) # Z x 1 x 1
    is_chain = (batch_idxs == batch_idxs_flat) & (seq_idxs >= start_idxs_flat) & (seq_idxs < end_idxs_flat) # Z x num_chains x N

    # convert the boolean tensors to Z x N
    # create boolean tensors defining if each position acts as a logical N, CA, or C, where N[i] is adjacent to CA[i] is adjacent to C[i]
    # the max works as an OR operation along num chains dim to get Z x N boolean tensor of where each logical backbone atom goes
    is_logical_N = (is_chain & ((seq_idxs+2) < end_idxs_flat)).max(dim=1).values # Z x N
    is_logical_CA = (is_chain & ((seq_idxs-1) >= start_idxs_flat) & ((seq_idxs+1) < end_idxs_flat)).max(dim=1).values # Z x N
    is_logical_C = (is_chain & ((seq_idxs-2) >= start_idxs_flat)).max(dim=1).values # Z x N

    # extract the coordinates for each
    logical_N = coordsA[is_logical_N, :] # num_positions_in_all_chains-(2*num_chains) x 3
    logical_CA = coordsA[is_logical_CA, :] # num_positions_in_all_chains-(2*num_chains) x 3
    logical_C = coordsA[is_logical_C, :] # num_positions_in_all_chains-(2*num_chains) x 3

    # compute the virtual beta carbons
    # shapes are all # num_positions_in_all_chains-(2*num_chains) x 3
    # uses ideal bond lengths
    b1 = logical_CA - logical_N
    b1 = 1.458 * b1 / torch.linalg.vector_norm(b1, dim=1, keepdim=True).clamp(min=1e-6)
    b2 = logical_C - logical_CA
    b2 = 1.525 * b2 / torch.linalg.vector_norm(b2, dim=1, keepdim=True).clamp(min=1e-6)
    b3 = torch.linalg.cross(b1, b2, dim=1)

    # compute virtual cb w/ empirical constants
    virtual_CB = -0.58273431*b2 + 0.56802827*b1 - 0.54067466*b3

    # only the logical CA get a CB, so use the already computed boolean tensor to assign CB
    coordsB[is_logical_CA] = virtual_CB

    # make a unit vector
    coordsB = coordsB / torch.linalg.vector_norm(coordsB, dim=2, keepdim=True).clamp(1e-6)

    return coordsA, coordsB

def get_Cb_from_BB(coords):

    '''
    don't need chain idxs here, since the input is a batch x N x 3(N,Ca,C) x 3 tensor, so can use the coords tensor directly
    no masking necessary, as cb ends up being 0,0,0 for masked vals, since masked coords are 0,0,0
    '''

    n = coords[:, :, 0, :]
    ca = coords[:, :, 1, :]
    c = coords[:, :, 2, :]
    
    b1 = ca - n
    b2 = c - ca
    b3 = torch.linalg.cross(b1, b2, dim=2)

    cb = -0.58273431*b2 + 0.56802827*b1 - 0.54067466*b3

    cb = cb / torch.linalg.vector_norm(cb, dim=2, keepdim=True).clamp(1e-6)

    return ca, cb