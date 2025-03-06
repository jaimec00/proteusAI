
import torch

# setup params for testing
torch.manual_seed(1)
batch, N, num_classes = 8, 128, 21

mean_mask_pct = 0.5
std_mask_pct = 0.25
min_mask_pct = 0.15
max_mask_pct = 1.00
mask_pct = torch.clamp((torch.randn((batch, ))*std_mask_pct) + mean_mask_pct, min=min_mask_pct, max=max_mask_pct)
mean_span = 10
std_span = 5
mask = torch.rand([batch, N]) > 0.8 # no masking in this test

# utils
seq_idx = torch.arange(N).unsqueeze(0)
valid = (~mask).sum(dim=1)
sample_done = lambda span_mask, valid, mask_pct: (((span_mask.sum(dim=1)+mean_span) / (valid+1e-6)) >= mask_pct) & (span_mask.any(dim=1) | (valid==0))
done = lambda span_mask, valid, mask_pct: sample_done(span_mask, valid, mask_pct).all()

# sample a span length for each residue from gaussian dist with mean span length and std span length defined above
span_lengths = torch.round(torch.clamp((torch.randn((batch, N))*std_span) + mean_span, min=1)).to(torch.int) # Z x N

# compute the number of spans to select per iteration for each sample, to avoid small updates on large sequence lengths
# valid samples / mean_span lengths is approx the number of spans that fit (assuming perfect spacing). 
# multiply by mask pct to get number of spans to reach mask_pct
num_spans_per_iter = torch.clamp(torch.ceil(mask_pct * (valid / mean_span)), min=1, max=N).long().unsqueeze(1) # Z x 1

# initialize the span mask
span_mask = torch.zeros_like(mask) # Z x N

# loop until each sample reaches its mask_pct
while not done(span_mask, valid, mask_pct):

    # get rand vals
    rand_vals = torch.rand(batch, N) # Z x N

    # get the 1D distance from the nearest span token for each token. use this to increase likelihood of choosing isolated region for next span
    span_batch_idxs, span_N_idxs = torch.nonzero(span_mask, as_tuple=True) # (numMask, numMask)
    dists_raw = (span_N_idxs.unsqueeze(1) - seq_idx).abs() # numMask x N
    dists = torch.full((batch, N), N, dtype=torch.long) # max dist is N, set it to this so amin works properly in next line
    dists.scatter_reduce_(0, span_batch_idxs.unsqueeze(1).expand(-1,N), dists_raw, reduce="amin") # amin gets the minimum distance of each token from all other mask tokens

    # multiply rand vals by dists so isolated regions are more likely to be in topk
    rand_vals *= dists
    rand_vals.masked_fill_(span_mask | mask, -float("inf"))

    # sort rand vals and find the kth largest (diff for each sample), use that as threshold to get start_idxs
    rand_vals_sorted, _ = torch.sort(rand_vals, dim=1, descending=True) # Z x N
    rand_val_thresh = torch.gather(rand_vals_sorted, 1, num_spans_per_iter-1)  # Z x 1
    rand_val_thresh.masked_fill_(sample_done(span_mask, valid, mask_pct).unsqueeze(1), float("inf")) # so don't select anything for samples that are done

    # use the thresh to select start idxs
    batch_idx, start_idx = torch.nonzero(rand_vals > rand_val_thresh, as_tuple=True) # numTrue

    # define the end idx by looking up the span length of the start idx
    end_idx = start_idx + span_lengths[batch_idx, start_idx] # numTrue

    # find tokens in the span
    in_span = (seq_idx >= start_idx.unsqueeze(1)) & (seq_idx < end_idx.unsqueeze(1)) & (~mask[batch_idx]) # numTrue x N

    # aggregate the span tokens for each batch, from numTrue x N --> Z x N, amax functions as an OR operation, since max is 1 for bool tensor
    span_mask.scatter_reduce_(0, batch_idx.unsqueeze(1).expand(-1, N), in_span, reduce="amax")

    # update the number of spans needed in next iter
    num_spans_per_iter = torch.clamp(torch.ceil(mask_pct * ((valid-span_mask.sum(dim=1)) / mean_span)), min=1, max=N).long().unsqueeze(1) # Z x 1

actual_pct = span_mask.sum(dim=1) / valid
print(mask_pct)
print(actual_pct)
print(((mask_pct-actual_pct).abs()/actual_pct).mean())
# print(span_mask)