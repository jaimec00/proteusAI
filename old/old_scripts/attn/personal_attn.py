# ------------------------------------------------------------------------------
'''
title: 			personal_attn.py
author: 		jaime cardenas
descripition:   a new idea i had for attention, which i.m.o. truly extends graph 
				neural networks. have a Q, K, and V matrix, as usual in self 
				attention, which are derived by doing matmults with x and wq, 
				wk, and wv. However, you also pass the coordinates, and you also
				have a Wg (g stands for geometric) weight matrix. for each token 
				in q, you create a personalized g matrix, essentially using the 
				edge features that proteinMPNN uses in their model, but extending 
				it a bit. the features of the g matrix will be 22 radial basis 
				functions (rbfs) with evenly spaced spreads, for each atom pair 
				in the both tokens, for my ca only model, there is Ca_q-Ca_k, 
				Ca_q-Cb_k, Cb_q-Ca_k, Cb_q-Cb_k, so 4*22 = 88 features, along 
				with a dot product for Cb_q(relative to Ca_q)-Ca_k(relative to 
				Ca_q) and Cb_q(relative to Ca_q)-Cb_k(relative to Ca_k) so 88+2=90
				then do a matul of g and Wg to get G, which is the learned 
				PERSONALIZED bias for each token pair, for each head. attention 
				is then 
				
				softmax([Q(K+G)^T]/sqrt(dk))V

				i know geometric attention is a better name, but will wait to see 
				if it works first before claiming the name for this method

				flash attention implementation, with on the fly computation of the 
				personalized key biases
'''
# ------------------------------------------------------------------------------

import torch
import triton
import triton.language as tl
import os

# ------------------------------------------------------------------------------


# define configurations for autotuning
configs = [	triton.Config({"BLOCK_I": i, "BLOCK_J": j}, num_warps=w)
			for i in [16]
			for j in [16]
			for w in [1, 2, 4, 8, 16]
		]

# filter out configs that are too big
def keep_fwd(conf):
	autotune = os.environ.get("ATTN_AUTOTUNE")
	BLOCK_I = conf.kwargs["BLOCK_I"]
	BLOCK_J = conf.kwargs["BLOCK_J"]
	if autotune == "1":
		return (BLOCK_I * BLOCK_J) <= 2048
	else:
		return ((BLOCK_I == 16) and (BLOCK_J == 16) and (conf.num_warps==8))

def keep_bwd(conf):
	autotune = os.environ.get("ATTN_AUTOTUNE")
	BLOCK_I = conf.kwargs["BLOCK_I"]
	BLOCK_J = conf.kwargs["BLOCK_J"]
	if autotune == "1":
		return (BLOCK_I * BLOCK_J) <= 2048
	else:
		return ((BLOCK_I == 16) and (BLOCK_J == 16) and (conf.num_warps==4))

@triton.autotune(list(filter(keep_fwd, configs)),
				 key=['tot_N', 'tot_Z', 'tot_H', 'tot_Dk'], # triton will not rerun autotune if these inputs are the same (size of input tensor)
				 restore_value=["O_ptr", "L_ptr"]) # make sure autotuning resets the outputs of this function for each configuration
@triton.jit
def _attn_fwd(  O_ptr, stride_O_Z, stride_O_H, stride_O_N, stride_O_Dk, 
				L_ptr, stride_L_Z, stride_L_H, stride_L_N, 
				Q_ptr, stride_Q_Z, stride_Q_H, stride_Q_N, stride_Q_Dk, 
				K_ptr, stride_K_Z, stride_K_H, stride_K_N, stride_K_Dk, 
				V_ptr, stride_V_Z, stride_V_H, stride_V_N, stride_V_Dk, 
				Wg_ptr, stride_Wg_H, stride_Wg_Dg, stride_Wg_Dk, 
				Ca_ptr, stride_Ca_Z, stride_Ca_N, stride_Ca_S, 
				Cb_ptr, stride_Cb_Z, stride_Cb_N, stride_Cb_S, 
				mask_ptr, stride_mask_Z, stride_mask_N, 
				spreads_ptr, stride_spreads_S,
				tot_Z: tl.constexpr, tot_H: tl.constexpr, tot_N: tl.constexpr, tot_Dk: tl.constexpr, tot_Dg: tl.constexpr, pad_Dg: tl.constexpr, sm_scale: tl.constexpr, 
				BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr
			):
	
	# --------------------------------------------------------------------------
	# compute offsets

	start_I = tl.program_id(0)
	offs_I = start_I*BLOCK_I

	offs_ZH = tl.program_id(1)
	offs_Z = offs_ZH // tot_H
	offs_H = offs_ZH % tot_H

	# --------------------------------------------------------------------------
	# setup block pointers

	# --------------------------------------------------------------------------
	# create Q, K, V, and Wg block pointers

	Qi_block_ptr = tl.make_block_ptr( # Ni x Dk
		base=Q_ptr + (offs_Z*stride_Q_Z) + (offs_H*stride_Q_H),
		shape=(tot_N, tot_Dk),
		strides=(stride_Q_N, stride_Q_Dk),
		offsets=(offs_I, 0),
		block_shape=(BLOCK_I, tot_Dk),
		order=(0, 1)
	)

	Kj_block_ptr = tl.make_block_ptr( # Nj x Dk
		base=K_ptr + (offs_Z*stride_K_Z) + (offs_H*stride_K_H),
		shape=(tot_N, tot_Dk),
		strides=(stride_K_N, stride_K_Dk),
		offsets=(0, 0),
		block_shape=(BLOCK_J, tot_Dk),
		order=(0, 1)
	)

	Vj_block_ptr = tl.make_block_ptr( # Nj x Dk
		base=V_ptr + (offs_Z*stride_V_Z) + (offs_H*stride_V_H),
		shape=(tot_N, tot_Dk),
		strides=(stride_V_N, stride_V_Dk),
		offsets=(0, 0),
		block_shape=(BLOCK_J, tot_Dk),
		order=(0, 1)
	)

	Wg_block_ptr = tl.make_block_ptr( # Dg x Dk
		base=Wg_ptr + (offs_H*stride_Wg_H),
		shape=(tot_Dg, tot_Dk),
		strides=(stride_Wg_Dg, stride_Wg_Dk),
		offsets=(0, 0),
		block_shape=(pad_Dg, tot_Dk), # use next power of 2, as is a requirement for triton tensors
		order=(0, 1)
	)

	# --------------------------------------------------------------------------
	# init ptr for Ca and Cb coordinates for I and J

	Cai_block_ptr = tl.make_block_ptr( # Ni x 4
		base=Ca_ptr + (offs_Z*stride_Ca_Z),
		shape=(tot_N, 3),
		strides=(stride_Ca_N, stride_Ca_S),
		offsets=(offs_I, 0),
		block_shape=(BLOCK_I, 4), # tensor need to be power of 2, 4th value is masked (x,y,z,mask)
		order=(0, 1)
	)

	Cbi_block_ptr = tl.make_block_ptr(
		base=Cb_ptr + (offs_Z*stride_Cb_Z),
		shape=(tot_N, 3),
		strides=(stride_Cb_N, stride_Cb_S),
		offsets=(offs_I, 0),
		block_shape=(BLOCK_I, 4), 
		order=(0, 1)
	)

	Caj_block_ptr = tl.make_block_ptr( # Nj x 4
		base=Ca_ptr + (offs_Z*stride_Ca_Z),
		shape=(tot_N, 3),
		strides=(stride_Ca_N, stride_Ca_S),
		offsets=(0, 0),
		block_shape=(BLOCK_J, 4), 
		order=(0, 1)
	)

	Cbj_block_ptr = tl.make_block_ptr(
		base=Cb_ptr + (offs_Z*stride_Cb_Z),
		shape=(tot_N, 3),
		strides=(stride_Cb_N, stride_Cb_S),
		offsets=(0, 0),
		block_shape=(BLOCK_J, 4), 
		order=(0, 1)
	)

	# --------------------------------------------------------------------------
	# create K/V column mask pointer 

	maski_block_ptr = tl.make_block_ptr( # Nj,
		base=mask_ptr + (offs_Z*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(offs_I, ),
		block_shape=(BLOCK_I, ),
		order=(0, )
	)

	maskj_block_ptr = tl.make_block_ptr( # Nj,
		base=mask_ptr + (offs_Z*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(0, ),
		block_shape=(BLOCK_J, ),
		order=(0, )
	)

	# also create spreads block pointer, needs padding so using block pointer
	num_spreads = (tot_Dg-2)//4 # convenience
	spreads_block_ptr = tl.make_block_ptr( # pad_Dg,
		base=spreads_ptr,
		shape=(num_spreads, ),
		strides=(stride_spreads_S, ),
		offsets=(0, ),
		block_shape=(pad_Dg//4, ), # this only works if num_spreads is a power of 2 - 1, which i will make a requirement
		order=(0, )
	)

	# --------------------------------------------------------------------------
	# load Qi, Cai, Cbi, and Wg and initialize output and statistics blocks

	Qi = tl.expand_dims(tl.load(Qi_block_ptr, boundary_check=(0,1), padding_option="zero"), axis=1) # Ni x 1 x Dk (fp16)
	Cai = tl.load(Cai_block_ptr, boundary_check=(0,1), padding_option="zero") # Ni x 4
	Cbi = tl.load(Cbi_block_ptr, boundary_check=(0,1), padding_option="zero") # Ni x 4
	Wg = tl.expand_dims(tl.load(Wg_block_ptr, boundary_check=(0,1), padding_option="zero"), axis=0) # 1 x Dg x Dk
	maski = tl.load(maski_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.int1)

	# initialize output and statistics block, all in fp32
	Oi = tl.zeros((BLOCK_I, tot_Dk), dtype=tl.float32) # Ni x Dk
	li = tl.zeros((BLOCK_I, ), dtype=tl.float32) # Ni,
	inf = float("inf") # convenience
	mi = (tl.zeros_like(li) - inf) # Ni, 

	# initialize spreads, padded to pad_Dg for ease of writing, and since must be a power of 2
	spreads = tl.load(spreads_block_ptr, boundary_check=(0,), padding_option="zero") # S+1,
	masked_spreads = (tl.arange(0,pad_Dg//4)==num_spreads)[None, None, :] # 1 x 1 x S+1
	spread_term = tl.where(masked_spreads, 0, 1/(2*spreads*spreads)) # 1 x 1 x S+1

	# --------------------------------------------------------------------------
	# loop through columns of K and V

	for j in tl.range(0, triton.cdiv(tot_N, BLOCK_J), 1, loop_unroll_factor=1): # no loop unrolling

		# ----------------------------------------------------------------------
		# first get the distances

		# load the coords for J
		Caj = tl.load(Caj_block_ptr, boundary_check=(0,1), padding_option="zero") # Nj x 4
		Cbj = tl.load(Cbj_block_ptr, boundary_check=(0,1), padding_option="zero") # Nj x 4

		# get distances (i is first atom, j is second atom in the name)
		# note that Cb coords are passed as unit vectors with orientations relative to corresponding Ca
		CaCa_raw = Cai[:, None, :] - Caj[None, :, :] # Ni x Nj x 4
		CaCb_raw = Cai[:, None, :] - (Caj[None, :, :] + Cbj[None, :, :]*1.53) # 1.53 is avg Ca-Cb dist, 
		CbCa_raw = (Cai[:, None, :] + Cbi[:, None, :]*1.53) - Caj[None, :, :] 
		CbCb_raw = (Cai[:, None, :] + Cbi[:, None, :]*1.53) - (Caj[None, :, :] + Cbj[None, :, :]*1.53) 

		CaCa = tl.sqrt(tl.sum(CaCa_raw*CaCa_raw, axis=2)) # Ni x Nj
		CaCb = tl.sqrt(tl.sum(CaCb_raw*CaCb_raw, axis=2)) 
		CbCa = tl.sqrt(tl.sum(CbCa_raw*CbCa_raw, axis=2)) 
		CbCb = tl.sqrt(tl.sum(CbCb_raw*CbCb_raw, axis=2)) 

		# ----------------------------------------------------------------------
		# now compute rbfs and dots

		# compute the rbfs, linearly mapped to [-1,1]
		CaCa_rbfs = (2*tl.exp(-CaCa[:, :, None]*CaCa[:, :, None] * spread_term)) - 1 - masked_spreads # Ni x Nj x S+1, last idxs is 2*exp(0)-1 - ismasked = 2*1 - 1 - 1 = 0 
		CaCb_rbfs = (2*tl.exp(-CaCb[:, :, None]*CaCb[:, :, None] * spread_term)) - 1 - masked_spreads
		CbCa_rbfs = (2*tl.exp(-CbCa[:, :, None]*CbCa[:, :, None] * spread_term)) - 1 - masked_spreads
		CbCb_rbfs = (2*tl.exp(-CbCb[:, :, None]*CbCb[:, :, None] * spread_term)) - 1 - masked_spreads

		# also include two dot product features, one for CbCa and another for CbCb
		# Cb are already unit vectors, so need to make the Caj a unit vector too 
		CaCa_rev = -CaCa_raw / tl.where(CaCa==0, 1, CaCa)[:, :, None] # negative so that this is the vector pointing from Cai to Caj, divide by CaCa dist to get unit vector
		CbCa_dot = tl.sum(Cbi[:, None, :] * CaCa_rev, axis=2, keep_dims=True) # Ni x Nj x 1
		CbCb_dot = tl.sum(Cbi[:, None, :] * Cbj[None, :, :], axis=2, keep_dims=True)

		# now create g, put CbCa_dot at the last idx of CaCa_rbfs, and CbCb_dot at the last idx of CbCa_rbfs
		# then leave 0 at last idx of CaCb_rbfs and CbCb_rbfs (already done in rbf computation masking)
		# will then interleave(CaCa, CaCb), which means the zero i added at the last idx of CaCb will be at the end of the joined tensor
		# same logic goes for interleave(CbCa, CbCb)
		# then interleave the above two interleaved tensors, and the last two values will be zeros, so they wont contribute, and get g of shape Ni x Nj x pad_Dg
		CaCa_rbfs += CbCa_dot * masked_spreads # Ni x Nj x 1 * 1 x 1 x S+1 --> Ni x Nj x S+1
		CbCa_rbfs += CbCb_dot * masked_spreads 

		g = tl.interleave(tl.interleave(CaCa_rbfs, CaCb_rbfs), tl.interleave(CbCa_rbfs, CbCb_rbfs)) # Ni x Nj x Dg

		# ----------------------------------------------------------------------
		# compute the personalized key matrix

		# compute G
		G = tl.dot(g, tl.broadcast_to(Wg, (BLOCK_I, pad_Dg, tot_Dk))) # Ni x Nj x Dg @ 1 x Dg x Dk --> Ni x Nj x Dk

		# now need to add the bias to the key matrix, first load it
		Kj = tl.expand_dims(tl.load(Kj_block_ptr, boundary_check=(0,1), padding_option="zero"), axis=0) # 1 x Nj x Dk

		# add the bias
		KjG = (Kj + G).to(tl.float16) # Ni x Dk x Nj

		# ----------------------------------------------------------------------
		# compute the attention matrix with a personalized key matrix for each query token and mask invalid pairs

		# compute attention
		# cant do batched mma because Q is Ni x 1 x Dk, ie K is < 16, so would hav to pad which would blow up register usage, doing manual dot product 
		Sij = tl.sum(Qi * KjG, axis=2) * sm_scale

		# set masked positions to -inf
		maskj = tl.load(maskj_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.int1) # 1 x Nj
		attn_mask = maski[:, None] & maskj[None, :]
		Sij = tl.where(attn_mask, Sij, -inf) # Ni x Nj

		# ----------------------------------------------------------------------
		# update the statistics

		# max of each row
		mij = tl.maximum(mi, tl.max(Sij, axis=1)) # Ni,  (fp32)

		# compute softmax(Sij - mij) = Pij
		Pij = tl.exp(tl.where(mij[:, None]==-inf, -inf, Sij - mij[:, None])) # Ni x Nj (fp32)

		# compute alpha
		alpha = tl.exp(tl.where((mi==-inf) | (mij==-inf), tl.where((mi==-inf) & (mij==-inf), 0, -inf), mi - mij)) # Ni,(fp32)

		# update li
		li = alpha*li + tl.sum(Pij, axis=1) # Ni, (fp32)

		# load Vj compute output. convert Pij to fp16, Vj is already in fp16. output is in fp32
		Oi = tl.dot(Pij.to(tl.float16), tl.load(Vj_block_ptr, boundary_check=(0,1), padding_option="zero"), acc=(Oi*alpha[:, None])) # Ni x Dk

		# update statistics for next iteration
		mi = mij

		# ----------------------------------------------------------------------
		# advance block pointers for columns

		Kj_block_ptr = tl.advance(Kj_block_ptr, (BLOCK_J, 0))
		Vj_block_ptr = tl.advance(Vj_block_ptr, (BLOCK_J, 0))
		Caj_block_ptr = tl.advance(Caj_block_ptr, (BLOCK_J, 0))
		Cbj_block_ptr = tl.advance(Cbj_block_ptr, (BLOCK_J, 0))
		maskj_block_ptr = tl.advance(maskj_block_ptr, (BLOCK_J, ))

	# --------------------------------------------------------------------------
	# epilogue

	# normalize output. li==0 means that all columns in that row are masked out
	Oi = tl.where(li[:, None]!=0, Oi / li[:, None], 0.0)

	# compute log sum exponential (li==0 --> mi + log(li) = -inf)
	mi += tl.log(li)

	# --------------------------------------------------------------------------
	# create output and log sum exp block pointers

	Oi_block_ptr = tl.make_block_ptr( # N x Dk
		base=O_ptr + (offs_Z*stride_O_Z) + (offs_H*stride_O_H),
		shape=(tot_N, tot_Dk),
		strides=(stride_O_N, stride_O_Dk),
		offsets=(offs_I, 0),
		block_shape=(BLOCK_I, tot_Dk),
		order=(0, 1)
	)

	Li_block_ptr = tl.make_block_ptr( # N,
		base=L_ptr + (offs_Z*stride_L_Z) + (offs_H*stride_L_H),
		shape=(tot_N, ),
		strides=(stride_L_N, ),
		offsets=(offs_I, ),
		block_shape=(BLOCK_I, ),
		order=(0, )
	)

	# --------------------------------------------------------------------------
	# store output and logsum exp

	tl.store(Oi_block_ptr, Oi, boundary_check=(0,1))
	tl.store(Li_block_ptr, mi, boundary_check=(0,))

@triton.autotune(list(filter(keep_fwd, configs)),
				 key=['tot_N', 'tot_Z', 'tot_H', 'tot_Dg', 'tot_Dk'], # triton will not rerun autotune if these inputs are the same (size of input tensor)
				 restore_value=["dQ_ptr", "dK_ptr", "dV_ptr", "dWg_ptr"]) # make sure autotuning resets the outputs of this function for each configuration
@triton.jit
def _attn_bwd(	D_ptr, stride_D_Z, stride_D_H, stride_D_N, 
				L_ptr, stride_L_Z, stride_L_H, stride_L_N, 

				Q_ptr, stride_Q_Z, stride_Q_H, stride_Q_N, stride_Q_Dk, 
				K_ptr, stride_K_Z, stride_K_H, stride_K_N, stride_K_Dk, 
				V_ptr, stride_V_Z, stride_V_H, stride_V_N, stride_V_Dk, 
				Wg_ptr, stride_Wg_H, stride_Wg_Dg, stride_Wg_Dk, 
				Ca_ptr, stride_Ca_Z, stride_Ca_N, stride_Ca_S, 
				Cb_ptr, stride_Cb_Z, stride_Cb_N, stride_Cb_S, 

				dO_ptr, stride_dO_Z, stride_dO_H, stride_dO_N, stride_dO_Dk, 
				dQ_ptr, stride_dQ_Z, stride_dQ_H, stride_dQ_N, stride_dQ_Dk, 
				dK_ptr, stride_dK_Z, stride_dK_H, stride_dK_N, stride_dK_Dk, 
				dV_ptr, stride_dV_Z, stride_dV_H, stride_dV_N, stride_dV_Dk, 
				dWg_ptr, stride_dWg_H, stride_dWg_Dg, stride_dWg_Dk, 
				
				mask_ptr, stride_mask_Z, stride_mask_N, 
				spreads_ptr, stride_spreads_S,

				tot_Z: tl.constexpr, tot_H: tl.constexpr, tot_N: tl.constexpr, tot_Dk: tl.constexpr, tot_Dg: tl.constexpr, pad_Dg: tl.constexpr, sm_scale: tl.constexpr, 
				BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr
			):

	# --------------------------------------------------------------------------
	# compute offsets

	start_J = tl.program_id(0)
	offs_J = start_J*BLOCK_J

	offs_ZH = tl.program_id(1)
	offs_Z = offs_ZH // tot_H
	offs_H = offs_ZH % tot_H

	# --------------------------------------------------------------------------
	# setup block pointers 

	# --------------------------------------------------------------------------
	# create block pointers for Di and Li

	Di_block_ptr = tl.make_block_ptr( # Ni x Dk
		base=D_ptr + (offs_Z*stride_D_Z) + (offs_H*stride_D_H),
		shape=(tot_N, ),
		strides=(stride_D_N, ),
		offsets=(0, ),
		block_shape=(BLOCK_I, ),
		order=(0, )
	)

	Li_block_ptr = tl.make_block_ptr( # Ni x Dk
		base=L_ptr + (offs_Z*stride_L_Z) + (offs_H*stride_L_H),
		shape=(tot_N, ),
		strides=(stride_L_N, ),
		offsets=(0, ),
		block_shape=(BLOCK_I, ),
		order=(0, )
	)

	# --------------------------------------------------------------------------
	# create Q, K, V, and Wg block pointers

	Qi_block_ptr = tl.make_block_ptr( # Ni x Dk
		base=Q_ptr + (offs_Z*stride_Q_Z) + (offs_H*stride_Q_H),
		shape=(tot_N, tot_Dk),
		strides=(stride_Q_N, stride_Q_Dk),
		offsets=(0, 0),
		block_shape=(BLOCK_I, tot_Dk),
		order=(0, 1)
	)

	Kj_block_ptr = tl.make_block_ptr( # Nj x Dk
		base=K_ptr + (offs_Z*stride_K_Z) + (offs_H*stride_K_H),
		shape=(tot_N, tot_Dk),
		strides=(stride_K_N, stride_K_Dk),
		offsets=(offs_J, 0),
		block_shape=(BLOCK_J, tot_Dk),
		order=(0, 1)
	)

	Vj_block_ptr = tl.make_block_ptr( # Nj x Dk
		base=V_ptr + (offs_Z*stride_V_Z) + (offs_H*stride_V_H),
		shape=(tot_N, tot_Dk),
		strides=(stride_V_N, stride_V_Dk),
		offsets=(offs_J, 0),
		block_shape=(BLOCK_J, tot_Dk),
		order=(0, 1)
	)

	Wg_block_ptr = tl.make_block_ptr( # Dg x Dk
		base=Wg_ptr + (offs_H*stride_Wg_H),
		shape=(tot_Dg, tot_Dk),
		strides=(stride_Wg_Dg, stride_Wg_Dk),
		offsets=(0, 0),
		block_shape=(pad_Dg, tot_Dk), # use next power of 2, as is a requirement for triton tensors
		order=(0, 1)
	)

	# --------------------------------------------------------------------------
	# init ptr for Ca and Cb coordinates for I and J

	Cai_block_ptr = tl.make_block_ptr( # Ni x 4
		base=Ca_ptr + (offs_Z*stride_Ca_Z),
		shape=(tot_N, 3),
		strides=(stride_Ca_N, stride_Ca_S),
		offsets=(0, 0),
		block_shape=(BLOCK_I, 4), # tensor need to be power of 2, 4th value is masked (x,y,z,mask)
		order=(0, 1)
	)

	Cbi_block_ptr = tl.make_block_ptr(
		base=Cb_ptr + (offs_Z*stride_Cb_Z),
		shape=(tot_N, 3),
		strides=(stride_Cb_N, stride_Cb_S),
		offsets=(0, 0),
		block_shape=(BLOCK_I, 4), 
		order=(0, 1)
	)

	Caj_block_ptr = tl.make_block_ptr( # Nj x 4
		base=Ca_ptr + (offs_Z*stride_Ca_Z),
		shape=(tot_N, 3),
		strides=(stride_Ca_N, stride_Ca_S),
		offsets=(offs_J, 0),
		block_shape=(BLOCK_J, 4), 
		order=(0, 1)
	)

	Cbj_block_ptr = tl.make_block_ptr(
		base=Cb_ptr + (offs_Z*stride_Cb_Z),
		shape=(tot_N, 3),
		strides=(stride_Cb_N, stride_Cb_S),
		offsets=(offs_J, 0),
		block_shape=(BLOCK_J, 4), 
		order=(0, 1)
	)

 	# --------------------------------------------------------------------------
	# create dOi and dQi block pointers, dK, dV, and dWg instantiated after the loop
	
	dOi_block_ptr = tl.make_block_ptr( # N x d_k
		base=dO_ptr + (offs_Z*stride_dO_Z) + (offs_H*stride_dO_H),
		shape=(tot_N, tot_Dk),
		strides=(stride_dO_N, stride_dO_Dk),
		offsets=(0, 0),
		block_shape=(BLOCK_I, tot_Dk),
		order=(0, 1)
	)

	# perform atomic adds on dQi, which don't support block pointers, so do manual indexing
	dQi_start_ptr = dQ_ptr + (offs_Z*stride_dQ_Z) + (offs_H*stride_dQ_H)
	dQi_block_ptr = dQi_start_ptr + (tl.arange(0,BLOCK_I)[:, None]*stride_dQ_N) + (tl.arange(0,tot_Dk)[None, :]*stride_dQ_Dk)

	# --------------------------------------------------------------------------
	# create K/V column mask pointer 

	maski_block_ptr = tl.make_block_ptr( # Nj,
		base=mask_ptr + (offs_Z*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(0, ),
		block_shape=(BLOCK_I, ),
		order=(0, )
	)

	maskj_block_ptr = tl.make_block_ptr( # Nj,
		base=mask_ptr + (offs_Z*stride_mask_Z),
		shape=(tot_N, ),
		strides=(stride_mask_N, ),
		offsets=(offs_J, ),
		block_shape=(BLOCK_J, ),
		order=(0, )
	)

	# also create spreads block pointer, needs padding so using block pointer
	num_spreads = (tot_Dg-2)//4 # convenience
	spreads_block_ptr = tl.make_block_ptr( # pad_Dg,
		base=spreads_ptr,
		shape=(num_spreads, ),
		strides=(stride_spreads_S, ),
		offsets=(0, ),
		block_shape=(pad_Dg//4, ), # this only works if num_spreads is a power of 2 - 1, which i will make a requirement
		order=(0, )
	)

	# --------------------------------------------------------------------------
	# load Kj, Vj, Caj, Cbj, and Wg 

	Kj = tl.expand_dims(tl.load(Kj_block_ptr, boundary_check=(0, 1), padding_option="zero"), axis=0) # 1 x Nj x Dk
	Vj = tl.load(Vj_block_ptr, boundary_check=(0, 1), padding_option="zero")
	Caj = tl.load(Caj_block_ptr, boundary_check=(0,1), padding_option="zero") # Ni x 4
	Cbj = tl.load(Cbj_block_ptr, boundary_check=(0,1), padding_option="zero") # Ni x 4
	Wg = tl.expand_dims(tl.load(Wg_block_ptr, boundary_check=(0,1), padding_option="zero"), axis=0) # 1 x Dg x Dk

	# load mask
	maskj = tl.load(maskj_block_ptr, boundary_check=(0, ), padding_option="zero").to(tl.int1)

	# initialize dKj and dVj and d_spread(scalar), in fp32
	dKj = tl.zeros((BLOCK_J, tot_Dk), dtype=tl.float32)
	dVj = tl.zeros((BLOCK_J, tot_Dk), dtype=tl.float32)
	dWg = tl.zeros((pad_Dg, tot_Dk), dtype=tl.float32)
	inf = float("inf") # convenience

	# initialize spreads, padded to pad_Dg for ease of writing, and since must be a power of 2
	spreads = tl.load(spreads_block_ptr, boundary_check=(0,), padding_option="zero") # S+1,
	masked_spreads = (tl.arange(0,pad_Dg//4)==num_spreads)[None, None, :] # 1 x 1 x S+1
	spread_term = tl.where(masked_spreads, 0, 1/(2*spreads*spreads)) # 1 x 1 x S+1

	# --------------------------------------------------------------------------
	# loop through columns of K and V

	for i in tl.range(0, triton.cdiv(tot_N, BLOCK_I), 1, loop_unroll_factor=1): # no loop unrolling

	    # ----------------------------------------------------------------------
	    # first get the distances

		# load the coords for J
		Cai = tl.load(Cai_block_ptr, boundary_check=(0,1), padding_option="zero") # Nj x 4
		Cbi = tl.load(Cbi_block_ptr, boundary_check=(0,1), padding_option="zero") # Nj x 4

		# raw dists
		CaCa_raw = Cai[:, None, :] - Caj[None, :, :] # Ni x Nj x 4
		CaCb_raw = Cai[:, None, :] - (Caj[None, :, :] + Cbj[None, :, :]*1.53) # 1.53 is avg Ca-Cb dist, 
		CbCa_raw = (Cai[:, None, :] + Cbi[:, None, :]*1.53) - Caj[None, :, :] 
		CbCb_raw = (Cai[:, None, :] + Cbi[:, None, :]*1.53) - (Caj[None, :, :] + Cbj[None, :, :]*1.53) 

		# dists
		CaCa = tl.sqrt(tl.sum(CaCa_raw*CaCa_raw, axis=2)) # Ni x Nj
		CaCb = tl.sqrt(tl.sum(CaCb_raw*CaCb_raw, axis=2)) 
		CbCa = tl.sqrt(tl.sum(CbCa_raw*CbCa_raw, axis=2)) 
		CbCb = tl.sqrt(tl.sum(CbCb_raw*CbCb_raw, axis=2)) 

	    # ----------------------------------------------------------------------
		# now compute rbfs and dots

		# compute the rbfs, linearly mapped to [-1,1]
		CaCa_rbfs = (2*tl.exp(-CaCa[:, :, None]*CaCa[:, :, None] * spread_term)) - 1 - masked_spreads # Ni x Nj x S+1, last idxs is 2*exp(0)-1 - ismasked = 2*1 - 1 - 1 = 0 
		CaCb_rbfs = (2*tl.exp(-CaCb[:, :, None]*CaCb[:, :, None] * spread_term)) - 1 - masked_spreads
		CbCa_rbfs = (2*tl.exp(-CbCa[:, :, None]*CbCa[:, :, None] * spread_term)) - 1 - masked_spreads
		CbCb_rbfs = (2*tl.exp(-CbCb[:, :, None]*CbCb[:, :, None] * spread_term)) - 1 - masked_spreads

		# also include two dot product features, one for CbCa and another for CbCb
		CaCa_rev = -CaCa_raw / tl.where(CaCa==0, 1, CaCa)[:, :, None] # negative so that this is the vector pointing from Cai to Caj, divide by CaCa dist to get unit vector
		CbCa_dot = tl.sum(Cbi[:, None, :] * CaCa_rev, axis=2, keep_dims=True) # Ni x Nj x 1
		CbCb_dot = tl.sum(Cbi[:, None, :] * Cbj[None, :, :], axis=2, keep_dims=True)

		# join the dots and the corresponding rbfs at empty index
		CaCa_rbfs += CbCa_dot * masked_spreads # Ni x Nj x 1 * 1 x 1 x S+1 --> Ni x Nj x S+1
		CbCa_rbfs += CbCb_dot * masked_spreads 

		# interleave to get g
		g = tl.interleave(tl.interleave(CaCa_rbfs, CaCb_rbfs), tl.interleave(CbCa_rbfs, CbCb_rbfs)) # Ni x Nj x Dg

		# ----------------------------------------------------------------------
		# compute the personalized key matrix

		# compute G
		G = tl.dot(g, tl.broadcast_to(Wg, (BLOCK_I, pad_Dg, tot_Dk))) # Ni x Nj x Dg @ 1 x Dg x Dk --> Ni x Nj x Dk

		# add the bias
		KjG = (Kj + G).to(tl.float16) # Ni x Nj x Dk

		# ----------------------------------------------------------------------
		# compute the attention matrix with a personalized key matrix for each query token and mask invalid pairs

		# now load Qi
		Qi = tl.load(Qi_block_ptr, boundary_check=(0,1), padding_option="zero") # Ni x Dk

		# compute attention
		Sij = tl.sum(Qi[:, None, :] * KjG, axis=2) * sm_scale

		# set masked positions to -inf
		maski = tl.load(maski_block_ptr, boundary_check=(0, ), padding_option="zero").to(tl.int1)
		attn_mask = maski[:, None] & maskj[None, :]
		Sij = tl.where(attn_mask, Sij, -inf) # Ni x Nj

	    # ----------------------------------------------------------------------
		# load log sum exp statistics
		Li = tl.load(Li_block_ptr, boundary_check=(0, ), padding_option="zero") # (fp32)

		# exp(Sij - Lij) = exp(Sij - mi - log(li)) = exp(Sij - mi) / exp(log(li)) 
		# = exp(Sij - mi) / li
		# mi is max for the row pre-softmax (for safe softmax), li is the normalizing term (sum of exponentials for that row)
		Pij = tl.exp(tl.where(attn_mask, Sij - Li[:, None], -inf)) # N x N (fp32)

		# load gradient w.r.t output (fp16)
		dOi = tl.load(dOi_block_ptr, boundary_check=(0, 1), padding_option="zero") # Ni x d_k

		# compute gradient wrt Vj (dOi already in fp16, out is in fp32)
		dVj += tl.dot(tl.permute(Pij, (1,0)).to(tl.float16), dOi) # Nj x d_k

		# compute gradient wrt Pij (dOi and Vj already in fp16)
		dPij = tl.where(attn_mask, tl.dot(dOi, tl.permute(Vj, (1,0))), 0.0) # N x N

		# load Di = rowsum(O*dO) to compute gradient wrt SRij (grad of loss wrt Sij*rbf)
		dSij = Pij * (dPij -  tl.load(Di_block_ptr, boundary_check=(0, ), padding_option="zero")[:, None]) # N x N

		# compute gradient wrt Qij and perform atomic add to communicate between thread blocks 
		dQi = tl.sum(dSij[:, :, None] * KjG, axis=1) * sm_scale # Ni x Nj x 1 * Ni x Nj x Dk
		dQi_mask = ((i*BLOCK_I + tl.arange(0,BLOCK_I)[:, None]) < tot_N)
		tl.atomic_add(dQi_block_ptr, dQi, mask=dQi_mask)

		# note that S = Q(K+G) = QK + QG, so dK and dG computed differently, since QG is batched along Ni
		# compute dKj
		dKj += tl.dot(tl.permute(attn_mask*dSij, (1,0)).to(tl.float16), Qi) * sm_scale # Nj x Dk

		# dSij is originally Ni x 1 x Nj and Q was Ni x 1 x Dk
		# dG = S.t @ Q --> Ni x Nj x 1 @ Ni x 1 x Dk --> Ni x Nj x Dk
		dG = dSij[:, :, None] * Qi[:, None, :] * sm_scale 

		# and finally, dWg # sumi(g.T @ dG) --> sumi(Ni x Dg x Nj @ Ni x Nj x Dk)--> sumi(Ni x Dg x Dk) --> Dg x Dk 
		dWg += tl.sum(tl.dot(tl.trans(g, (0,2,1)), dG), axis=0)

		# advance the pointers
		Qi_block_ptr = tl.advance(Qi_block_ptr, (BLOCK_I, 0))
		dQi_block_ptr += BLOCK_I*stride_dQ_N
		dOi_block_ptr = tl.advance(dOi_block_ptr, (BLOCK_I, 0))
		Li_block_ptr = tl.advance(Li_block_ptr, (BLOCK_I, ))
		Di_block_ptr = tl.advance(Di_block_ptr, (BLOCK_I, ))
		Cai_block_ptr = tl.advance(Cai_block_ptr, (BLOCK_I, 0))
		Cbi_block_ptr = tl.advance(Cbi_block_ptr, (BLOCK_I, 0))
		maski_block_ptr = tl.advance(maski_block_ptr, (BLOCK_I, ))


	# initialize dK and dV pointers to write output
	dKj_block_ptr = tl.make_block_ptr( # N x d_k
		base=dK_ptr + (offs_Z*stride_dK_Z) + (offs_H*stride_dK_H),
		shape=(tot_N, tot_Dk),
		strides=(stride_dK_N, stride_dK_Dk),
		offsets=(offs_J, 0),
		block_shape=(BLOCK_J, tot_Dk),
		order=(0, 1)
	)

	dVj_block_ptr = tl.make_block_ptr( # N x d_k
		base=dV_ptr + (offs_Z*stride_dV_Z) + (offs_H*stride_dV_H),
		shape=(tot_N, tot_Dk),
		strides=(stride_dV_N, stride_dV_Dk),
		offsets=(offs_J, 0),
		block_shape=(BLOCK_J, tot_Dk),
		order=(0, 1)
	)

	# write dK and dV to HBM, dQ was written to HBM progressively in for loop via atomic adds
	tl.store(dKj_block_ptr, dKj.to(tl.float16), boundary_check=(0,1))
	tl.store(dVj_block_ptr, dVj.to(tl.float16), boundary_check=(0,1))

	# store the grad wrt to Wg for this head
	dWg_ptr = dWg_ptr + (offs_H*stride_dWg_H) + (tl.arange(0,pad_Dg)[:, None]*stride_dWg_Dg) + (tl.arange(0,tot_Dk)[None, :]*stride_dWg_Dk)
	tl.atomic_add(dWg_ptr, dWg, mask=tl.arange(0,pad_Dg)[:, None]<tot_Dg)

def personal_attn(Q, K, V, Wg, Ca, Cb, spreads, mask=None):
	return _personal_attention.apply(Q, K, V, Wg, Ca, Cb, spreads, mask)

class _personal_attention(torch.autograd.Function):

	@staticmethod
	def forward(ctx, Q, K, V, Wg, Ca, Cb, spreads, mask):
		'''
		Q (torch.Tensor, shape-->ZxHxNxDk):  Projected query matrix (q@Wq)
		K (torch.Tensor, shape-->ZxHxNxDk):  Projected key matrix (k@Wk)
		V (torch.Tensor, shape-->ZxHxNxDk):  Projected value matrix (v@Wv)
		Wg (torch.Tensor, shape-->HxDgxDk):  Weight matrix for the projection of personalized geometric features
		Ca (torch.Tensor, shape-->ZxNx3):  alpha-carbon coordinates
		Cb (torch.Tensor, shape-->ZxNx3):  beta-carbon coordinates, relative to the corresponding alpha-carbon
		spreads (torch.Tensor, shape-->HxS): spreads to use in the rbf computations for each atom pair for each token pair
		mask (torch.Tensor, shape-->ZxN):  mask for the K matrix
		'''
	
		# checks
		assert Q.shape == K.shape == V.shape, f"Q, K, and V projection shapes must match, but got {Q.shape=}, {K.shape=}, {V.shape=}"
		Z, H, N, Dk = Q.shape
		Dm = H*Dk
		sm_scale = 1/(Dk**0.5)
		assert Dm % 2 == 0, f"d_model must be divisible by 2, not {d_model=}"
		assert Dk>=16, f"Dk must be greater than 16 due to triton matmul requirements, not {Dk=}"
		assert (Ca.dim() == Cb.dim() == 3) and (Ca.size(2) == Cb.size(2) == 3), f"coordinates must be of shape (batch, N, 3), not {Ca.shape=}, {Cb.shape=}" 
		_, Dg, _ = Wg.shape
		num_spreads = spreads.size(0)
		assert Dg == (num_spreads*4 + 2), f"Dg must be equal to num_spreads*4+2: {num_spreads*4+2}, not {Dg}"
		assert (num_spreads+1) == triton.next_power_of_2(num_spreads), 	f"the number of spreads per atom pair (4 atom pairs per query token in this kernel)"\
																		f"must be (a power of 2) - 1 (e.g. 2**4-1=16-1=15), got {num_spreads=}. this is a "\
																		f"practical consideration motivated by the requirement that triton tensors dims must be "\
																		f"powers of two and to minimize excess register allocation"
		assert torch.all(spreads>0) , f"spreads must be a tensor of values > 0, got {spreads=}"

		# matmults done in fp16
		Q = Q.to(torch.float16).contiguous()
		K = K.to(torch.float16).contiguous()
		V = V.to(torch.float16).contiguous()

		# geometric computations done in fp32
		Wg = Wg.to(torch.float32).contiguous()
		Ca = Ca.to(torch.float32).contiguous()
		Cb = Cb.to(torch.float32).contiguous()
		spreads = spreads.to(torch.float32).contiguous()

		# initialize mask, output, and logsumexp tensors
		mask = (torch.ones(Z, N, dtype=torch.bool, device=Q.device) if mask is None else ~mask).contiguous() # batch x N		
		O = torch.zeros(Z, H, N, Dk, dtype=torch.float32, device=Q.device).contiguous() # batch x N x d_model
		L = torch.zeros(Z, H, N, dtype=torch.float32, device=Q.device).contiguous() # batch x nheads x N

		# define the grid
		grid = lambda args: (   triton.cdiv(args["tot_N"], args["BLOCK_I"]), 
								args["tot_Z"]*args["tot_H"],	
								1
							)

		# run the kernel
		_attn_fwd[grid](  	O, O.stride(0), O.stride(1), O.stride(2), O.stride(3),
							L, L.stride(0), L.stride(1), L.stride(2), 

							Q, Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), 
							K, K.stride(0), K.stride(1), K.stride(2), K.stride(3), 
							V, V.stride(0), V.stride(1), V.stride(2), V.stride(3),							
							Wg, Wg.stride(0), Wg.stride(1), Wg.stride(2),
							Ca, Ca.stride(0), Ca.stride(1), Ca.stride(2), 
							Cb, Cb.stride(0), Cb.stride(1), Cb.stride(2),
							
							mask, mask.stride(0), mask.stride(1), 
							spreads, spreads.stride(0),
							Z, H, N, Dk, Dg, triton.next_power_of_2(Dg), sm_scale
						)

		# for backwards pass
		ctx.save_for_backward(Q, K, V, Wg, spreads, O, L, Ca, Cb, mask)
		ctx.sm_scale = sm_scale

		return O

	@staticmethod
	def backward(ctx, dO):
		
		Q, K, V, Wg, spreads, O, L, Ca, Cb, mask = ctx.saved_tensors

		Z, H, N, Dk = Q.shape
		_, Dg, _ = Wg.shape

		# compute D for dS calculation
		D = torch.sum(O*dO, dim=3).to(torch.float16).contiguous() # Z x H x N x D -> Z x H x N

		# cast to float16 for matmults
		dO = dO.to(torch.float16).contiguous()

		dQ = torch.zeros_like(Q).contiguous() # batch x N x d_model
		dK = torch.zeros_like(K).contiguous() # batch x N x d_model
		dV = torch.zeros_like(V).contiguous() # batch x N x d_model
		dWg = torch.zeros_like(Wg).contiguous() # batch x N x d_model

		# define the grid
		grid = lambda args: (
			triton.cdiv(args["tot_N"], args["BLOCK_J"]), # parralel along J for bwd
			args["tot_Z"]*args["tot_H"],
			1
		)

		_attn_bwd[grid](	D, D.stride(0), D.stride(1), D.stride(2),
							L, L.stride(0), L.stride(1), L.stride(2),
							
							Q, Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), 
							K, K.stride(0), K.stride(1), K.stride(2), K.stride(3), 
							V, V.stride(0), V.stride(1), V.stride(2), V.stride(3),							
							Wg, Wg.stride(0), Wg.stride(1), Wg.stride(2),
							Ca, Ca.stride(0), Ca.stride(1), Ca.stride(2), 
							Cb, Cb.stride(0), Cb.stride(1), Cb.stride(2),
							
							dO, dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
							dQ, dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3), 
							dK, dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3), 
							dV, dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),							
							dWg, dWg.stride(0), dWg.stride(1), dWg.stride(2),
							
							mask, mask.stride(0), mask.stride(1), 
							spreads, spreads.stride(0),

							Z, H, N, Dk, Dg, triton.next_power_of_2(Dg), ctx.sm_scale
						)

		return dQ, dK, dV, dWg, None, None, None, None, None, None
