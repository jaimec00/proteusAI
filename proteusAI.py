# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		proteusAI.py
description:	predicts the amino acid sequence of a protein based on backbone coordinates. 
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn

from utils.model_utils.mpnn.mpnn import MPNN
from utils.model_utils.mpnn.featurize import FeaturizeProtein
from data.constants import canonical_aas, aa_2_lbl

# ----------------------------------------------------------------------------------------------------------------------

class proteusAI(nn.Module):
	def __init__(self,  K=30, d_model=128, # model dims
						min_wl=3.5, max_wl=12.0, base_wl=20.0, learn_wl=True, # node embedding
						min_rbf=2.0, max_rbf=22.0, num_rbfs=16, # edge embedding
						layers=3, dropout=0.00 # general
					):

		super(proteusAI, self).__init__()

		self.featurizer = FeaturizeProtein( K=K, d_model=128, # model dims
											min_wl=min_wl, max_wl=max_wl, base_wl=base_wl, learn_wl=learn_wl, # node features (wf embedding)
											min_rbf=min_rbf, max_rbf=max_rbf, num_rbfs=num_rbfs # edge features
										)

		self.encoders = nn.ModuleList([MPNN(d_model=128, dropout=dropout, update_edge=i!=(layers-1)) for i in range(layers)])

		self.out_proj = nn.Linear(d_model, len(canonical_aas))

	def forward(self, C, L, chain_idxs, node_mask=None, inference=False, temp=1e-6, cycles=4):
		'''
		C is coords
		L is labels (-1 means to be decoded, unless nodemask is true at that position)
		'''

		if inference:
			V = self.inference(C, L, chain_idxs, node_mask=node_mask, temp=temp, cycles=cycles)
		else:

			# featurize protein
			V, E, K, edge_mask = self.featurizer(C, L, chain_idxs, node_mask=node_mask)

			# encode structure
			V = self.encode(V, E, K, edge_mask)

		return V

	def encode(self, V, E, K, edge_mask):

		for encoder in self.encoders:
			V, E = encoder(V, E, K, edge_mask)

		V = self.out_proj(V)

		return V

	def sample(self, V, temp=1e-6):

		probs = torch.softmax(V, dim=2)
		entropy = -torch.sum(probs*torch.log(probs), dim=2)
		sample = torch.multinomial(probs.view(-1,probs.size(2)), 1).view(probs.size(0), probs.size(1))
		return sample, entropy

	def inference(self, C, L, chain_idxs, node_mask=None, temp=1e-6, cycles=4):

		# initialize boolean mask of fixed positions (where L is not the <mask> token)
		fixed = L != aa_2_lbl("<mask>") # Z x N

		# threshold for rand_val*entropy_normed to update the aas
		update_thresh = 1/cycles

		# get constant values so dont recompute
		C, Ca, Cb = self.featurizer.get_coords(C, chain_idxs, norm=False)
		K, edge_mask = self.featurizer.get_neighbors(Ca, node_mask)
		E = self.featurizer.get_edges(C, K)

		# perform multiple cycles
		for cycle in range(cycles):

			# get nodes
			V = self.featurizer.get_nodes(Ca, Cb, L, node_mask)

			# get seq logits
			L_logits = self.encode(V, E, K, edge_mask)
			
			# sample aas and get entropies
			L_sampled, entropies = self.sample(L_logits, temp=temp)

			# norm entropies so fall in between 0 and 1 for each sample
			entropy_min =  entropies.min(dim=1, keepdim=True).values
			entropy_max = entropies.max(dim=1, keepdim=True).values
			entropies_norm = (entropies - entropy_min) / (entropy_max - entropy_min)

			# decide which aas to update, low entropy vals more likely to be updated, regardless of whether theyve been decoded or not
			update_aa = ((torch.rand_like(L, dtype=torch.float32) * entropies_norm) < update_thresh) & (~fixed)

			# update L
			L = torch.where(update_aa, L_sampled, L)

		# after go through last cycles use the last round of predictions as the final prediction
		L = torch.where(fixed, L, L_sampled)

		return L # dont return entropies yet

